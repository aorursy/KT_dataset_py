!conda remove -y greenlet

!pip install pytorch-pretrained-bert

!pip install allennlp

!pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.5.zip
!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q
import os



# This variable is used by helperbot to make the training deterministic

os.environ["SEED"] = "323"



import logging

from pathlib import Path



import torch

import torch.nn as nn

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import Dataset, DataLoader, TensorDataset

from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel

from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor



from helperbot import (

    TriangularLR, BaseBot, WeightDecayOptimizerWrapper

)
BERT_MODEL = 'bert-large-uncased'

CASED = False
def extract_target(df):

    df["Neither"] = 0

    df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1

    df["target"] = 0

    df.loc[df['B-coref'] == 1, "target"] = 1

    df.loc[df["Neither"] == 1, "target"] = 2

    print(df.target.value_counts())

    return df



df_train = pd.concat([

    pd.read_csv("gap-test.tsv", delimiter="\t"),

    pd.read_csv("gap-validation.tsv", delimiter="\t")

], axis=0)

df_test = pd.read_csv("gap-development.tsv", delimiter="\t")

df_train = extract_target(df_train)

df_test = extract_target(df_test)

sample_sub = pd.read_csv("../input/sample_submission_stage_1.csv")

assert sample_sub.shape[0] == df_test.shape[0]
df_train["text_length"] = df_train.Text.str.len()

df_test["text_length"] = df_test.Text.str.len()

df_train.sort_values("text_length", inplace=True)

df_test.sort_values("text_length", inplace=True)
df_train["A-offset"].max(), df_train["B-offset"].max(), df_train["Pronoun-offset"].max()
def tokenize(row, tokenizer):

    break_points = sorted(

        [

            ("A", row["A-offset"], row["A"]),

            ("B", row["B-offset"], row["B"]),

            ("P", row["Pronoun-offset"], row["Pronoun"]),

        ], key=lambda x: x[0]

    )

    tokens, spans, current_pos = [], {}, 0

    for name, offset, text in break_points:

        tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))

        # Make sure we do not get it wrong

        assert row["Text"][offset:offset+len(text)] == text

        # Tokenize the target

        tmp_tokens = tokenizer.tokenize(row["Text"][offset:offset+len(text)])

        spans[name] = [len(tokens), len(tokens) + len(tmp_tokens) - 1] # inclusive

        tokens.extend(tmp_tokens)

        current_pos = offset + len(text)

    tokens.extend(tokenizer.tokenize(row["Text"][current_pos:offset]))

    assert spans["P"][0] == spans["P"][1]

    return tokens, (spans["A"] + spans["B"] + [spans["P"][0]])





class GAPDataset(Dataset):

    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):

        self.labeled = labeled

        if labeled:

            self.y = df.target.values.astype("uint8")

        

        self.offsets, self.tokens = [], []

        for _, row in df.iterrows():

            tokens, offsets = tokenize(row, tokenizer)

            self.offsets.append(offsets)

            self.tokens.append(tokenizer.convert_tokens_to_ids(

                ["[CLS]"] + tokens + ["[SEP]"]))

        

    def __len__(self):

        return len(self.tokens)



    def __getitem__(self, idx):

        if self.labeled:

            return self.tokens[idx], self.offsets[idx], self.y[idx]

        return self.tokens[idx], self.offsets[idx]



    

def collate_examples(batch, truncate_len=450):

    """Batch preparation.

    

    1. Pad the sequences

    2. Transform the target.

    """    

    transposed = list(zip(*batch))

    max_len = min(

        max((len(x) for x in transposed[0])),

        truncate_len

    )

    tokens = np.zeros((len(batch), max_len), dtype=np.int64)

    for i, row in enumerate(transposed[0]):

        row = np.array(row[:truncate_len])

        tokens[i, :len(row)] = row

    token_tensor = torch.from_numpy(tokens)

    # Offsets

    offsets = torch.stack([

        torch.LongTensor(x) for x in transposed[1]

    ], dim=0) + 1 # Account for the [CLS] token

    # Labels

    if len(transposed) == 2:

        return token_tensor, offsets

    labels = torch.LongTensor(transposed[2])

    return token_tensor, offsets, labels
tokenizer = BertTokenizer.from_pretrained(

    BERT_MODEL,

    do_lower_case=CASED,

    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")

)
class FeatureExtractionModel(nn.Module):

    """To extract features from BERT."""

    def __init__(self, bert_model: str, device: torch.device, use_layer: int = -2):

        super().__init__()

        self.device = device

        self.use_layer = use_layer

        if bert_model in ("bert-base-uncased", "bert-base-cased"):

            self.bert_hidden_size = 768

        elif bert_model in ("bert-large-uncased", "bert-large-cased"):

            self.bert_hidden_size = 1024

        else:

            raise ValueError("Unsupported BERT model.")

        self.span_extractor = EndpointSpanExtractor(

            self.bert_hidden_size, "x+y"

        ).to(device)            

        self.bert = BertModel.from_pretrained(bert_model).to(device)

    

    def forward(self, token_tensor, offsets):

        token_tensor = token_tensor.to(self.device)

        offsets = offsets.to(self.device)

        bert_outputs, _ =  self.bert(

            token_tensor, attention_mask=(token_tensor > 0).long(), 

            token_type_ids=None, output_all_encoded_layers=True)

        bert_outputs = bert_outputs[self.use_layer]

        spans_contexts = self.span_extractor(

            bert_outputs, 

            offsets[:, :4].reshape(-1, 2, 2)

        )

        emb_P = torch.gather(

            bert_outputs, 1,

            offsets[:, [4]].unsqueeze(2).expand(-1, -1, self.bert_hidden_size)

        ).squeeze(1)

        return spans_contexts, emb_P
model = FeatureExtractionModel(BERT_MODEL, torch.device("cuda:0"), use_layer=-2)

# Make it deterministic

_ = model.eval()
def extract_features(loader):

    spc, embp = [], []

    with torch.no_grad():

        for token, offsets in tqdm_notebook(loader):

            spans_contexts, emb_P = model(token, offsets)

            spc.append(spans_contexts.cpu())

            embp.append(emb_P.cpu())

    return torch.cat(spc, dim=0), torch.cat(embp, dim=0)
train_ds = GAPDataset(df_train, tokenizer, labeled=False)

train_loader = DataLoader(

    train_ds,

    collate_fn = collate_examples,

    batch_size=128,

    num_workers=2,

    pin_memory=True,

    shuffle=False

)

spc_train, embp_train = extract_features(train_loader)

ys_train = torch.from_numpy(df_train.target.values)

spc_train.size(), embp_train.size(), ys_train.size()
test_ds = GAPDataset(df_test, tokenizer, labeled=False)

test_loader = DataLoader(

    test_ds,

    collate_fn = collate_examples,

    batch_size=128,

    num_workers=2,

    pin_memory=True,

    shuffle=False

)

spc_test, embp_test = extract_features(test_loader)

ys_test = torch.from_numpy(df_test.target.values)

spc_test.size(), embp_test.size(), ys_test.size()
torch.save([spc_train, embp_train, ys_train], "train.pkl")

torch.save([spc_test, embp_test, ys_test], "test.pkl")
spc_train, embp_train, ys_train = torch.load("train.pkl")

spc_test, embp_test, ys_test = torch.load("test.pkl")
assert np.array_equal(df_test.target, ys_test.numpy())

assert np.array_equal(df_train.target, ys_train.numpy())
class GAPBot(BaseBot):

    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,

        avg_window=100, log_dir="./cache/logs/", log_level=logging.INFO,

        checkpoint_dir="./cache/model_cache/", batch_idx=0, echo=False,

        device="cuda:0", use_tensorboard=False):

        super().__init__(

            model, train_loader, val_loader, 

            optimizer=optimizer, clip_grad=clip_grad,

            log_dir=log_dir, checkpoint_dir=checkpoint_dir, 

            batch_idx=batch_idx, echo=echo,

            device=device, use_tensorboard=use_tensorboard

        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.loss_format = "%.6f"

        

    def extract_prediction(self, tensor):

        return tensor

    

    def snapshot(self):

        """Override the snapshot method because Kaggle kernel has limited local disk space."""

        loss = self.eval(self.val_loader)

        loss_str = self.loss_format % loss

        self.logger.info("Snapshot loss %s", loss_str)

        self.logger.tb_scalars(

            "losses", {"val": loss},  self.step)

        target_path = (

            self.checkpoint_dir / "best.pth")        

        if not self.best_performers or (self.best_performers[0][0] > loss):

            torch.save(self.model.state_dict(), target_path)

            self.best_performers = [(loss, target_path, self.step)]

        self.logger.info("Saving checkpoint %s...", target_path)

        assert Path(target_path).exists()

        return loss    
class GAPMlpModel(nn.Module):

    """The MLP submodule"""

    def __init__(self, span_dim: int, bert_hidden_size: int, device: torch.device):

        super().__init__()

        self.device = device

        self.fc = nn.Sequential(

            nn.BatchNorm1d(span_dim * 2 + bert_hidden_size),

            nn.Dropout(0.5),

            nn.Linear(span_dim * 2 + bert_hidden_size, 128),           

            nn.ReLU(),

            nn.BatchNorm1d(128),             

            nn.Dropout(0.5),

            nn.Linear(128, 128),           

            nn.ReLU(),

            nn.BatchNorm1d(128),             

            nn.Dropout(0.5),                           

#             nn.Linear(128, 128),           

#             nn.ReLU(),

#             nn.BatchNorm1d(128),             

#             nn.Dropout(0.5),                        

            nn.Linear(128, 3)

        ).to(device)

        for i, module in enumerate(self.fc):

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):

                nn.init.constant_(module.weight, 1)

                nn.init.constant_(module.bias, 0)

                print("Initing batchnorm")

            elif isinstance(module, nn.Linear):

                nn.init.kaiming_normal_(module.weight)

                print("Initing linear")

                nn.init.constant_(module.bias, 0)

                

    def forward(self, spans_contexts, emb_P):

        return self.fc(torch.cat([

            spans_contexts.reshape(spans_contexts.size(0), -1), emb_P

        ], dim=1))
test_ds = TensorDataset(spc_test, embp_test, ys_test)

test_loader = DataLoader(

    test_ds,

    batch_size=128,

    num_workers=2,

    pin_memory=True,

    shuffle=False

)
skf = StratifiedKFold(n_splits=5, random_state=234)



val_preds, test_preds, val_ys, val_losses = [], [], [], []

for train_index, valid_index in skf.split(df_train, ys_train.numpy()):

    print("=" * 20)

    print(f"Fold {len(val_preds) + 1}")

    print("=" * 20)

    train_ds = TensorDataset(spc_train[train_index], embp_train[train_index], ys_train[train_index])

    val_ds = TensorDataset(spc_train[valid_index], embp_train[valid_index], ys_train[valid_index])

    train_loader = DataLoader(

        train_ds,

        batch_size=32,

        num_workers=0,

        pin_memory=True,

        shuffle=True,

        drop_last=True

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=128,

        num_workers=0,

        pin_memory=True,

        shuffle=False

    )

    model = GAPMlpModel(spc_train.size(2), embp_train.size(1), torch.device("cuda:0"))

    optimizer = WeightDecayOptimizerWrapper(

        torch.optim.Adam(model.parameters(), lr=1e-3),

        0.05

    )

#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)

    bot = GAPBot(

        model, train_loader, val_loader,

        optimizer=optimizer, echo=False,

        avg_window=50

    )

    steps_per_epoch = len(train_loader) 

    n_steps = steps_per_epoch * 30

    bot.train(

        n_steps,

        log_interval=steps_per_epoch // 1,

        snapshot_interval=steps_per_epoch // 1,

        scheduler=TriangularLR(

            optimizer, 100, ratio=2, steps_per_cycle=n_steps)

    )

    # Load the best checkpoint

    bot.load_model(bot.best_performers[0][1])

    bot.remove_checkpoints(keep=0)    

    val_preds.append(torch.softmax(bot.predict(val_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())

    val_ys.append(df_train.iloc[valid_index].target.astype("uint8").values)

    val_losses.append(log_loss(val_ys[-1], val_preds[-1]))

    bot.logger.info("Confirm val loss: %.4f", val_losses[-1])

    test_preds.append(torch.softmax(bot.predict(test_loader), -1).clamp(1e-4, 1-1e-4).cpu().numpy())
val_losses
final_test_preds = np.mean(test_preds, axis=0)

# final_test_preds = torch.stack(test_preds, dim=0).mean(dim=0).numpy()

final_test_preds.shape
log_loss(df_test.target, final_test_preds)
# Create submission file

df_sub = pd.DataFrame(final_test_preds, columns=["A", "B", "NEITHER"])

df_sub["ID"] = df_test.ID.values

df_sub.to_csv("submission.csv", index=False)

df_sub.head()