!pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.4.zip

from fastai.text import *

from tqdm import tqdm_notebook as tqdm



from helperbot import BaseBot, TriangularLR
!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q
data_path = Path(".")
test = pd.read_csv(data_path/"gap-development.tsv", sep="\t")

val = pd.read_csv(data_path/"gap-validation.tsv", sep="\t")

train = pd.read_csv(data_path/"gap-test.tsv", sep="\t")
print(len(train), len(val), len(test))
train["is_valid"] = True

test["is_valid"] = False

val["is_valid"] = True



df_pretrain = pd.concat([train, test, val])
db = (TextList.from_df(df_pretrain, data_path/"db", cols="Text").split_from_df(col="is_valid").label_for_lm().databunch())
vocab = db.vocab
lm = language_model_learner(db, AWD_LSTM, drop_mult=0.5, pretrained=True)
lm.unfreeze()
lm.lr_find()
lm.recorder.plot()
lm.fit_one_cycle(5, 1e-01, moms=(0.8,0.7))
spacy_tok = SpacyTokenizer("en")

tokenizer = Tokenizer(spacy_tok)
df_pretrain.Text.apply(lambda x: len(tokenizer.process_text(x, spacy_tok))).describe()
import spacy

nlp = spacy.blank("en")



def get_token_num_by_offset(s, offset):

  s_pre = s[:offset]

  return len(spacy_tok.tokenizer(s_pre))



# note that 'xxunk' is not special in this sense

special_tokens = ['xxbos','xxfld','xxpad', 'xxmaj','xxup','xxrep','xxwrep']



def collate_examples(batch, truncate_len=500):

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

        return token_tensor, offsets, None

    one_hot_labels = torch.stack([

        torch.from_numpy(x.astype("uint8")) for x in transposed[2]

    ], dim=0)

    _, labels = one_hot_labels.max(dim=1)

    return token_tensor, offsets, labels





def adjust_token_num(processed, token_num):

  """

  As fastai tokenizer introduces additional tokens, we need to adjust for them.

  """

  counter = -1

  do_unrep = None

  for i, token in enumerate(processed):

    if token not in special_tokens:

      counter += 1

    if do_unrep:

      do_unrep = False

      if processed[i+1] != ".":

        token_num -= (int(token) - 2) # one to account for the num itself

      else:  # spacy doesn't split full stops

        token_num += 1

    if token == "xxrep":

      do_unrep = True

    if counter == token_num:

      return i

  else:

    counter = -1

    for i, t in enumerate(processed):

      if t not in special_tokens:

        counter += 1

      print(i, counter, t)

    raise Exception(f"{token_num} is out of bounds ({processed})")
def dataframe_to_tensors(df, max_len=512):

  # offsets are: pron_tok_offset, a_tok_offset, a_tok_right_offset, b_tok_offset, b_tok_right_offset

  offsets = list()

  labels = np.zeros((len(df),), dtype=np.int64)

  processed = list()

  for i, row in tqdm(df.iterrows()):

    try:

      text = row["Text"]

      a_offset = row["A-offset"]

      a_len = len(nlp(row["A"]))

      

      b_offset = row["B-offset"]

      b_len = len(nlp(row["B"]))



      pron_offset = row["Pronoun-offset"]

      is_a = row["A-coref"]

      is_b = row["B-coref"]

      is_female_pronoun = row["Pronoun"]

      if(is_female_pronoun == "her" or is_female_pronoun == "she" or is_female_pronoun == "hers"):

        add_feature = 1

      else:

        add_feature = 0

      a_tok_offset = get_token_num_by_offset(text, a_offset)

      b_tok_offset = get_token_num_by_offset(text, b_offset)

      a_right_offset = a_tok_offset + a_len - 1

      b_right_offset = b_tok_offset + b_len - 1

      pron_tok_offset = get_token_num_by_offset(text, pron_offset)

      tokenized = tokenizer.process_text(text, spacy_tok)[:max_len]

      tokenized = ["xxpad"] * (max_len - len(tokenized)) + tokenized  # add padding

      a_tok_offset = adjust_token_num(tokenized, a_tok_offset)

      a_tok_right_offset = adjust_token_num(tokenized, a_right_offset)

      b_tok_offset = adjust_token_num(tokenized, b_tok_offset)

      b_tok_right_offset = adjust_token_num(tokenized, b_right_offset)

      pron_tok_offset = adjust_token_num(tokenized, pron_tok_offset)

      numericalized = vocab.numericalize(tokenized)

      processed.append(torch.tensor(numericalized, dtype=torch.long))

      offsets.append([pron_tok_offset, a_tok_offset, a_tok_right_offset, b_tok_offset, b_tok_right_offset, add_feature])

      if is_a:

        labels[i] = 0

      elif is_b:

        labels[i] = 1

      else:

        labels[i] = 2

    except Exception as e:

      print(i)

      raise

  processed = torch.stack(processed)

  offsets = torch.tensor(offsets, dtype=torch.long)

  labels = torch.from_numpy(labels)

  return processed, offsets, labels
train_ds = TensorDataset(*dataframe_to_tensors(train))

valid_ds = TensorDataset(*dataframe_to_tensors(val))

test_ds = TensorDataset(*dataframe_to_tensors(test))
train_dl = DataLoader(

    train_ds,

#     collate_fn = collate_examples,

    batch_size=20,

    num_workers=2,

    pin_memory=True,

    shuffle=True,

    drop_last=True)

valid_dl = DataLoader(

    valid_ds,

#     collate_fn = collate_examples,

    batch_size=32,

    num_workers=2,

    pin_memory=True,

    shuffle=True,

    drop_last=True)

test_dl = DataLoader(

    test_ds,

#     collate_fn = collate_examples,

    batch_size=16,

    num_workers=2,

    pin_memory=True,

    shuffle=True,

    drop_last=True)
lm.freeze()
encoder_hidden_sz = 400

import logging



device = torch.device("cuda")



class Head(nn.Module):

    """The MLP submodule"""

    

    def __init__(self, ulm_hidden_size: int):

        super().__init__()

        self.ulm_hidden_size = ulm_hidden_size

        self.fc = nn.Sequential(

            nn.BatchNorm1d(ulm_hidden_size * 6 + 2),

            nn.Dropout(0.5),

            nn.Linear(ulm_hidden_size * 6  + 2, 25),

            nn.ReLU(),

            nn.BatchNorm1d(25),

            nn.Dropout(0.5),

            nn.Linear(25, 3)

        )

        for i, module in enumerate(self.fc):

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):

                nn.init.constant_(module.weight, 1)

                nn.init.constant_(module.bias, 0)

                print("Initing batchnorm")

            elif isinstance(module, nn.Linear):

                if getattr(module, "weight_v", None) is not None:

                    nn.init.uniform_(module.weight_g, 0, 1)

                    nn.init.kaiming_normal_(module.weight_v)

                    print("Initing linear with weight normalization")

                    assert model[i].weight_g is not None

                else:

                    nn.init.kaiming_normal_(module.weight)

                    print("Initing linear")

                nn.init.constant_(module.bias, 0)

    def forward(self, ulm_outputs):

          return self.fc(ulm_outputs)



class CorefResolver(nn.Module):

  def __init__(self, encoder, dropout_p=0.3):

    super(CorefResolver, self).__init__()

    self.device = device

    self.dropout = nn.Dropout(dropout_p)

    self.encoder = encoder

    self.head = Head(encoder_hidden_sz).to(device)

    

  def forward(self, seqs, offsets, labels=None):

    encoded = self.dropout(self.encoder(seqs)[0][2])

    a_q = list()

    b_q = list()

    for enc, offs in zip(encoded, offsets):

    # extract the hidden states that correspond to A, B and the pronoun, and make pairs of those 

        a_repr = enc[offs[2]]

        b_repr = enc[offs[4]]

        a_q.append(torch.cat([enc[offs[0]], a_repr, enc[offs[5]], torch.dot(enc[offs[0]], a_repr).unsqueeze(0)]))

        b_q.append(torch.cat([enc[offs[0]], b_repr, enc[offs[5]], torch.dot(enc[offs[0]], b_repr).unsqueeze(0)]))

    a_q = torch.stack(a_q)

    b_q = torch.stack(b_q)

    return self.head(torch.cat([a_q, b_q], dim=1))
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
enc = lm.model[0]
model = CorefResolver(enc, dropout_p=0.5)
model.to(device)
for param in model.head.parameters():

  param.requires_grad = False
lr = 0.001



loss_fn = nn.NLLLoss()
from sklearn.metrics import classification_report

optimizer = torch.optim.Adam(model.parameters(), lr=lr)



bot = GAPBot(

    model, train_dl, valid_dl,

    optimizer=optimizer, echo=True,

    avg_window=25

)



steps_per_epoch = len(train_dl) 

n_steps = steps_per_epoch * 2

bot.train(

    n_steps,

    log_interval=steps_per_epoch // 4,

    snapshot_interval=steps_per_epoch,

    scheduler=TriangularLR(

        optimizer, 20, ratio=2, steps_per_cycle=n_steps)

)
for param in model.encoder.parameters():

  param.requires_grad = True
lr = 3e-4

optimizer = torch.optim.Adam(model.parameters(), lr=lr)



bot = GAPBot(

    model, train_dl, valid_dl,

    optimizer=optimizer, echo=True,

    avg_window=25

)



steps_per_epoch = len(train_dl) 

n_steps = steps_per_epoch * 2

bot.train(

    n_steps,

    log_interval=steps_per_epoch // 4,

    snapshot_interval=steps_per_epoch,

    scheduler=TriangularLR(

        optimizer, 20, ratio=2, steps_per_cycle=n_steps)

)
bot.load_model(bot.best_performers[0][1])

bot.eval(test_dl)
preds = bot.predict(test_dl)
df_sub = pd.DataFrame(torch.softmax(preds, -1).cpu().numpy().clip(1e-3, 1-1e-3), columns=["A", "B", "NEITHER"])

df_sub["ID"] = test.ID

df_sub.to_csv("submission.csv", index=id)

df_sub.head()
# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
create_download_link(df_sub)