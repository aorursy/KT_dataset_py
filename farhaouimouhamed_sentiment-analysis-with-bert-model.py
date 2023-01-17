import transformers
from tqdm import tqdm
import torch.nn as nn
import pandas as pd 
import torch
from sklearn import model_selection
from sklearn import metrics
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import seaborn as sns
import matplotlib.pyplot as plt


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "../input/bert-base-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768,1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(
            ids = ids,
            mask = mask,
            token_type_ids = token_type_ids
        )

        loss = loss_fn(outputs, targets)
        loss.backward()
        
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_outputs = []
    fin_targets = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids = ids,
                mask = mask,
                token_type_ids = token_type_ids
            )
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
dfx = pd.read_csv(TRAINING_FILE).fillna("none")
dfx.sentiment = dfx.sentiment.apply(
    lambda x: 1 if x == "positive" else 0
)
df_train, df_valid_and_test = model_selection.train_test_split(
    dfx,
    test_size=0.1,
    random_state=42,
    stratify=dfx.sentiment.values
)

df_train = df_train.reset_index(drop=True)
df_valid, df_test = model_selection.train_test_split(df_valid_and_test,test_size=0.5,random_state=42)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

train_dataset = BERTDataset(
    review=df_train.review.values,
    target=df_train.sentiment.values
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=4,
)

valid_dataset = BERTDataset(
    review=df_valid.review.values,
    target=df_valid.sentiment.values
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1,
)
test_dataset = BERTDataset(
    review=df_test.review.values,
    target=df_test.sentiment.values
)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=VALID_BATCH_SIZE,
    num_workers=1,
)

df_train.shape, df_valid.shape, df_test.shape
class_names = ['negative', 'positive']
ax = sns.countplot(dfx.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names);
def run(train_data_loader, valid_data_loader):
    
    device = torch.device("cuda")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.001}
    ]

    num_training_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler) 
        outputs, targets = eval_fn(valid_data_loader, model, device) 
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")

        if accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            print(best_accuracy)
            best_accuracy = accuracy



if __name__ == "__main__":
    run(train_data_loader, valid_data_loader)

