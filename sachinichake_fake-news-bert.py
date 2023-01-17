# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install transformers
import transformers
from tqdm import tqdm
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = "/kaggle/input/bert-base-uncased/"
MODEL_PATH = "/kaggle/input/modelbertv1.pt"
TRAINING_FILE = "/kaggle/input/fake-news-dataset/train.csv"
TESTING_FILE = "/kaggle/input/fake-news-dataset/test.csv" 
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

# /kaggle/input/fake-news-dataset/train.csv
# /kaggle/input/fake-news-dataset/submit.csv
# /kaggle/input/fake-news-dataset/test.csv
# /kaggle/input/bert-base-uncased/vocab.txt
# d/kaggle/input/bert-base-uncase/config.json
# /kaggle/input/bert-base-uncased/pytorch_model.bin



class BERTDataset:
    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())
        # TODO TEXT PROCESSING

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(self.label[item], dtype=torch.float),
        }


def loss_fn(outputs, labels):
    return nn.BCEWithLogitsLoss()(outputs, labels.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        labels = d["labels"]

        # ids = ids.to(device, dtype=torch.long)
        # token_type_ids = token_type_ids.to(device, dtype=torch.long)
        # mask = mask.to(device, dtype=torch.long)
        # labels = labels.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


def eval_fn(data_loader, model, device):
    model.eval()
    fin_labels = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            labels = d["labels"]

            # ids = ids.to(device, dtype=torch.long)
            # token_type_ids = token_type_ids.to(device, dtype=torch.long)
            # mask = mask.to(device, dtype=torch.long)
            # labels = labels.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_labels.extend(labels.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_labels
class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output


def run():
    dfx = pd.read_csv(TRAINING_FILE).fillna("none").reset_index(drop=True)
    dfx.dropna(inplace=True)
    # df_test = pd.read_csv(config.TESTING_FILE).fillna("none").reset_index(drop=True)

    df_train, df_test = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.label.values
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = BERTDataset(
        text=df_train.text.values, label=df_train.label.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4
    )

    test_dataset = BERTDataset(
        text=df_test.text.values, label=df_test.label.values
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=TEST_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    model = BERTBaseUncased()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    best_accuracy = 0
    for epoch in range(EPOCHS):
        train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, labels = eval_fn(test_data_loader, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(labels, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = accuracy


if __name__ == "__main__":
    run()