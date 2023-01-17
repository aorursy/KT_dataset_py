import os
import torch
import pandas as pd
from scipy import stats
import numpy as np

from tqdm import tqdm
import torch.nn as nn
import joblib

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
import sys
class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 30)

    def forward(
            self,
            ids,
            mask,
            token_type_ids
    ):
        _, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)

        bo = self.bert_drop(o2)
        p2 = self.out(bo)
        return p2
class BERTDatasetTest:
    def __init__(self, qtitle, qbody, answer, tokenizer, max_length):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        answer_text = str(self.answer[item])

        question_title = " ".join(question_title.split())
        question_body = " ".join(question_body.split())
        answer_text = " ".join(answer_text.split())

        inputs = self.tokenizer.encode_plus(
            question_title + " " + question_body,
            answer_text,
            add_special_tokens=True,
            max_length=self.max_length,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = self.max_length - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }
def predict():
    DEVICE = torch.device("cuda")
    TEST_BATCH_SIZE = 8
    TEST_DATASET = "../input/google-quest-challenge/test.csv"
    df = pd.read_csv(TEST_DATASET).fillna("none")

    qtitle = df.question_title.values.astype(str).tolist()
    qbody = df.question_body.values.astype(str).tolist()
    answer = df.answer.values.astype(str).tolist()
    category = df.category.values.astype(str).tolist()

    tokenizer = transformers.BertTokenizer.from_pretrained("../input/bert-base-uncased/", 
                                                           do_lower_case=True)
    maxlen = 512
    predictions = []

    test_dataset = BERTDatasetTest(
        qtitle=qtitle,
        qbody=qbody,
        answer=answer,
        tokenizer=tokenizer,
        max_length=maxlen
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = BERTBaseUncased("../input/bert-base-uncased/")
    model.to(DEVICE)
    model.load_state_dict(torch.load("../input/tpu-bert/model.bin"))
    model.eval()

    tk0 = tqdm(test_data_loader, total=int(len(test_dataset) / test_data_loader.batch_size))
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]

        ids = ids.to(DEVICE, dtype=torch.long)
        mask = mask.to(DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            outputs = torch.sigmoid(outputs).cpu().numpy()
            predictions.append(outputs)

    return np.vstack(predictions)
preds = predict()
SAMPLE_SUBMISSION = "../input/google-quest-challenge/sample_submission.csv"
sample = pd.read_csv(SAMPLE_SUBMISSION)
target_cols = list(sample.drop("qa_id", axis=1).columns)
sample[target_cols] = preds
sample.head()
sample.to_csv("submission.csv", index=False)