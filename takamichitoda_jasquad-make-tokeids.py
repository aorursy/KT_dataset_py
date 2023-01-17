!yes | pip install transformers==2.10.0



!apt install aptitude -y

!aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y

!pip install mecab-python3==0.996.6rc2



!pip install unidic-lite
import pandas as pd



import MeCab

import tqdm

import pickle



import torch

import torch.nn as nn



from transformers import *

import tokenizers
INPUT = "/kaggle/input/squad-japanese/"

MAX_LEN = 512



TOKENIZER = BertJapaneseTokenizer.from_pretrained("bert-base-japanese")

TOKENIZER.save_pretrained("./")

TOKENIZER = tokenizers.BertWordPieceTokenizer("./vocab.txt", lowercase=True)
def process_data(question, context, answer, id_):

    len_st = len(answer)



    for ind in (i for i, e in enumerate(context) if e == answer[0]):

        if context[ind: ind+len_st] == answer:

            idx0 = ind

            idx1 = ind + len_st - 1

            break

        

    char_targets = [0] * len(context)

    for ct in range(idx0, idx1 + 1):

        char_targets[ct] = 1



    tok = TOKENIZER.encode(context)

    c_ids = tok.ids

    offsets = tok.offsets



    target_idx = []

    for j, (offset1, offset2) in enumerate(offsets):

        if sum(char_targets[offset1: offset2]) > 0:

            target_idx.append(j)



    targets_start = target_idx[0]

    targets_end = target_idx[-1]+1



    q_ids = TOKENIZER.encode(question).ids

    q_ids = q_ids[1:-1]



    q_len = (len(q_ids)+2)



    input_ids = [2] + q_ids + [3] + c_ids[1:]

    token_type_ids = q_len*[0] + (len(c_ids)-1)*[1]

    mask = [1] * len(token_type_ids)



    targets_start += q_len-1

    targets_end += q_len-1

    offsets = [(0, 0)] * (q_len-1)  + offsets



    padding_length = MAX_LEN - len(input_ids)

    if padding_length < 0:

        return

    

    input_ids = input_ids + ([1] * padding_length)

    mask = mask + ([0] * padding_length)

    token_type_ids = token_type_ids + ([0] * padding_length)

    offsets = offsets + ([(0, 0)] * padding_length) 

    

    return {

        'ids': input_ids,

        'mask': mask,

        'token_type_ids': token_type_ids,

        'targets_start': targets_start,

        'targets_end': targets_end,

        'offsets': offsets,

        'uuid': id_

    }
class JaSQuADDataset:

    def __init__(self, data):

        self.data = data

        

    def __len__(self):

        return len(self.data)



    def __getitem__(self, item):

        data = self.data[item]

        return {

            'ids': torch.tensor(data["ids"], dtype=torch.long),

            'mask': torch.tensor(data["mask"], dtype=torch.long),

            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),

            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),

            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),

            'offsets': torch.tensor(data["offsets"], dtype=torch.long),

            'uuid': data["uuid"]

        }
def generate_processed_data(df):

    lst = []

    for idx in tqdm.notebook.tqdm(range(len(df))):

        context = df.loc[idx, "context_join"]

        answer = df.loc[idx, "answer"]

        question = df.loc[idx, "question_join"]

        id_ = df.loc[idx, "id"]

    

        data = process_data(question, context, answer, id_)

        if data is None:

            continue

        lst.append(data)

    return lst
train_df = pd.read_json(f"{INPUT}/train.jsonl", orient='records', lines=True)



train_df = train_df[train_df.apply(lambda x: x["end"] - x["start"], axis=1) != 0].reset_index(drop=True)



train_df["question_join"] = train_df["question"].map(lambda x: "".join(x.split()))

train_df["context_join"] = train_df["context"].map(lambda x: "".join(x.split()))

train_df["answer"] = train_df.apply(lambda x: "".join(x["context"].split()[x["start"]:x["end"]]), axis=1)



train_df = train_df[["id", "question_join", "context_join", "answer"]]

train_df.head()
valid_df = pd.read_json(f"{INPUT}/valid.jsonl", orient='records', lines=True)

valid_df = valid_df[valid_df.apply(lambda x: x["end"] - x["start"], axis=1) != 0].reset_index(drop=True)



valid_df["question_join"] = valid_df["question"].map(lambda x: "".join(x.split()))

valid_df["context_join"] = valid_df["context"].map(lambda x: "".join(x.split()))

valid_df["answer"] = valid_df.apply(lambda x: "".join(x["context"].split()[x["start"]:x["end"]]), axis=1)



valid_df = valid_df[["id", "question_join", "context_join", "answer"]]

valid_df.head()
train_data = generate_processed_data(train_df)

valid_data = generate_processed_data(valid_df)
class JaSQuADBert(nn.Module):

    def __init__(self):

        super(JaSQuADBert, self).__init__()



        self.bert = BertModel.from_pretrained("bert-base-japanese-whole-word-masking")

        self.qa_outputs = nn.Linear(768, 2)



    

    def forward(self, ids, mask, token_type_ids):

        out, _ = self.bert(

            ids,

            attention_mask=mask,

            token_type_ids=token_type_ids

        )

        logits = self.qa_outputs(out)

        

        start_logits, end_logits = logits.split(1, dim=-1)



        start_logits = start_logits.squeeze(-1)

        end_logits = end_logits.squeeze(-1)



        return start_logits, end_logits

    

    

def loss_fn(start_logits, end_logits, start_positions, end_positions):

    loss_fct = nn.CrossEntropyLoss()

    start_loss = loss_fct(start_logits, start_positions)

    end_loss = loss_fct(end_logits, end_positions)

    

    total_loss = start_loss + end_loss



    return total_loss
valid_dataset = JaSQuADDataset(valid_data)

valid_data_loader = torch.utils.data.DataLoader(

        valid_dataset,

        batch_size=2,

        drop_last=True,

        num_workers=2

)
model = JaSQuADBert()

model.eval()

for d in valid_data_loader:

    ids = d["ids"]

    mask = d["mask"]

    token_type_ids = d["token_type_ids"]

    start_positions = d["targets_start"]

    end_positions = d["targets_end"]

    uuid = d["uuid"]

    

    outputs_start, outputs_end = model(ids, mask, token_type_ids)

    loss = loss_fn(outputs_start, outputs_end, start_positions, end_positions)

    print(uuid)

    print(loss.item())

    break
with open("squad_train_data.pkl", "wb") as f:

    pickle.dump(train_data, f)

    

with open("squad_valid_data.pkl", "wb") as f:

    pickle.dump(valid_data, f)
!ls