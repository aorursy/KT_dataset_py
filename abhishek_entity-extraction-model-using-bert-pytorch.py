import joblib

import torch

import torch.nn as nn

import transformers



import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn import model_selection



from tqdm import tqdm

from transformers import AdamW

from transformers import get_linear_schedule_with_warmup
class config:

    MAX_LEN = 128

    TRAIN_BATCH_SIZE = 32

    VALID_BATCH_SIZE = 8

    EPOCHS = 3

    BASE_MODEL_PATH = "../input/bert-base-uncased/"

    MODEL_PATH = "model.bin"

    TRAINING_FILE = "../input/entity-annotated-corpus/ner_dataset.csv"

    TOKENIZER = transformers.BertTokenizer.from_pretrained(

        BASE_MODEL_PATH,

        do_lower_case=True

    )
class EntityDataset:

    def __init__(self, texts, pos, tags):

        self.texts = texts

        self.pos = pos

        self.tags = tags

    

    def __len__(self):

        return len(self.texts)

    

    def __getitem__(self, item):

        text = self.texts[item]

        pos = self.pos[item]

        tags = self.tags[item]



        ids = []

        target_pos = []

        target_tag =[]



        for i, s in enumerate(text):

            inputs = config.TOKENIZER.encode(

                s,

                add_special_tokens=False

            )

            # abhishek: ab ##hi ##sh ##ek

            input_len = len(inputs)

            ids.extend(inputs)

            target_pos.extend([pos[i]] * input_len)

            target_tag.extend([tags[i]] * input_len)



        ids = ids[:config.MAX_LEN - 2]

        target_pos = target_pos[:config.MAX_LEN - 2]

        target_tag = target_tag[:config.MAX_LEN - 2]



        ids = [101] + ids + [102]

        target_pos = [0] + target_pos + [0]

        target_tag = [0] + target_tag + [0]



        mask = [1] * len(ids)

        token_type_ids = [0] * len(ids)



        padding_len = config.MAX_LEN - len(ids)



        ids = ids + ([0] * padding_len)

        mask = mask + ([0] * padding_len)

        token_type_ids = token_type_ids + ([0] * padding_len)

        target_pos = target_pos + ([0] * padding_len)

        target_tag = target_tag + ([0] * padding_len)



        return {

            "ids": torch.tensor(ids, dtype=torch.long),

            "mask": torch.tensor(mask, dtype=torch.long),

            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),

            "target_pos": torch.tensor(target_pos, dtype=torch.long),

            "target_tag": torch.tensor(target_tag, dtype=torch.long),

        }
def train_fn(data_loader, model, optimizer, device, scheduler):

    model.train()

    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):

        for k, v in data.items():

            data[k] = v.to(device)

        optimizer.zero_grad()

        _, _, loss = model(**data)

        loss.backward()

        optimizer.step()

        scheduler.step()

        final_loss += loss.item()

    return final_loss / len(data_loader)





def eval_fn(data_loader, model, device):

    model.eval()

    final_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):

        for k, v in data.items():

            data[k] = v.to(device)

        _, _, loss = model(**data)

        final_loss += loss.item()

    return final_loss / len(data_loader)
def loss_fn(output, target, mask, num_labels):

    lfn = nn.CrossEntropyLoss()

    active_loss = mask.view(-1) == 1

    active_logits = output.view(-1, num_labels)

    active_labels = torch.where(

        active_loss,

        target.view(-1),

        torch.tensor(lfn.ignore_index).type_as(target)

    )

    loss = lfn(active_logits, active_labels)

    return loss





class EntityModel(nn.Module):

    def __init__(self, num_tag, num_pos):

        super(EntityModel, self).__init__()

        self.num_tag = num_tag

        self.num_pos = num_pos

        self.bert = transformers.BertModel.from_pretrained(

            config.BASE_MODEL_PATH

        )

        self.bert_drop_1 = nn.Dropout(0.3)

        self.bert_drop_2 = nn.Dropout(0.3)

        self.out_tag = nn.Linear(768, self.num_tag)

        self.out_pos = nn.Linear(768, self.num_pos)

    

    def forward(

        self, 

        ids, 

        mask, 

        token_type_ids, 

        target_pos, 

        target_tag

    ):

        o1, _ = self.bert(

            ids, 

            attention_mask=mask, 

            token_type_ids=token_type_ids

        )



        bo_tag = self.bert_drop_1(o1)

        bo_pos = self.bert_drop_2(o1)



        tag = self.out_tag(bo_tag)

        pos = self.out_pos(bo_pos)



        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)



        loss = (loss_tag + loss_pos) / 2



        return tag, pos, loss
def process_data(data_path):

    df = pd.read_csv(data_path, encoding="latin-1")

    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")



    enc_pos = preprocessing.LabelEncoder()

    enc_tag = preprocessing.LabelEncoder()



    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])

    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])



    sentences = df.groupby("Sentence #")["Word"].apply(list).values

    pos = df.groupby("Sentence #")["POS"].apply(list).values

    tag = df.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, pos, tag, enc_pos, enc_tag
sentences, pos, tag, enc_pos, enc_tag = process_data(config.TRAINING_FILE)



meta_data = {

    "enc_pos": enc_pos,

    "enc_tag": enc_tag

}



joblib.dump(meta_data, "meta.bin")



num_pos = len(list(enc_pos.classes_))

num_tag = len(list(enc_tag.classes_))



(

    train_sentences,

    test_sentences,

    train_pos,

    test_pos,

    train_tag,

    test_tag

) = model_selection.train_test_split(

    sentences, 

    pos, 

    tag, 

    random_state=42, 

    test_size=0.1

)



train_dataset = EntityDataset(

    texts=train_sentences, pos=train_pos, tags=train_tag

)



train_data_loader = torch.utils.data.DataLoader(

    train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4

)



valid_dataset = EntityDataset(

    texts=test_sentences, pos=test_pos, tags=test_tag

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1

)



device = torch.device("cuda")

model = EntityModel(num_tag=num_tag, num_pos=num_pos)

model.to(device)



param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [

    {

        "params": [

            p for n, p in param_optimizer if not any(

                nd in n for nd in no_decay

            )

        ],

        "weight_decay": 0.001,

    },

    {

        "params": [

            p for n, p in param_optimizer if any(

                nd in n for nd in no_decay

            )

        ],

        "weight_decay": 0.0,

    },

]



num_train_steps = int(

    len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS

)

optimizer = AdamW(optimizer_parameters, lr=3e-5)

scheduler = get_linear_schedule_with_warmup(

    optimizer, 

    num_warmup_steps=0, 

    num_training_steps=num_train_steps

)



best_loss = np.inf

for epoch in range(config.EPOCHS):

    train_loss = train_fn(

        train_data_loader, 

        model, 

        optimizer, 

        device, 

        scheduler

    )

    test_loss = eval_fn(

        valid_data_loader,

        model,

        device

    )

    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")

    if test_loss < best_loss:

        torch.save(model.state_dict(), config.MODEL_PATH)

        best_loss = test_loss
meta_data = joblib.load("meta.bin")

enc_pos = meta_data["enc_pos"]

enc_tag = meta_data["enc_tag"]



num_pos = len(list(enc_pos.classes_))

num_tag = len(list(enc_tag.classes_))



sentence = """

abhishek is going to india

"""

tokenized_sentence = config.TOKENIZER.encode(sentence)



sentence = sentence.split()

print(sentence)

print(tokenized_sentence)



test_dataset = EntityDataset(

    texts=[sentence], 

    pos=[[0] * len(sentence)], 

    tags=[[0] * len(sentence)]

)



device = torch.device("cuda")

model = EntityModel(num_tag=num_tag, num_pos=num_pos)

model.load_state_dict(torch.load(config.MODEL_PATH))

model.to(device)



with torch.no_grad():

    data = test_dataset[0]

    for k, v in data.items():

        data[k] = v.to(device).unsqueeze(0)

    tag, pos, _ = model(**data)



    print(

        enc_tag.inverse_transform(

            tag.argmax(2).cpu().numpy().reshape(-1)

        )[:len(tokenized_sentence)]

    )

    print(

        enc_pos.inverse_transform(

            pos.argmax(2).cpu().numpy().reshape(-1)

        )[:len(tokenized_sentence)]

    )