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
!cd /kaggle/input/apexpytorch/ && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
import unicodedata

from tqdm import tqdm



def normalize(s):

    s = str(s)

    #s = unicodedata.normalize('NFKC', str(s))

    #s = s.replace('`', "'")

    return s.strip()



def read_dataset(file_name):

    data = pd.read_csv(file_name)

    dataset = []

    for i in tqdm(range(len(data))):

        if "selected_text" in data:

            items = [data['text'][i], data['selected_text'][i], data['sentiment'][i]]

        else:

            items = [data['text'][i], "NO_DATA", data['sentiment'][i]]

        textID = data['textID'][i]

        items = [normalize(item) for item in items]

        text = items[0]

        selected_text = items[1]

        label = items[2]

        words = text.split()

        selected_text_len = len(selected_text.split())

        sentence = []

        i = 0

        while i < len(words):

            word = words[i]

            next_sentence = " ".join(words[i:i+selected_text_len])

            if next_sentence == selected_text:

                sentence.append((words[i], "B-"+label))

                for _word in words[i+1:i+selected_text_len]:

                    sentence.append((_word, "I-"+label))

                i += selected_text_len

            else:

                sentence.append((word, "O"))

                i += 1

        

        dataset.append({"sentence": sentence, "label": label, "textID": textID})

    return dataset



trainset = read_dataset('/kaggle/input/tweet-sentiment-extraction/train.csv')

testset = read_dataset('/kaggle/input/tweet-sentiment-extraction/test.csv')

print("trainset:", len(trainset))

print("testset:", len(testset))

from transformers import BertTokenizer, BertModel

import torch

tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bert-base-uncased/')

print(tokenizer.encode("hello my world"))
labels = ["O", "B-neutral", "I-neutral", "B-negative", "I-negative", "B-positive", "I-positive"]

def encode_dataset(dataset, ignore_index=-100, max_seq_len=512):

    intent_labels = ["neutral", "negative", "positive"]

    data_inputs = []

    for data in tqdm(dataset):

        input_ids = []

        label_ids = []

        token_type_ids = []

        for word, label in data["sentence"]:

            ids = tokenizer.encode(word, add_special_tokens=False)

            input_ids += ids

            label_ids += [labels.index(label)] + [ignore_index]*(len(ids) - 1)

            

        token_type_ids = [0]* (len(input_ids)+1)

        ids = tokenizer.encode(data["label"], add_special_tokens=False)

        input_ids += [tokenizer.sep_token_id] + ids

        label_ids += [ignore_index]*(len(ids)+1)

        

        input_ids = input_ids[:max_seq_len-2]

        label_ids = label_ids[:max_seq_len-2]

        token_type_ids = token_type_ids[:max_seq_len]

        

        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        label_ids = [ignore_index] + label_ids + [ignore_index]

        

        mask_ids = [1] * len(input_ids) + [0] * (max_seq_len - len(input_ids))

        input_ids += [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))

        token_type_ids += [1] * (max_seq_len - len(token_type_ids))

        label_ids += [ignore_index] * (max_seq_len - len(label_ids))

        

        data_inputs.append((input_ids, mask_ids, token_type_ids, label_ids, data))

    return data_inputs



train_ids = encode_dataset(trainset)

test_ids = encode_dataset(testset)

print("train_ids:", len(train_ids))

print("test_ids:", len(test_ids))
from torch.utils.data import DataLoader



class TextDataLoader(DataLoader):

    def __init__(self, data_set, shuffle=False, device="cuda", batch_size=16):

        super(TextDataLoader, self).__init__(dataset=data_set, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)

        self.device = device



    def collate_fn(self, data):

        examples = []

        data_infor = []

        max_length = max(map(lambda x: sum(x[1]), data))

        for sample in data:

            example = []

            example.append(sample[0][:max_length])

            example.append(sample[1][:max_length])

            example.append(sample[2][:max_length])

            example.append(sample[3][:max_length])

            data_infor.append(sample[4])

            examples.append(example)

        result = []

        for sample in zip(*examples):

            result.append(torch.LongTensor(sample).to(self.device))

        result.append(data_infor)

        return result
class Model(torch.nn.Module):

    def __init__(self, decoder_output_dim, dropout_rate=0.5, device="cuda"):

        super(Model, self).__init__()

        self.encoder = BertModel.from_pretrained('/kaggle/input/bert-base-uncased/')

        encoder_output_dim = self.encoder.config.hidden_size

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.decoder = torch.nn.Linear(encoder_output_dim, decoder_output_dim)

        self.output_dim = decoder_output_dim

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.to(device)



    def forward(self, input_ids, mask_ids, token_type_ids, label_ids=None):

        bert_output = self.encoder(input_ids, mask_ids, token_type_ids)[0]

        decoder_input = self.dropout(bert_output)

        logits = self.decoder(decoder_input)

        if label_ids is not None:

            return self.loss_fct(logits.view(-1, self.output_dim), label_ids.view(-1))

        else:

            return logits
model = Model(len(labels))

print(model)
from transformers import AdamW



def get_optimizer(model, bert_lr, lr, bert_weight_decay=0.05, adam_epsilon=1e-8):

    optimizer_grouped_parameters = []

    for n, p in model.named_parameters():

        optimizer_params = {"params": p}

        if "encoder" in n:

            optimizer_params["lr"] = bert_lr

            if any(x in n for x in ['bias', 'LayerNorm.weight']):

                optimizer_params["weight_decay"] = 0

            else:

                optimizer_params["weight_decay"] = bert_weight_decay

        else:

            optimizer_params["lr"] = lr

        optimizer_grouped_parameters.append(optimizer_params)

    return AdamW(optimizer_grouped_parameters, eps=adam_epsilon)
optimizer = get_optimizer(model, 2e-5, 0.001)

print(optimizer)
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=10000)
def train(model, optimizer, scheduler, data_loader, epochs=10):

    model.train()

    for epoch in range(epochs):

        for step, batch in enumerate(data_loader):

            input_ids = batch[0]

            mask_ids = batch[1]

            token_type_ids = batch[2]

            label_ids = batch[3]

            loss = model(input_ids, mask_ids, token_type_ids, label_ids)

            print(f"Epoch: {epoch} - step: {step} - loss: {loss.item()}", end="\n" if step%100==0 else "\r")

            with amp.scale_loss(loss, optimizer) as scaled_loss:

                scaled_loss.backward()

            optimizer.step()

            scheduler.step()

            model.zero_grad()
def eval(model, data_loader, ignore_index=-100):

    results = []

    model.eval()

    for step, batch in enumerate(data_loader):

        input_ids = batch[0]

        mask_ids = batch[1]

        token_type_ids = batch[2]

        label_ids = batch[3].cpu().data.numpy()

        data_infor = batch[4]

        logits = model(input_ids, mask_ids, token_type_ids)

        logits = torch.argmax(logits, -1).cpu().data.numpy()

        for infor, predicted_label_ids, target_label_ids in zip(data_infor, logits, label_ids):

            predicted_labels = []

            for predicted_label_id, target_label_id in zip(predicted_label_ids, target_label_ids):

                if target_label_id != ignore_index:

                    predicted_labels.append(labels[predicted_label_id])

            

            result = []

            for word, predicted_label in zip(infor['sentence'], predicted_labels):

                if predicted_label.endswith(infor['label']):

                    result.append(word[0])

                else:

                    if len(result) > 0:

                        break

            if len(result) == 0:

                for word, predicted_label in zip(infor['sentence'], predicted_labels):

                    if predicted_label != "O":

                        result.append(word[0])

                    else:

                        if len(result) > 0:

                            break

            if len(result) == 0:

                for word, predicted_label in zip(infor['sentence'], predicted_labels):

                    result.append(word[0])

            results.append((infor["textID"], " ".join(result)))

    return results
train_loader = TextDataLoader(train_ids, shuffle=True, batch_size=32)

train(model, optimizer, scheduler, train_loader, epochs=50)
test_loader = TextDataLoader(test_ids, shuffle=False, batch_size=32)

results = eval(model, test_loader)
print(len(results))
submission = pd.DataFrame(results, columns=["textID", "selected_text"])

submission.to_csv("submission.csv", index=False)

submission