!pip install transformers
import pandas as pd

import numpy as np

import re

from tqdm.notebook import tqdm

import torch

import copy

import matplotlib.pyplot as plt

from transformers import BertTokenizer, BertModel



tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

bert = BertModel.from_pretrained("bert-base-multilingual-cased")
def read_data_2(file_name):

    """

    - Input: crawled data in csv format - Output: sentences

    - punctual_res=False : only reads the text normalization labels

    - punctual_res=True : reads also the punctual restoration labels

    """

    p = re.compile(".*[,].*")

    sentences = []

    with open(file_name, 'r', encoding="utf-8") as f:

        sentence = []

        for line in f:

            parts = line[line.index(',')+1:].split()

            if p.match(parts[0]): 

                if parts[0].split(',')[-1] == 'PERIOD':

                    if len(sentence) > 0: 

                        sentence = [word[0].split(',') for word in sentence]

                        sentences.append(sentence)

                    sentence = []

                elif len(parts[0].split(',')) == 3:

                    if parts[0][:parts[0].index(",")] not in ['&', '*', '>>', '+', '=']: 

                        rev_parts = list(reversed(list(parts[0])))

                        index = len(parts[0]) - rev_parts.index(",") - 1

                        sentence.append([parts[0][:index]])

        if len(sentence) > 0: 

            sentence = [word[0].split(',') for word in sentence]

            sentences.append(sentence)

    return sentences





def encode_data(data):

    """Tokenize words and convert their BIO labels to numerical values"""

    results = []

    for sentence in tqdm(data):

        input_ids = []

        label_ids = []

        for word, label in sentence:

            subwords = tokenizer.encode(word, add_special_tokens=False)

            input_ids += subwords

            label_ids += [labels.index(label)] + [-100]*(len(subwords) - 1)

        results.append((input_ids, label_ids))

    return results
bio = read_data_2("../input/vnexpress-bio-1/vnexpress_bio_1.csv")
labels = ["B-CAP","I-CAP","B-NUMB","I-NUMB","B-DATE","I-DATE","O"]



encoded_bio = encode_data(bio)



train_set = encoded_bio[len(encoded_bio)//5:]

test_set = encoded_bio[:len(encoded_bio)//5]



print(train_set[0][0])

print(train_set[0][1])

print("Train size: {0} - Test size: {1}".format(len(train_set), len(test_set)))
from torch.utils.data import DataLoader



class TextDataLoader(DataLoader):

    def __init__(self, data_set, shuffle=False, device="cuda", batch_size=16, max_seq_len=512):

        super(TextDataLoader, self).__init__(dataset=data_set, collate_fn=self.collate_fn, shuffle=shuffle, batch_size=batch_size)

        self.device = device

        self.max_seq_len = max_seq_len



    def collate_fn(self, data):

        examples = []

        max_length = max(map(lambda x: len(x[0]), data)) + 2

        if max_length > self.max_seq_len: 

            max_length = self.max_seq_len

            for input_ids, label_ids in data:

                input_ids = input_ids[:max_length-2]

                label_ids = label_ids[:max_length-2]

                input_ids = [101] + input_ids + [102]

                label_ids = [-100] + label_ids + [-100]



                attention_mask_ids = [1] * len(input_ids)

                attention_mask_ids += [0] * (max_length - len(attention_mask_ids))

                input_ids += [0] * (max_length - len(input_ids))

                token_type_ids = [0] * max_length

                label_ids += [-100] * (max_length - len(label_ids))



                example = []

                example.append(input_ids)

                example.append(attention_mask_ids)

                example.append(token_type_ids)

                example.append(label_ids)

                examples.append(example)

        else: 

            for input_ids, label_ids in data:

                input_ids = [101] + input_ids + [102]

                label_ids = [-100] + label_ids + [-100]



                attention_mask_ids = [1] * len(input_ids)

                attention_mask_ids += [0] * (max_length - len(attention_mask_ids))

                input_ids += [0] * (max_length - len(input_ids))

                token_type_ids = [0] * max_length

                label_ids += [-100] * (max_length - len(label_ids))



                example = []

                example.append(input_ids)

                example.append(attention_mask_ids)

                example.append(token_type_ids)

                example.append(label_ids)

                examples.append(example)



        result = []

        for sample in zip(*examples):

            result.append(torch.LongTensor(sample).to(self.device))

        return result
train_loader = TextDataLoader(train_set, shuffle=False, device="cuda", batch_size=8)
class Model(torch.nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased') 

        self.gru = torch.nn.GRU(768, 512//2, bidirectional=True)

        self.decoder = torch.nn.Linear(512, 7)



    def forward(self, input_ids, attention_mask_ids, token_type_ids, label_ids=None):

        features, _ = self.encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask_ids)

        X, _ = self.gru(features)

        X = self.decoder(X)



        loss_fct = torch.nn.CrossEntropyLoss()

        

        if label_ids is not None:

            X = X.reshape(-1, 7)

            label_ids = label_ids.reshape(-1,)

            loss = loss_fct(X, label_ids)

            return loss

        else: return torch.argmax(X, -1)



model = Model()

model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

print(optimizer)
accs = []

losses = []

n_epoch = 1

for epoch in range(1, n_epoch+1):

    model.train()

    for batch in tqdm(train_loader):

        model.zero_grad()

        input_ids = batch[0]

        attention_mask_ids = batch[1]

        token_type_ids = batch[2]

        label_ids = batch[3]

        loss = model(input_ids, attention_mask_ids, token_type_ids, label_ids)

        losses.append([epoch, loss])

        loss.backward()

        optimizer.step()

        

        model.eval()

        outputs = model(input_ids, attention_mask_ids, token_type_ids)

        

        output_list = [pd.Series(output) for output in outputs.cpu()]

        label_ids_list = [pd.Series(label_id) for label_id in label_ids.cpu()]

        outputs_series = pd.concat(output_list)

        label_ids_series = pd.concat(label_ids_list)



        acc = sum(outputs_series == label_ids_series)/sum(label_ids_series != -100) * 100

        accs.append([epoch, acc])

        

        model.train()
acc_e1 = [acc[1] for acc in accs if acc[0] == 1]

acc_e2 = [acc[1] for acc in accs if acc[0] == 2]

acc_e3 = [acc[1] for acc in accs if acc[0] == 3]

acc_e4 = [acc[1] for acc in accs if acc[0] == 4]

acc_e5 = [acc[1] for acc in accs if acc[0] == 5]



loss_e1 = [loss[1] for loss in losses if loss[0] == 1]

loss_e2 = [loss[1] for loss in losses if loss[0] == 2]

loss_e3 = [loss[1] for loss in losses if loss[0] == 3]

loss_e4 = [loss[1] for loss in losses if loss[0] == 4]

loss_e5 = [loss[1] for loss in losses if loss[0] == 5]



# Plotting accuracy scores

fig = plt.figure(figsize=(20,12))



# Every step

ax1 = fig.add_subplot(2,1,1)

ax1.margins(0.05)

ax1.plot(acc_e1, color='blue', linewidth=2, label='Epoch 1')

ax1.plot(acc_e2, color='yellow', linewidth=2, label='Epoch 2')

ax1.plot(acc_e3, color='olive', linewidth=2, label='Epoch 3')

ax1.plot(acc_e4, color='purple', linewidth=2, label='Epoch 4')

ax1.plot(acc_e5, color='green', linewidth=2, label='Epoch 5')



ax1.set_title("Trainset: Accuracy scores every step", size=20)

ax1.set_xlabel("Steps", size=20)

ax1.set_ylabel("Score (max 100)", size=20)

ax1.legend()



ax2 = fig.add_subplot(2,1,2)

ax2.margins(0.05)

ax2.plot(loss_e1, color='blue', linewidth=2, label='Epoch 1')

ax2.plot(loss_e2, color='yellow', linewidth=2, label='Epoch 2')

ax2.plot(loss_e3, color='olive', linewidth=2, label='Epoch 3')

ax2.plot(loss_e4, color='purple', linewidth=2, label='Epoch 4')

ax2.plot(loss_e5, color='green', linewidth=2, label='Epoch 5')



ax2.set_title("Trainset: Loss every step", size=20)

ax2.set_xlabel("Steps", size=20)

ax2.set_ylabel("Score (max 1)", size=20)

ax2.legend()



fig.tight_layout()

plt.margins(x=0)

plt.show()
# test_loader = TextDataLoader(test_set, shuffle=False, device="cuda", batch_size=8)
accs_test = []

losses_test = []

for batch in test_loader:

    model.zero_grad()

    input_ids = batch[0]

    attention_mask_ids = batch[1]

    token_type_ids = batch[2]

    label_ids = batch[3]

    

    model.eval()

    outputs = model(input_ids, attention_mask_ids, token_type_ids)

    loss = model(input_ids, attention_mask_ids, token_type_ids, label_ids)

    losses_test.append(loss)

    

    output_list = [pd.Series(output) for output in outputs.cpu()]

    label_ids_list = [pd.Series(label_id) for label_id in label_ids.cpu()]

    outputs_series = pd.concat(output_list)

    label_ids_series = pd.concat(label_ids_list)

    

    acc = sum(outputs_series == label_ids_series)/sum(label_ids_series != -100) * 100

    accs_test.append(acc)
# Plotting accuracy scores

fig = plt.figure(figsize=(20,12))



# Every step

ax = fig.add_subplot(2,1,1)

ax.margins(0.05)

ax.plot(accs_test, color='blue', linewidth=2)



ax.set_title("Testset: Accuracy scores every step", size=20)

ax.set_xlabel("Steps", size=20)

ax.set_ylabel("Score (max 100)", size=20)

ax.legend()



ax2 = fig.add_subplot(2,1,2)

ax2.margins(0.05)

ax2.plot(losses_test, color='red', linewidth=2)



ax2.set_title("Testset: Loss every step", size=20)

ax2.set_xlabel("Steps", size=20)

ax2.set_ylabel("Score (max 1)", size=20)

ax2.legend()



fig.tight_layout()

plt.margins(x=0)

plt.show()