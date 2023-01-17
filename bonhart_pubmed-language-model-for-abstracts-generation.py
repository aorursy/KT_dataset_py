import pandas as pd



df = pd.read_csv("/kaggle/input/pubmed-abstracts/pubmed_abstracts.csv", index_col=0)

df.tail()
import re



# function for removing html tags

def preprocess_text(text):

    text = text.str.replace(r"[^a-zA-Zа-яА-Я1-9]+", ' ') # remove () & []

    text = text.str.replace(r"\#","") # replaces hashtags

    text = text.str.replace(r"http\S+","URL")  # remove URL addresses

    text = text.str.replace(r"@","")

    text = text.str.replace("\s{2,}", " ")

    return text







df = df[df.columns.tolist()[:8]].apply(lambda x: preprocess_text(x))

df.head()
# Data

dl = df.deep_learning.dropna()

covid_19 = df.covid_19.dropna()

human_connectome = df.human_connectome.dropna()

virtual_reality = df.virtual_reality.dropna()

brain_machine_interfaces = df.brain_machine_interfaces.dropna()

electroactive_polymers = df.electroactive_polymers.dropna()

pedot_electrodes = df.pedot_electrodes.dropna()

neuroprosthetics = df.neuroprosthetics.dropna()



data = list(pd.concat([dl, covid_19, 

                        human_connectome, virtual_reality, 

                        brain_machine_interfaces, electroactive_polymers, 

                        pedot_electrodes, neuroprosthetics]).unique())





print(data[:1])



# array of texts

new_df = pd.DataFrame({"data": data})

new_df.head()
from torchtext.data import Field



text_field = Field(init_token='<s>', eos_token='</s>', lower=True, tokenize=lambda line: list(line))



text_field.preprocess(data[0])
import matplotlib.pyplot as plt

plt.style.use("dark_background")



df = df.fillna('')

dl_lines = df.apply(lambda row: text_field.preprocess(row['deep_learning']), axis=1).tolist()

c_lines = df.apply(lambda row: text_field.preprocess(row['covid_19']), axis=1).tolist()

hc_lines = df.apply(lambda row: text_field.preprocess(row['human_connectome']), axis=1).tolist()

vr_lines = df.apply(lambda row: text_field.preprocess(row['virtual_reality']), axis=1).tolist()

bmi_lines = df.apply(lambda row: text_field.preprocess(row['brain_machine_interfaces']), axis=1).tolist()

ep_lines = df.apply(lambda row: text_field.preprocess(row['electroactive_polymers']), axis=1).tolist()

pe_lines = df.apply(lambda row: text_field.preprocess(row['pedot_electrodes']), axis=1).tolist()

np_lines = df.apply(lambda row: text_field.preprocess(row['neuroprosthetics']), axis=1).tolist()

dl_lengths = [len(line) for line in dl_lines]

c_lengths = [len(line) for line in c_lines]

hc_lengths = [len(line) for line in hc_lines]

vr_lengths = [len(line) for line in vr_lines]

bmi_lengths = [len(line) for line in bmi_lines]

ep_lengths = [len(line) for line in ep_lines]

pe_lengths = [len(line) for line in pe_lines]

np_lengths = [len(line) for line in np_lines]





# plot

fig, ax = plt.subplots(nrows=4,  ncols=2, figsize=(20, 20))



ax[0, 0].hist(dl_lengths, bins=50, color='y')[-1]

ax[0, 0].set_title("deep learning")

ax[0, 1].hist(c_lengths, bins=50, color='g')[-1]

ax[0, 1].set_title("covid 19")

ax[1, 0].hist(hc_lengths, bins=50, color='r')[-1]

ax[1, 0].set_title("human connectome")

ax[1, 1].hist(vr_lengths, bins=50, color='purple')[-1]

ax[1, 1].set_title("virtual reality")

ax[2, 0].hist(bmi_lengths, bins=50)[-1]

ax[2, 0].set_title("brain machine interfaces")

ax[2, 1].hist(ep_lengths, bins=50, color='pink')[-1]

ax[2, 1].set_title("electroactive polymers")

ax[3, 0].hist(pe_lengths, bins=50, color='b')[-1]

ax[3, 0].set_title("pedot electrodes")

ax[3, 1].hist(np_lengths, bins=50)[-1]

ax[3, 1].set_title("neuroprosthetics")

plt.suptitle("Text length distribution", y=.95, fontsize=20)

plt.show()
from torchtext.data import Example

from torchtext.data import Dataset



DEVICE = "cpu"#"cuda:0"



# lines longer than 5k characters and less than 50 are eliminated.

lines = new_df.apply(lambda row: text_field.preprocess(row['data']), axis=1).tolist()

lines = [line for line in lines if 4000 > len(line) >= 50]



fields = [('text', text_field)]

examples = [Example.fromlist([line], fields) for line in lines]

dataset = Dataset(examples, fields)
# split data into train and validation 

train_dataset, test_dataset = dataset.split(split_ratio=0.75)



text_field.build_vocab(train_dataset, min_freq=30)



print('Vocab size =', len(text_field.vocab))

print(text_field.vocab.itos)
# Iterative object



from torchtext.data import BucketIterator



train_iter, test_iter = BucketIterator.splits(datasets=(train_dataset, test_dataset), batch_sizes=(32, 192), 

                                              shuffle=True, device=DEVICE, sort=False)



batch = next(iter(train_iter))

batch
import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



DEVICE = 'cuda:0'
class RnnLM(nn.Module):

    def __init__(self, vocab_size, emb_dim=32, lstm_hidden_dim=192, num_layers=256):  

        super().__init__()



        self._emb = nn.Embedding(vocab_size, emb_dim)

        self._rnn = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden_dim)

        self._out_layer = nn.Linear(lstm_hidden_dim, vocab_size)



    def forward(self, inputs, hidden=None):

        outputs = self._emb(inputs)

        outputs, hidden = self._rnn(outputs, hidden)

        outputs = self._out_layer(outputs)

        

        return outputs, hidden



model = RnnLM(vocab_size=len(train_iter.dataset.fields['text'].vocab)).to(DEVICE)


def sample(probs, temp):

    probs = F.log_softmax(probs.squeeze(), dim=0)

    probs = (probs / temp).exp()

    probs /= probs.sum()

    probs = probs.cpu().numpy()



    return np.random.choice(np.arange(len(probs)), p=probs)





def generate(model, temp=0.8):

    model.eval()

    with torch.no_grad():

        prev_token = train_iter.dataset.fields['text'].vocab.stoi['<s>']

        end_token = train_iter.dataset.fields['text'].vocab.stoi['</s>']

        

        hidden = None

        for _ in range(1000):# After preprocessing text contains a length of at least 50 to 5000 characters.

            probs, hidden = model(torch.LongTensor([[prev_token]]).to(DEVICE), 

                                  hidden)

            prev_token = sample(probs[-1], temp)

        

            print(train_dataset.fields['text'].vocab.itos[prev_token], end='')

            if prev_token == end_token:

                return

            



generate(model)
import math

from tqdm import tqdm





def do_epoch(model, criterion, data_iter, unk_idx, pad_idx, optimizer=None, name=None):

    epoch_loss = 0

    

    is_train = not optimizer is None

    name = name or ''

    model.train(is_train)

    

    batches_count = len(data_iter)

    

    with torch.autograd.set_grad_enabled(is_train):

        with tqdm(total=batches_count) as progress_bar:

            for i, batch in enumerate(data_iter):

    

                labels = batch.text[1:, :]

                labels = labels.view(-1).to(DEVICE)

            

                logits, _ = model(batch.text.to(DEVICE))

                logits = logits[:-1, :, :]

                logits = logits.view(-1, logits.shape[-1])

                

                target = ((labels != pad_idx) * (labels != unk_idx)).float()

                loss = torch.sum(criterion(logits, labels.view(-1)) * target) / torch.sum(target)

                

                epoch_loss += loss.item()



                if optimizer:

                    optimizer.zero_grad()

                    loss.backward()

                    nn.utils.clip_grad_norm_(model.parameters(), 1.)

                    optimizer.step()



                progress_bar.update()

                progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(name, loss.item(), 

                                                                                         math.exp(loss.item())))

                

            progress_bar.set_description('{:>5s} Loss = {:.5f}, PPX = {:.2f}'.format(

                name, epoch_loss / batches_count, math.exp(epoch_loss / batches_count))

            )



    return epoch_loss / batches_count





def fit(model, criterion, optimizer, train_iter, epochs_count=1, unk_idx=0, pad_idx=1, val_iter=None):

    for epoch in range(epochs_count):

        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)

        train_loss = do_epoch(model, criterion, train_iter, unk_idx, pad_idx, optimizer, name_prefix + 'Train:')

        

        if not val_iter is None:

            val_loss = do_epoch(model, criterion, val_iter, unk_idx, pad_idx, None, name_prefix + '  Val:')



        generate(model)

        print()
model = RnnLM(vocab_size=len(train_iter.dataset.fields['text'].vocab)).to(DEVICE)



pad_idx = train_iter.dataset.fields['text'].vocab.stoi['<pad>']

unk_idx = train_iter.dataset.fields['text'].vocab.stoi['<unk>']

criterion = nn.CrossEntropyLoss().to(DEVICE)



optimizer = optim.Adam(model.parameters())



fit(model, criterion, optimizer, train_iter, epochs_count=30, unk_idx=unk_idx, pad_idx=pad_idx, val_iter=test_iter)