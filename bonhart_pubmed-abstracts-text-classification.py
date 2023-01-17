import pandas as pd



df = pd.read_csv("/kaggle/input/pubmed-abstracts/pubmed_abstracts.csv", index_col=0)

df.head()
import re



def preprocess_tex(text):

    text = text.str.lower() # lowercase

    text = text.str.replace(r"[^a-zA-Zа-яА-Я1-9]+", ' ') # remove () & []

    text = text.str.replace(r"\#","") # replaces hashtags

    text = text.str.replace(r"http\S+","URL")  # remove URL addresses

    text = text.str.replace(r"@","")

    text = text.str.replace("\s{2,}", " ")

    return text





cols = ['deep_learning', 'covid_19', 

        'human_connectome','virtual_reality', 

        'brain_machine_interfaces', 'electroactive_polymers', 

        'pedot_electrodes', 'neuroprosthetics']



df = df[cols].apply(lambda x: preprocess_tex(x))

df.head()
# Data

label_names = []



dl = df.deep_learning.dropna()

label_names += len(dl) * ["deep_learning"]

covid_19 = df.covid_19.dropna()

label_names += len(covid_19) * ["covid_19"]

human_connectome = df.human_connectome.dropna()

label_names += len(human_connectome) * ["human_connectome"]

virtual_reality = df.virtual_reality.dropna()

label_names += len(virtual_reality) * ["virtual_reality"]

brain_machine_interfaces = df.brain_machine_interfaces.dropna()

label_names += len(brain_machine_interfaces) * ["brain_machine_interfaces"]

electroactive_polymers = df.electroactive_polymers.dropna()

label_names += len(electroactive_polymers) * ["electroactive_polymers"]

pedot_electrodes = df.pedot_electrodes.dropna()

label_names += len(pedot_electrodes) * ["pedot_electrodes"]

neuroprosthetics = df.neuroprosthetics.dropna()

label_names += len(neuroprosthetics) * ["neuroprosthetics"]





text = list(pd.concat([dl, covid_19, 

                        human_connectome, virtual_reality, 

                        brain_machine_interfaces, electroactive_polymers, 

                        pedot_electrodes, neuroprosthetics]))#.unique())



df = pd.DataFrame({"text" : text,

                  "label_name" : label_names})





# create a new digital column matching labels

labels = {"deep_learning": 0 , "covid_19": 1, 

          "human_connectome": 2, "virtual_reality": 3, 

          "brain_machine_interfaces": 4, "electroactive_polymers": 5,

          "pedot_electrodes": 6, "neuroprosthetics": 7}



df['label'] = df['label_name'].apply(lambda x: labels[x])



# removing duplicates

df = df.drop_duplicates('text').reset_index(drop=True)

df
# Plot distribution

import matplotlib.pyplot as plt

plt.style.use("dark_background")



df['label_name'].value_counts().plot(kind='bar',

                                     title="Distribution of data grouped by classes", 

                                     figsize=(10, 6));

import torch

import torch.nn as nn



from torchtext import data
TEXT = data.Field(tokenize = 'spacy', include_lengths = True)

LABEL = data.LabelField(dtype = torch.float)
from sklearn.model_selection import train_test_split

# split data into train and validation 

train_df, valid_df = train_test_split(df[['text', 'label']])

train_df, valid_df = train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
class DataFrameDataset(data.Dataset):



    def __init__(self, df, fields, is_test=False, **kwargs):

        examples = []

        for i, row in df.iterrows():

            label = row.label if not is_test else None

            text = row.text

            examples.append(data.Example.fromlist([text, label], fields))



        super().__init__(examples, fields, **kwargs)



    @staticmethod

    def sort_key(ex):

        return len(ex.text)



    @classmethod

    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):

        train_data, val_data, test_data = (None, None, None)

        data_field = fields



        if train_df is not None:

            train_data = cls(train_df.copy(), data_field, **kwargs)

        if val_df is not None:

            val_data = cls(val_df.copy(), data_field, **kwargs)

        if test_df is not None:

            test_data = cls(test_df.copy(), data_field, True, **kwargs)



        return tuple(d for d in (train_data, val_data, test_data) if d is not None)
fields = [('text',TEXT), ('label',LABEL)]



train_ds, val_ds = DataFrameDataset.splits(fields, train_df=train_df, val_df=valid_df)
# Lets look at a random example

print(vars(train_ds[15]))



# Check the type 

print(type(train_ds[15]))
MAX_VOCAB_SIZE = 25000



TEXT.build_vocab(train_ds, 

                 max_size = MAX_VOCAB_SIZE, 

                 vectors = 'glove.6B.200d',

                 unk_init = torch.Tensor.zero_)
LABEL.build_vocab(train_ds)
BATCH_SIZE = 128



device = 'cuda:0'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')



train_iterator, valid_iterator = data.BucketIterator.splits(

    (train_ds, val_ds), 

    batch_size = BATCH_SIZE,

    sort_within_batch = True,

    device = device)
# Hyperparameters

num_epochs = 20

learning_rate = 0.001



INPUT_DIM = len(TEXT.vocab)

EMBEDDING_DIM = 200

HIDDEN_DIM = 256

OUTPUT_DIM = 8

N_LAYERS = 2

BIDIRECTIONAL = True

DROPOUT = 0.2

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token] # padding
class LSTM_net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 

                 bidirectional, dropout, pad_idx):

        super().__init__()

        

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        

        self.rnn = nn.LSTM(embedding_dim, 

                           hidden_dim, 

                           num_layers=n_layers, 

                           bidirectional=bidirectional, 

                           dropout=dropout)

        

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        

        self.fc2 = nn.Linear(hidden_dim, 8)

        

        self.dropout = nn.Dropout(dropout)

        

    def forward(self, text, text_lengths):

        

        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        output = self.fc1(hidden)

        output = self.dropout(self.fc2(output))

            

        return output
#creating instance of our LSTM_net class

model = LSTM_net(INPUT_DIM, 

            EMBEDDING_DIM, 

            HIDDEN_DIM, 

            OUTPUT_DIM, 

            N_LAYERS, 

            BIDIRECTIONAL, 

            DROPOUT, 

            PAD_IDX)
pretrained_embeddings = TEXT.vocab.vectors



print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)
#  to initiaise padded to zeros

model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



print(model.embedding.weight.data)
model = model.cuda()





# Loss and optimizer

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
import numpy as np



def accuracy(preds, y):

    """

    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8

    """



    indicies = torch.argmax(preds, 1)



    correct = (indicies == y).float()

    acc = correct.sum() / len(correct)

    return acc
# training function 

def train(model, iterator):

    

    epoch_loss = 0

    epoch_acc = 0

    

    model.train()

    

    for batch in iterator:

        text, text_lengths = batch.text

        

        optimizer.zero_grad()

        predictions = model(text.cuda(), text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label.long().cuda())

        acc = accuracy(predictions, batch.label)



        loss.backward()

        optimizer.step()

        

        epoch_loss += loss.item()

        epoch_acc += acc.item()

        



    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator):

    

    epoch_acc = 0

    model.eval()

    

    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.text

            predictions = model(text.cuda(), text_lengths).squeeze(1)

            acc = accuracy(predictions.detach().cpu(), batch.label.cpu())

            

            epoch_acc += acc.item()

        

    return epoch_acc / len(iterator)
import time



t = time.time()

loss=[]

acc=[]

val_acc=[]



for epoch in range(num_epochs):

    

    train_loss, train_acc = train(model, train_iterator)

    valid_acc = evaluate(model, valid_iterator)

    

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')

    print(f'\t Val. Acc: {valid_acc*100:.2f}%')

    

    loss.append(train_loss)

    acc.append(train_acc)

    val_acc.append(valid_acc)

    

print(f'time:{time.time()-t:.3f}')
plt.xlabel("runs")

plt.ylabel("normalised measure of loss/accuracy")

x_len=list(range(len(acc)))

plt.axis([0, max(x_len), 0, 1])

plt.title('result of LSTM')

loss=np.asarray(loss)/max(loss)

plt.plot(x_len, loss, 'r.',label="loss")

plt.plot(x_len, acc, 'b.', label="accuracy")

plt.plot(x_len, val_acc, 'g.', label="val_accuracy")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.2)

plt.show()