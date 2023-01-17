import numpy as np

import pandas as pd

import seaborn as sns

import torch



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))      
wiki_movie_pure_data = pd.read_csv('/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')
wiki_movie_pure_data.head(5)
sns.distplot(wiki_movie_pure_data['Release Year'], color="red")
sns.countplot(y="Origin/Ethnicity", data=wiki_movie_pure_data, color="pink", order = wiki_movie_pure_data["Origin/Ethnicity"].value_counts().index)
df_data = wiki_movie_pure_data[['Plot', 'Genre']]
df_data = df_data[df_data['Genre'] != 'unknown']
print('Count of rows in the dataframe: ', len(df_data))
df_data.head(5)
import string

translator = str.maketrans('','',string.punctuation)

df_data['Plot'] = df_data.apply(lambda row : row['Plot'].translate(translator), axis = 1) 
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder() 

  

df_data['Genre']= label_encoder.fit_transform(df_data['Genre']) 

  

df_data['Genre'].unique() 
df_data.head(5)
print('Mean plot length: ', df_data['Plot'].apply(lambda x: len(x.split())).mean())
df_train, df_validation, df_test = np.split(df_data.sample(frac=1), [int(.7*len(df_data)), int(.8*len(df_data))])



df_train.to_csv('data_train.csv', index=False)

df_validation.to_csv('data_validation.csv', index=False)

df_test.to_csv('data_test.csv', index=False)
from torchtext.data import Field

tokenize = lambda x: x.split()

SENTENCE_LEN = 400



TEXT = Field(sequential=True, tokenize=tokenize, lower=True, init_token='<START>', eos_token='<END>', fix_length=SENTENCE_LEN)

LABEL = Field(sequential=False, use_vocab=False)
from torchtext.data import TabularDataset



wiki_movie_datafields = [("Plot", TEXT), ("Genre", LABEL)]

train_td, vad_td = TabularDataset.splits(

               path="/kaggle/working",

               train='data_train.csv', validation="data_validation.csv",

               format='csv',

               skip_header=True,

               fields=wiki_movie_datafields)



test_datafields = [("Plot", TEXT), ("Genre", None)]

test_td = TabularDataset(

           path="/kaggle/working/data_test.csv",

           format='csv',

           skip_header=True,

           fields=test_datafields)
TEXT.build_vocab(train_td)
from torchtext.data import Iterator, BucketIterator



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



BATH_SIZE = 64



train_iter, val_iter = BucketIterator.splits(

 (train_td, vad_td),

 batch_sizes=(BATH_SIZE, BATH_SIZE),

 device=-1, # if you want to use the GPU, specify the GPU number here

 sort_key=lambda x: len(x.Plot),

 sort_within_batch=False,

 repeat=False 

)



test_iter = Iterator(test_td, batch_size=BATH_SIZE, device=-1, sort=False, sort_within_batch=False, repeat=False)
class BatchWrapper:

      def __init__(self, dl, x_var, y_vars):

            self.dl, self.x_var, self.y_vars = dl, x_var, y_vars 



      def __iter__(self):

            for batch in self.dl:

                  x = getattr(batch, self.x_var)

                  if self.y_vars is not None:

                      y = getattr(batch, self.y_vars)

                      yield (x, y)

                  else: 

                      yield (x, -1)



      def __len__(self):

            return len(self.dl)



train_dl = BatchWrapper(train_iter, "Plot", "Genre")

valid_dl = BatchWrapper(val_iter, "Plot", "Genre")

test_dl = BatchWrapper(test_iter, "Plot", None)
import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable



class SimpleLSTMBaseline(nn.Module):

    def __init__(self, hidden_dim, emb_dim=300, num_linear=1):

        super().__init__() # don't forget to call this!

        self.embedding = nn.Embedding(len(TEXT.vocab), emb_dim)

        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)

        self.linear_layers = []

        for _ in range(num_linear - 1):

            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))

            self.linear_layers = nn.ModuleList(self.linear_layers)

        self.predictor = nn.Linear(hidden_dim, 1)



    def forward(self, seq):

        hdn, _ = self.encoder(self.embedding(seq))

        feature = hdn[-1, :, :]

        for layer in self.linear_layers:

          feature = layer(feature)

        preds = self.predictor(feature)

        return preds



em_sz = 100

nh = 500

nl = 10

model = SimpleLSTMBaseline(nh, emb_dim=em_sz, num_linear=nl)
import tqdm

import torch



opt = optim.Adam(model.parameters(), lr=1e-3)

loss_func = nn.MSELoss()



epochs = 2



for epoch in range(1, epochs + 1):

    running_loss = 0.0

    running_corrects = 0

    model.train()

    for x, y in tqdm.tqdm(train_dl, position=0, leave=True):

        opt.zero_grad()



        preds = model(x)

        loss = loss_func(y, preds.squeeze())

        loss.backward()

        opt.step()



        running_loss += loss.data * x.size(0)



    epoch_loss = running_loss / len(train_td)



    val_loss = 0.0

    model.eval()

    for x, y in tqdm.tqdm(valid_dl, position=0, leave=True):

        preds = model(x)

        loss = loss_func(y, preds)

        val_loss += loss.data * x.size(0)



    val_loss /= len(vad_td)

    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))