import spacy

from gensim.parsing.preprocessing import remove_stopwords

from nltk.corpus  import stopwords

import re

from gensim.utils import lemmatize

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

from tensorflow import keras

import tensorflow as tf

import datetime

import numpy as np

import pandas as pd

from transformers import *

from keras.utils.np_utils import to_categorical
#nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser",'ner'])

df_train=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_train.csv',encoding='utf-8-sig')

df_val=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_val.csv',encoding='utf-8-sig')

df_test=pd.read_csv('/kaggle/input/dbpedia-classes/DBPEDIA_test.csv',encoding='utf-8-sig')
df_train=df_train.dropna(axis=0)

df_val=df_val.dropna(axis=0)

df_test=df_test.dropna(axis=0)
def cleaning(df):

    #df.loc[:,'SentimentText']=pd.DataFrame(df.loc[:,'SentimentText'].str.lower())

    df.loc[:,'text'] = [re.sub(r'\d+','', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(r'[^a-zA-Z]',' ', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(r"\b[a-zA-Z]\b", ' ', i) for i in df.loc[:,'text']]

    

    #df.loc[:,'text'] = [re.sub(r"[#|\.|_|\^|\$|\&|=|;|,|‐|-|–|(|)|//|\\+|\|*|\']+",'', i) for i in df.loc[:,'text']]

    df.loc[:,'text'] = [re.sub(' +',' ', i) for i in df.loc[:,'text']]

    return(df)
df_train=cleaning(df_train)

df_val=cleaning(df_val)

df_test=cleaning(df_test)


from sklearn.preprocessing import LabelEncoder

# encode class values as integers

encoder = LabelEncoder()

encoder.fit(df_train['l3'])

encoded_Y = encoder.transform(df_train['l3'])
import pandas as pd

import torch

import transformers

from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertModel, DistilBertTokenizer,DistilBertForMaskedLM
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
# Defining some key variables that will be used later on in the training

MAX_LEN = 512

TRAIN_BATCH_SIZE = 4

VALID_BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 1e-05

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
df_train['encoded_l3']=encoded_Y

df_val['encoded_l3']=encoder.transform(df_val['l3'])

df_test['encoded_l3']=encoder.transform(df_test['l3'])
df_data=df_train.iloc[:,[0,4]]
class Triage(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):

        self.len = len(dataframe)

        self.data = dataframe

        self.tokenizer = tokenizer

        self.max_len = max_len

        

    def __getitem__(self, index):

        text = str(self.data.text[index])

        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(

            text,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True,

            return_token_type_ids=True

        )

        ids = inputs['input_ids']

        mask = inputs['attention_mask']



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'targets': torch.tensor(self.data.encoded_l3[index], dtype=torch.long)

        } 

    

    def __len__(self):

        return self.len
'''

train_size = 0.8

train_dataset=df_data.sample(frac=train_size,random_state=200).reset_index(drop=True)

test_dataset=df_data.drop(train_dataset.index).reset_index(drop=True)





print("FULL Dataset: {}".format(df_data.shape))

print("TRAIN Dataset: {}".format(train_dataset.shape))

print("TEST Dataset: {}".format(test_dataset.shape)) 

'''

training_set = Triage(df_train, tokenizer, MAX_LEN)

testing_set = Triage(df_test, tokenizer, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



test_params = {'batch_size': VALID_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



training_loader = DataLoader(training_set, **train_params)

testing_loader = DataLoader(testing_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 



class DistillBERTClass(torch.nn.Module):

    def __init__(self):

        super(DistillBERTClass, self).__init__()

        self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.l2 = torch.nn.Dropout(0.2)

        self.l3 = torch.nn.Linear(768, 1)

    

    def forward(self, ids, mask):

        output_1= self.l1(ids, mask)

        output_2 = self.l2(output_1[0])

        output = self.l3(output_2)

        return output


model = DistillBERTClass()

model.to(device)
# Creating the loss function and optimizer

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
def train(epoch):

    model.train()

    for _,data in enumerate(training_loader, 0):

        ids = data['ids'].to(device, dtype = torch.long)

        mask = data['mask'].to(device, dtype = torch.long)

        targets = data['targets'].to(device, dtype = torch.long)



        outputs = model(ids, mask).squeeze()



        optimizer.zero_grad()

        loss = loss_function(outputs, targets)

        if _%10000==0:

            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
for epoch in range(EPOCHS):

    train(epoch)
def valid(model, testing_loader):

    model.eval()

    n_correct = 0; n_wrong = 0; total = 0

    with torch.no_grad():

        for _, data in enumerate(testing_loader, 0):

            ids = data['ids'].to(device, dtype = torch.long)

            mask = data['mask'].to(device, dtype = torch.long)

            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask).squeeze()

            big_val, big_idx = torch.max(outputs.data, dim=1)

            total+=targets.size(0)

            n_correct+=(big_idx==targets).sum().item()

    return (n_correct*100.0)/total
print('This is the validation section to print the accuracy and see how it performs')

print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')



acc = valid(model, testing_loader)

print("Accuracy on test data = %0.2f%%" % acc)
torch.save(model,'model_dbpedia.pt')