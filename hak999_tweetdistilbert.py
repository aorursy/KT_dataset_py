# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def prepare_data(df):

    df.keyword.fillna("unknown", inplace = True)

    df.location.fillna("unknown", inplace = True)

    for index in df.index:

        df.loc[index, 'keyword'] = ' keyword: ' + df.loc[index, 'keyword']

        df.loc[index, 'location'] = ' location: ' + df.loc[index, 'location']

    

    df['clean_tweet'] =  df.text + df.location + df.keyword

    
df = pd.read_csv(os.path.join('/kaggle/input/nlp-getting-started', 'train.csv'))

df = df.sample(frac = 1)

prepare_data(df)

train_df, val_df = np.split(df, [int(.9*len(df))])




training_args = TrainingArguments(

    output_dir= '/kaggle/output',          # output directory

    num_train_epochs=3,              # total number of training epochs

    per_device_train_batch_size=16,  # batch size per device during training

    per_device_eval_batch_size=64,   # batch size for evaluation

    warmup_steps=500,                # number of warmup steps for learning rate scheduler

    weight_decay=0.01,               # strength of weight decay

    logging_dir='/kaggle/output',            # directory for storing logs

    logging_steps=10,

)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_df.clean_tweet.to_list(), truncation=True, padding=True)

val_encodings = tokenizer(val_df.clean_tweet.to_list(), truncation=True, padding=True)

import torch



class TweetDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):

        self.encodings = encodings

        self.labels = labels



    def __getitem__(self, idx):

        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        item['labels'] = torch.tensor(self.labels[idx])

        return item



    def __len__(self):

        return len(self.labels)
train_ds = TweetDataset(train_encodings, train_df.target.to_list())

val_ds = TweetDataset(val_encodings, val_df.target.to_list())
trainer = Trainer(

    model=model,                         # the instantiated 🤗 Transformers model to be trained

    args=training_args,                  # training arguments, defined above

    train_dataset=train_ds,              # training dataset

    eval_dataset=val_ds,                 # evaluation dataset

)

trainer.train()
trainer.evaluate()
from transformers import pipeline

pl = pipeline('sentiment-analysis',  model=model, tokenizer=tokenizer, device=0)
print(pl('it is a sunny day'))

print(pl('o my god the airplane crashed!'))
test_df = pd.read_csv(os.path.join('/kaggle/input/nlp-getting-started', 'test.csv'))

prepare_data(test_df)

predictions = [ pl(test_df.loc[x, 'clean_tweet']) for x in test_df.index ]

preds = [ 0 if x[0]['label'] == 'LABEL_0' else 1  for x in predictions ]
for i in range(10):

    print(str(preds[i]) + '  ' + test_df.loc[i, 'clean_tweet'])
output = pd.DataFrame({'id': test_df.id, 'target': preds})



output.to_csv('hak_submission.csv', index=False)

print("Your submission was successfully saved!")
