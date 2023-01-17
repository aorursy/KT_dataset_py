!pip install nlp

import pandas as pd
from nlp import load_dataset

dataset_valid = load_dataset(

   'race',split='validation[:]')
dataset_train =  load_dataset(

   'race',split='train[:]')
dataset_train
dataset_valid
from pprint import pprint



print(f"👉Dataset len(dataset): {len(dataset_valid)}")

train = pd.DataFrame()

train['article'] = dataset_train['article']

train['answer'] = dataset_train['answer']

train['question'] = dataset_train['question']

train['options'] =dataset_train['options']
valid = pd.DataFrame()

valid['article'] = dataset_valid['article']

valid['answer'] = dataset_valid['answer']

valid['question'] = dataset_valid['question']

valid['options'] = dataset_valid['options']

train.head()
from sklearn import preprocessing

import pandas as pd

le = preprocessing.LabelEncoder()

train["answer"]=le.fit_transform(train['answer'])

valid["answer"]=le.fit_transform(valid['answer'])
def label(df):

    for i in range(len(df)):

        k = int(df['answer'][i])

        df["label"][i] = df["options"][i][k]

        

        
train["label"] = ""

valid["label"] = ""
label(train)
label(valid)
train.head()
valid.head()
train.head()
train = train[~train.question.str.contains("_")]
valid = valid[~valid.question.str.contains("_")]
train.shape
valid.shape
train.head()
train.drop(["answer","options"],inplace = True,axis = 1)
valid.drop(["answer","options"],inplace = True,axis = 1)
train.columns
train.head()
from nlp import load_dataset

trainbool = load_dataset(

   'boolq',split = 'train[:]')

validbool = load_dataset(

   'boolq',split = 'validation[:]')
trainbool
booltrain = pd.DataFrame(columns = ['article', 'question', 'label'])

boolvalid = pd.DataFrame(columns = ['article', 'question', 'label'])
booltrain["article"] = trainbool["passage"]

booltrain["question"] = trainbool["question"]

booltrain["label"] = trainbool["answer"]
boolvalid["article"] = validbool["passage"]

boolvalid["question"] = validbool["question"]

boolvalid["label"] = validbool["answer"]
booltrain.shape
booltrain.head()
train['article'][4]
train_final = pd.concat([train,booltrain])
valid_final = pd.concat([valid,boolvalid])
train_final.shape
valid_final.shape
# train_final.to_csv("train.csv",index = False)

# valid_final.to_csv("valid.csv",index = False)
booltrain["label"] = booltrain["label"].replace({True: 'Yes', False: 'No'})

boolvalid["label"] = boolvalid["label"].replace({True: 'Yes', False: 'No'})

booltrain.to_csv("train.csv",index = False)

boolvalid.to_csv("valid.csv",index = False)