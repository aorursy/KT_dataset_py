import numpy as np

import pandas as pd

import random

import torch



import os
from fastai.tabular import *
# set random seeds for reproducibility

seed = 23



# python RNG

random.seed(seed)



# pytorch RNGs



torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



np.random.seed(seed)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# rudimentary lookup code for replacing Age NA values with closer approximations based on distribution of other variables in data

def lookup(x,lkupdf):

    conditions_series = np.logical_and(np.logical_and(lkupdf['Parch'] == x.Parch, lkupdf['Sex'] == x.Sex), 

                                        lkupdf['Pclass'] == x.Pclass)

    try:

        condition_iloc = conditions_series[conditions_series==True].index[0]

        return lkupdf.iloc[condition_iloc]['Age']

    except IndexError:

        fewerconditions_series = np.logical_and(lkupdf['Pclass'] == x.Pclass,lkupdf['Sex'] == x.Sex)

        fewerconditions_iloc = fewerconditions_series[fewerconditions_series==True].index[0]

        return lkupdf.iloc[fewerconditions_iloc]['Age']

    except:

      return "Something else went wrong"
train.head()
import re

def ticketTransforms(df):

    df['TicketPartA'] = df.Ticket.apply(lambda x: x.rstrip('0123456789').split('/')[0].strip(',. '))

    df['TicketPartB'] = df.Ticket.apply(lambda x: x.rstrip('0123456789').split('/')[1].strip(',. ') if not IndexError else -1)

    df['TicketNum'] = df.Ticket.apply(lambda x: x.split(" ")[-1].strip()) # last part of ticket is the ticket num

    df.drop('Ticket',axis=1,inplace=True) # get rid of variable now that we've extracted relevant features



def nameTransforms(df):

    df['Title'] = df.Name.apply(lambda x: x.split(", ")[1].split('.')[0])

    df['LastName'] = df.Name.apply(lambda x: x.split(", ")[0])

    df['MarriedWoman'] = df.Name.apply(lambda x: (x.find('('))!= -1 & x.find('Mrs.')!=-1)

    df['NameParenthesis'] = df.Name.apply(lambda x: (x.find('(')!= -1))

    df.drop('Name',inplace=True,axis=1)

    

def cabinTransforms(df):

    from string import ascii_letters

    df['CabinNumber'] = df.Cabin.apply(lambda x: -1 if pd.isnull(x) else str(x).lstrip(ascii_letters))

    df['CabinLetter'] = df.Cabin.apply(lambda x: '-1' if pd.isnull(x) else str(x)[0])

    df['CabinCount'] = df.Cabin.apply(lambda x: len(str(x).split(' ')))

    df['CabinDigitLength'] = df.Cabin.apply(lambda x: '-1' if pd.isnull(x) 

                  else len(str(x).lstrip(ascii_letters).split(" ")[0]))

    #df.drop('Cabin',inplace=True,axis=1)

    

def fixNulls(df):

    mean = df.Fare.mean()

    df['FareNull'] = df.Fare.isnull()

    df.loc[df.Fare.isnull(),'Fare'] = mean

    

def ageTransforms(df):

    agelkupraw = pd.DataFrame(df.groupby(['Pclass','Parch','Sex'])['Age'].mean()).reset_index()

    agelookuptable = agelkupraw[np.isnan(agelkupraw.Age)==False]    

    df['Age'] = df.apply(lambda row: row.Age if not np.isnan(row.Age) else lookup(row,agelookuptable), axis=1)

    

def applyTransforms(df):

    ticketTransforms(df)

    cabinTransforms(df)

    nameTransforms(df)

    fixNulls(df)

    ageTransforms(df)
applyTransforms(train)

applyTransforms(test)
train_samples = train.sample(frac=0.8, random_state=seed)

valid_samples = train.drop(train_samples.index)

train_samples.loc[:,'isValid'] = 0

valid_samples.loc[:,'isValid'] = 1
valid_idx = valid_samples.index
undersampled_class_train = train_samples.loc[train_samples.Survived == 1,:]

dominant_class_train = train_samples.loc[train_samples.Survived == 0,:]
upsample_scalar = 3

upsample = pd.concat([undersampled_class_train for i in range(0,upsample_scalar)]).reset_index(drop=True).sample(n=448)
df = pd.concat([upsample,dominant_class_train,valid_samples]).reset_index(drop=True)

df.loc[:,'randomnoise'] = [random.random() for i in range(0,df.copy().shape[0])]
test.loc[:,'randomnoise'] = [random.random() for i in range(0,test.copy().shape[0])]
# need for fastai model dev

#valid_idx = np.random.choice(train.shape[0],int(train.shape[0]*.2)) # 20% for validation

procs = [FillMissing,Categorify,Normalize]

cont_names = ['Age','Fare']

cat_names = np.setdiff1d(train.columns.values,['Survived']+cont_names)
# def train_and_eval_tabular_learner(df,

#                                    valid_idx,`

#                                    lr=slice(3e-3), epochs=1, layers=[200, 100], ps=[0.5, 0.2], name='learner'):

#     data = (TabularList.from_df(df, path='', cat_names=cat_names, 

#                             cont_names=cont_names,

#                             procs=procs)

#                             .split_by_idx(valid_idx=valid_idx)

#                            .label_from_df(cols='Survived')

#                            .add_test(

#                                TabularList.from_df(test, path='',

#                                                    cat_names=cat_names, 

#                                                    cont_names=cont_names,

#                                                    procs=procs)

#                            )

#                             .databunch())



#     learner = tabular_learner(data, layers=layers, ps=ps)

#     learner.fit_one_cycle(epochs, lr)



#     learner.save(name,with_opt=False) # what does with_opt mean here?

        

#     # run prediction on validation set

#     valid_predicts, _ = learner.get_preds(ds_type=DatasetType.Valid)

#     valid_probs = np.array(valid_predicts[:,1])

#     valid_targets = df.loc[valid_idx,:].Survived.values # why write this way instead of reference target column in loc indexer?

#     #valid_score = accuracy(torch.max(learn.get_preds(ds_type=DatasetType.Test)[0],1)[1],tensor(valid_targets))

    

#     # run prediction on test    

#     test_predicts, _ = learner.get_preds(ds_type=DatasetType.Test)

#     test_probs = to_np(test_predicts[:, 1])



#     return valid_probs, test_probs #, valid_score, 
# %%time

# sub_features = []

# valid_scores = []

# valid_predictions = []

# predictions = []

# num_epochs = 30 # how many models do we want to ensemble

# cv_counts = len(df)//num_epochs # what does this do?

# saved_model_prefix = 'learner'



# for i in range(num_epochs):

#     print('training model {:}'.format(i))

#     # each model is trained with a different subsample of the features

#     # when we make predictions at the end, this isn't a problem because we simply average the different models' scores (or apply

#     # another voting scheme)

#     name = f'{saved_model_prefix}_{i}'

#     # TO DO: What do the commented out lines below mean?

# #     this_train_idx = list(valid_idx.values) + list(train_samples.index.values[:cv_counts * i]) + list(train_samples.index.values[cv_counts*(i+1):])

# #     this_train_df = df.loc[this_train_idx].reset_index()

#     valid_probs, test_probs = train_and_eval_tabular_learner(df, 

#                                                                     valid_idx=valid_idx,

#                                                                     epochs=2, # 8 works well, not overfitting too much, could up to 10?

#                                                                     lr=slice(3e-3), # fastai default works well here

#                                                                     name=name)

    

#     #valid_scores.append(score)

#     valid_predictions.append(valid_probs)

#     predictions.append(test_probs)
data = (TabularList.from_df(df, path='', cat_names=cat_names, 

                        cont_names=cont_names,

                        procs=procs)

                        .split_by_idx(valid_idx=valid_idx)

                       .label_from_df(cols='Survived')

                       .add_test(

                           TabularList.from_df(test, path='',

                                               cat_names=cat_names, 

                                               cont_names=cont_names,

                                               procs=procs)

                       )

                        .databunch())
learn = tabular_learner(data,layers=[400,100])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,3e-2,wd=.01)
learn.save("round1")
learn.fit_one_cycle(1,1e-4,wd=.01) # if valid loss goes up, we'll use round1 model for predictions
learn.save("round2")
learn.fit_one_cycle(1,1e-4,wd=.01) # if valid loss goes up, we'll use round2 model for predictions
learn.save('round3')
learn.fit_one_cycle(12,1e-4,wd=.01) # if valid loss goes up, we'll use round2 model for predictions
_, indices = torch.max(learn.get_preds(ds_type=DatasetType.Test)[0],1)
submission = pd.read_csv('../input/gender_submission.csv')
submission.Survived = indices
submission.to_csv('preds.csv',index=False)
from IPython.display import FileLink

FileLink(f'preds.csv')