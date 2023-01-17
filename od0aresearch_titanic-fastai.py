# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import random

from fastai.tabular import * 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the provided data into pandas dataframes

df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test = df_test.fillna(0)
# This is my feature engineering, it is in a function to make it easier to apply to both the validation and training sets

def prepdata(df):

    # Created new columns from the existing ones. 

    df[['surname', 'salutation+firstname']] = df['Name'].str.split(',', n=1, expand = True)

    df[['salutation', 'firstname']] = df['salutation+firstname'].str.split('.', n=1, expand = True)

    df[['ticket_firsthalf', 'ticket_secondhalf']] = df['Ticket'].str.split(' ', n=1, expand = True)

    #data["Team"]= data["Team"].str.split("t", n = 1, expand = True) 

    df.drop(['PassengerId', 'Name', 'salutation+firstname', 'Ticket', 'firstname'], axis=1, inplace=True)

    

    df['Deck'] = ''

    for index, row in df.iterrows():

        try:

            if df.at[index, 'Cabin'] != 'None':

                df.at[index, 'Deck'] = df.at[index, 'Cabin'][0]

        except TypeError as e:

            pass

        

    # Drop the columns that contain duplicate data or that I don't think add any value

    # For you 'don't think' should mean, 'carefully tested multiple times and have been clearly demonstrated to reduce performance'

    df.drop(['Deck', 'Cabin', 'ticket_firsthalf', 'ticket_secondhalf'], axis=1, inplace=True)



    return df
# Apply the data prep code to obtain the features we will use to train our model

df_train = prepdata(df_train)

df_test = prepdata(df_test)
# Have a look at them, always interesting, always a good idea to make sure we know what is going into our models

df_train.head()
# Never hurts to make sure the data still looks how we expect, it is surprisingly easy to mistakingly drop a column

df_train.shape
# The dependent variable is what we will try to predict

dep_var = 'Survived'

#cat_names = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'surname', 'salutation', 'ticket_firsthalf', 'ticket_secondhalf', 'Deck' ]

cat_names = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'surname', 'salutation']
# The validation set will be take as a random sample from the provided data, we need the indicies within training dataframe

valid_idx = random.sample(range(1, len(df_train)), 200)



# Processors are functions that will be applied to the data up front. 

procs = [FillMissing, Categorify, Normalize]



# This code creates a structure that contains our fully prepared data ready for the model

data = TabularDataBunch.from_df('/kaggle/output/', df_train, dep_var, valid_idx, test_df=df_test, procs=procs, cat_names=cat_names)

print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}
# Create a model that will be appropriate for our data

# wd = weight decay



learn = tabular_learner(data, layers=[50,2], metrics=accuracy, wd=0.1, ps=[0.7], emb_drop=0.5)
# Find a good learning rate

learn.lr_find()

learn.recorder.plot()
# Fit your model to the data

#learn.fit_one_cycle(7, 5e-3)

learn.fit_one_cycle(14, 8e-2)

learn.fit_one_cycle(20, 5e-4)

#learn.fit_one_cycle(50, 5e-3)
# Examine the losses from the training (only shows you the most recent training run)

learn.recorder.plot_losses()
# Calculate predictions on the test set. The test set is data that was provided to us by Kaggle, it is seperate from the training 

# Data because it has no labels. We will calculate predictions for these passengers and submit the results.

preds, _ = learn.get_preds(ds_type=DatasetType.Test)

pred_prob, pred_class = preds.max(1)

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':pred_class})
submission.head()
# Output our submission to csv

submission.to_csv('my_submission.csv', index=False)