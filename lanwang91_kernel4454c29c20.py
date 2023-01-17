# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

# Modelling Algorithms



from sklearn.svm import SVC, LinearSVC

from sklearn import linear_model





# Modelling Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale

from sklearn.feature_selection import RFECV

from sklearn.feature_extraction import DictVectorizer



# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# get TMDB Box Office Prediction train & test csv files as a DataFrame

train = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/train.csv")

test  = pd.read_csv("/kaggle/input/tmdb-box-office-prediction/test.csv")
# Splitting into Test and validation data and feature selection



# Selecting features Budget and Popularity

#train_mod = train[{"budget","popularity"}]



# Selecting features Budget and Popularity





# Selecting the first 2001 indices of the training data for training

train_train = train_mod[0:2000]

# Selecting the rest of the training data for validation

train_val= train_mod[2001:2999]



# Obtain labels

train_mod_y = train[{"revenue"}]

train_train_y = train_mod_y[0:2000]

train_val_y= train_mod_y[2001:2999]

train_val_title = train["original_title"][2001:2999]

train_genres = train[{"genres"}]

#print(train_genres.genres.unique())

#print(train_genres)

#vec = DictVectorizer()

#train_genres.head()

p = re.compile(r'\d+')

idList = []

for index, row in train_genres.iterrows():

    #print(row)

    for col in row.iteritems():

        if(type(col[1]) is not str):

            continue

        if(len(col[1]) <= 2):

                continue

        allIDs = p.findall(col[1])

        for id in allIDs:

            idNum = eval(id);

            if idNum not in idList:

                idList.append(idNum)

#train_genres.genres.unique()



print(idList)

Column_Names = []

for id in idList:

    name= 'Genre'+ str(id)

    Column_Names.append(name)



train_genre_list = pd.DataFrame()

idx = 0

for name in Column_Names:

    train_genre_list.insert(idx,name,np.zeros(len(train_genres)))

    idx = idx + 1



#print(train_genre_list)

for rowidx, row in train_genres.iterrows():

    for col in row.iteritems():

        if(type(col[1]) is not str):

            continue

        if(len(col[1]) <= 2):

                continue

        allIDs = p.findall(col[1])

        for id in allIDs:

            idNum = eval(id);

            train_genre_list.loc()

            train_genre_list[Column_Names[idList.index(idNum)]].loc[rowidx]=1

            #print(each_row)   

    #train_genre_list.

#vec.fit_transform(train_genres).toarray()



print(train_genre_list)

# Check for NaN

if(train_mod.isnull().values.any()):

    print("Too bad, Nan found...")

else :

    print("All right!!! Data ok!")





# Initialize and train a linear regression (Lasso) model

model = linear_model.Lasso(alpha=0.1)

model.fit(train_train,train_train_y.values.ravel())
# Evaluate on the training data

res = model.predict( train_val)

# Obtain R2 score (ordinary least square)

model.score(train_val, train_val_y)
# Create the table for comparing predictions with labels

evaluation = pd.DataFrame({'Title': train_val_title.values.ravel(),'Prediction': res.round(), 'Actual revenue': train_val_y.values.ravel(), 'Relative error': res/train_val_y.values.ravel()})

evaluation