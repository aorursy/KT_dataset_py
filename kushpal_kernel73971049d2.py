# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import json



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


pets = pd.read_csv('../input/train/train.csv')

pets2 = pd.read_csv('../input/test/test.csv')

breads = pd.read_csv('../input/breed_labels.csv')

colors = pd.read_csv('../input/color_labels.csv')

stats = pd.read_csv('../input/state_labels.csv')
pets.shape
pets.head(2)
pets2.shape
breads.shape
breads.head(5)
colors.shape
colors.head(5)
stats.shape
stats.head(3)
pets.columns
pets.isnull().sum()
pets_E_N=pets[pets.Name.isnull()==True]
pets_E_N.head(3)
curr_path= '../input/train_metadata/'

curr_names=os.listdir('../input/train_metadata')
#curr_w_json=curr_names.split('-')
pets.Name=pets.Name.fillna(method = 'bfill', axis=0)
pets=pets.drop(['Description'],axis=1)
pets.isnull().sum()
pets.Name=pets.Name.replace(regex=True,inplace=True,to_replace=r'<',value=r'')

le = preprocessing.LabelEncoder()

le.fit(pets.Name)

pets.Name=le.transform(pets.Name)

pets.dtypes
le = preprocessing.LabelEncoder()

le.fit(pets.RescuerID)

pets.RescuerID=le.transform(pets.RescuerID)
le = preprocessing.LabelEncoder()

le.fit(pets.PetID)

pets.PetID=le.transform(pets.PetID)
pets.dtypes
X=pets.drop(['AdoptionSpeed'],axis=1)

y=pets['AdoptionSpeed']
from sklearn.metrics import cohen_kappa_score

cf = RandomForestClassifier(class_weight = 'balanced', n_estimators = 10000, random_state = 42)

cf.fit(X,y)
pets2=pets2.drop(['Description'],axis=1)

pets2.Name=pets2.Name.replace(regex=True,inplace=True,to_replace=r'<',value=r'')

le = preprocessing.LabelEncoder()

le.fit(pets2.Name)

pets2.Name=le.transform(pets2.Name)

le = preprocessing.LabelEncoder()

le.fit(pets2.RescuerID)

pets2.RescuerID=le.transform(pets2.RescuerID)

le = preprocessing.LabelEncoder()

le.fit(pets2.PetID)

pets2.PetID=le.transform(pets2.PetID)
pets
y_pred=cf.predict(pets2)

#cf.score(X_test,y_test)

pid=list(le.inverse_transform(pets2.PetID))
#cohen_kappa_score(y_pred, y_test)
df = {'PetID': pid, 'AdoptionSpeed': y_pred}

df = pd.DataFrame(df)

df.AdoptionSpeed = df.AdoptionSpeed.astype('int32')
df.AdoptionSpeed.value_counts(dropna= False)
df.to_csv('submission.csv', index = False)