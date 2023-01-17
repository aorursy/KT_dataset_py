# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import random

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

random.seed(10)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import glob

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt









#input data from file

inputfile = "../input/minor-project-2020/train.csv"

df = pd.read_csv(inputfile)

df.head()
df.drop('id',axis=1,inplace=True) #removing the id as its not a feature of an instance

X = df[['col_'+str(i) for i in range(88)]] #making the X and y for the data

y = df['target'] 

X = X.reset_index(drop=True)

y = y.reset_index(drop=True)

print("0: " + str((y==0).sum()))

print("1: " + str((y==1).sum()))

print("x shape : " + str(X.shape))

print("y shape : " + str(y.shape))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.3) 

print(X_train.shape)

print("0's in test: " + str((y_test==0).sum()))

print("1's in test: " + str((y_test==1).sum()))
from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler(feature_range = (0,1))

# X_train = scalar.fit_transform(X_train)

# X_test = scalar.transform(X_test)

#scaling gives worse results


from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

#oversampler = RandomOverSampler(sampling_strategy=0.1,random_state=0)

#undersampler = RandomUnderSampler(sampling_strategy=0.3,random_state=0)

#X_train, y_train = oversampling.fit_resample(X_train,y_train)



#X_train, y_train = undersampler.fit_resample(X_train,y_train)

print("before 0: " + str((y_train==0).sum()))

print("before 1: " + str((y_train==1).sum()))



os = SMOTE(random_state=0) #following the pattern of 0 seed each time

X_train,y_train = os.fit_sample(X_train, y_train)

print("after 0: " + str((y_train==0).sum()))

print("after 1: " + str((y_train==1).sum()))



print(X_train.shape)

print(y_train.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score

param_grid={"C":[0.1,0.001,0.0001,1,10,100,1000,10000],"solver":["newton-cg","saga","sag"]}



# model = GridSearchCV(LogisticRegression(max_iter=10000), param_grid, cv=10, scoring=make_scorer(roc_auc_score), iid=True,n_jobs=-1)

# print(model.cv_results_)

# print(model.best_score_)

# print(model.best_params_)

#the above grid search returns C as 1000 , and newton-cg as best solver.

model = LogisticRegression(C=1000,solver="newton-cg") #using best model after grid search.

model.fit(X_train, y_train)
pred=model.predict_proba(X_test)
print(roc_auc_score(y_test,pred[:,1]))

#FOR PROB VALUES
testData = '../input/minor-project-2020/train.csv'

df = pd.read_csv(testData)

df.head()

X_test = df[['col_'+str(i) for i in range(88)]]
y_pred = model.predict_proba(X_test)

keys = list(df['id'])

target = list(y_pred[:,1])

dictionary = {'id':keys,'target':target}

df1 = pd.DataFrame(dictionary)

df1.to_csv('submission.csv',index=False)
print(y_pred[:,1])