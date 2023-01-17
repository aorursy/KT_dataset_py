# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import roc_auc_score



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading csv file

df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")

df.head()
#analysing data

df.info()
df.describe()
#dropping id's 

df.drop(columns = "id", inplace = True)

df.head()
X = df[df.columns[0:88]]

y = df.target

print("X shape: ",(X.shape))

print("Y shape: ",(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)

X_test = scalar.transform(X_test)
#sampling to remove skewing of data

under = RandomUnderSampler(sampling_strategy=0.75,random_state=42)

X_train, y_train = under.fit_resample(X_train,y_train)

over = SMOTE(random_state=42)

X_train,y_train = over.fit_sample(X_train, y_train)

print("Number of 0's: ",(y_train==0).sum())

print("Number of 1's: ",(y_train==1).sum())
#training model using logistic regression

parameters={"C":np.logspace(-3,3,7), "penalty":["l2"],"solver":["newton-cg"]}



model = GridSearchCV(LogisticRegression(max_iter=1000), parameters, cv=5, scoring=make_scorer(roc_auc_score), iid=True,n_jobs=-1)

model.fit(X_train, y_train)
#checking auc score on train_test data

pred=model.predict_proba(X_test)[:,1]

print("AUC score: ",roc_auc_score(y_test,pred))
#predictions for test.csv

df = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")

X_final = df[['col_'+str(i) for i in range(88)]]

X_final_scaled = scalar.transform(X_final)

y_final = model.predict_proba(X_final_scaled)[:,1]
#checking number of elements

print(y_final.shape)
idcol = list(df['id'])

y_finalcol = list(y_final)

temp = {'id':idcol,'target':y_finalcol}

submission = pd.DataFrame(temp)

submission.head()

submission.to_csv('submission_3.csv',index=False)