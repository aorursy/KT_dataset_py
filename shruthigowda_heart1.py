# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

heart = pd.read_csv("../input/heart.csv")
import numpy as np
## first 5 rows of training data

heart.head()
## Number of rows and columns of data

heart.shape
## column names

heart.columns
## find missing values in each column

heart.isna().sum()
## Basic statistical details of data

heart.describe()


print(heart['target'].value_counts())
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
heart['age'] = le.fit_transform(heart['age'])

heart['sex'] = le.fit_transform(heart['sex'])

heart['cp'] = le.fit_transform(heart['cp'])

heart['trestbps'] = le.fit_transform(heart['trestbps'])



heart['chol'] = le.fit_transform(heart['chol'])

heart['fbs'] = le.fit_transform(heart['fbs'])

heart['restecg'] = le.fit_transform(heart['restecg'])

heart['thalach'] = le.fit_transform(heart['thalach'])



heart['target'] = le.fit_transform(heart['target'])

heart['thal'] = le.fit_transform(heart['thal'])

heart['ca'] = le.fit_transform(heart['ca'])

heart['slope'] = le.fit_transform(heart['slope'])



heart['oldpeak'] = le.fit_transform(heart['oldpeak'])

heart['exang'] = le.fit_transform(heart['exang'])
## first 5 rows of training data after preprocessing

heart.head()
## impact is to predict

y=heart['target']

y
## drop the column 'target'

heart.drop(['target'],axis=1,inplace=False)
from sklearn import linear_model

import numpy as np

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(heart,y,test_size=0.2,random_state=400)
model = linear_model.LogisticRegression()

mod = model.fit(xtrain,ytrain)

preds = mod.predict(xtest)
preds
from sklearn.metrics import accuracy_score

accuracy_score(ytest,preds)