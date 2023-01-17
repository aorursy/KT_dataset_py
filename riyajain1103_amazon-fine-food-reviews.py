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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')

df.head()
df1=df[['HelpfulnessNumerator','HelpfulnessDenominator','Score','Summary','Text']]

df1.head()

df1['NotHelpful']=df1['HelpfulnessDenominator'] - df1['HelpfulnessNumerator']

df1.info()
df1.head()
X=df1[['HelpfulnessNumerator','NotHelpful','Text','Summary']]

y=df1['Score']

y.head()

X.head()
import seaborn as sbn
import matplotlib.pyplot as plt


corr=df1.corr()
sbn.heatmap(corr,annot=True)

X.shape
X.isnull().sum()
work=X['Text']
work.head
work.isnull().any()
len(work)
X.shape
from textblob import TextBlob

scores=[]
for i in range(0,len(work)):
    scores.append(TextBlob(work[i]).sentiment.polarity)
X.shape

len(scores)
X['sentiment score']=scores
X1=X[['HelpfulnessNumerator','NotHelpful','sentiment score']]

X1.head()

y.head()
eda=X1
eda['y']=y
eda.head()
corr=eda.corr()
sbn.heatmap(corr,annot=True)
sbn.distplot(X['HelpfulnessNumerator'],kde=True)

sbn.distplot(X['sentiment score'])

sbn.distplot(y,kde=True)

sbn.distplot(y)

sbn.relplot(x='y',y='sentiment score',hue='HelpfulnessNumerator',data=eda)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.1, random_state = 0)
from xgboost import XGBRegressor
regressor=XGBRegressor(n_estimators=1000,max_depth =3)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
print(y_pred)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 2)
accuracies.mean()


