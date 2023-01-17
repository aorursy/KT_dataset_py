# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import statsmodels.api as sm
import statsmodels.formula.api as smf
df = pd.read_csv('../input/FIFA 2018 Statistics.csv')
print ('There are',len(df.columns),'columns:')
for x in df.columns:
    print(x,end=',')
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.jointplot(x='Attempts',y='Goal Scored',data=df,kind='kde')
sns.distplot(df['Goal Scored'],kde=False,axlabel='Goals scored in a match')
sns.jointplot(x='Ball Possession %',y='Goal Scored',data=df,kind='kde')

plt.figure(figsize=(10,10))
sns.countplot(y='Saves',data=df)

sns.countplot(y='Goal Scored',data=df)
plt.figure(figsize=(10,10))
sns.countplot(y='Fouls Committed',data=df)
df = pd.concat([df,pd.get_dummies(df['Round'])],axis=1)
mom = pd.get_dummies(df['Man of the Match'],drop_first=True)
df = pd.concat([df,mom, pd.get_dummies(df['Round'])],axis=1)
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
X = df.drop(['Date','Team','Opponent','Round','PSO','Man of the Match','1st Goal','Own goals','Own goal Time'],axis=1)
y = df['Man of the Match']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lr = LogisticRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)
print (classification_report(y_test,predictions))
print('Feature Coefficients of the Regression:- \n', lr.coef_[0])























