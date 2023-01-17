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
df=pd.read_csv('/kaggle/input/adult-census-income/adult.csv')
df.info()
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

def tranfrominocme(incomeval):

    if str(incomeval)=='<=50K':

       return 0

    return 1 

 
df['income']=df['income'].apply(tranfrominocme)
sns.pairplot(df)
from sklearn.model_selection import train_test_split
X=df[['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']]

y=df[['income']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
LR =LogisticRegression() 
LR.fit(X_train,y_train)
LR.coef_
predict=LR.predict(X_test)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(LR, X_train, 

         y_train, cv=5)

acc_score= np.mean(scores)
acc_score