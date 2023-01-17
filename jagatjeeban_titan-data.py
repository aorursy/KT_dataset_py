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
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
import seaborn as sns 

import matplotlib.pyplot as plt

data.info()
sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
sex = pd.get_dummies(data['Sex'],drop_first=True)

embarked = pd.get_dummies(data['Embarked'],drop_first=True)
data.drop(['Sex','Embarked','Cabin'],axis = 1,inplace = True)
data.head()
data = pd.concat([data,sex,embarked],axis = 1)
data.head(2)
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=data,palette='winter')
def add_age(colm):

    Age = colm[0]

    Pclass = colm[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 30

        else:

            return 24

    else:

        return Age

      
data['Age'] = data[['Age','Pclass']].apply(add_age,axis=1)
sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
data.drop(['Ticket','Name'],axis=1,inplace = True)
data.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop('Survived',axis=1), 

                                                    data['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression()

logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))