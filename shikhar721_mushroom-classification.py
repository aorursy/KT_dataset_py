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
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
data.head()
data.isnull().sum()

data['class'].unique()

print(data.groupby('class').size())



ax = sns.countplot(x="class", data=data) #balanced daaset

ax = sns.countplot(x="class", hue="population", data=data)

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 
data.head()




corrmat = data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat,  square=True,cbar=True)



data=data.drop('veil-type',axis=1)

y=data['class'].values

data=data.drop('class',axis=1)
y




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=0.2,random_state=4)





from xgboost import XGBClassifier

classifier=XGBClassifier()





classifier.fit(X_train, y_train)

classifier.score(X_train, y_train)



from sklearn.model_selection import cross_val_score



scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')



scores

scores.mean()
y_pred=classifier.predict(X_test)
from sklearn import metrics
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc


from sklearn import metrics



confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix