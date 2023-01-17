import warnings

warnings.filterwarnings('ignore')
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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv('/kaggle/input/asian-and-indian-cuisines/asian_indian_recipes.csv')

data.head()
data.tail()
data.info()
data.shape
X = data.iloc[:,2:]

X.head()
y = data[['cuisine']]

y.head()
from sklearn.model_selection import train_test_split

# from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB,GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
models = [

    BernoulliNB(),

    DecisionTreeClassifier(criterion='gini'),

    RandomForestClassifier(n_estimators=100),

    SVC(kernel='linear')

]
CV = 10

cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []

i=0

for model in models:

    model_name = model.__class__.__name__

    accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV) 

    entries.append([model_name, accuracies.mean()])

    i += 1

cv_df = pd.DataFrame(entries, columns=['model_name', 'accuracy'])
cv_df
plt.figure(figsize=(10,5))

ax=sns.barplot(x="accuracy", y="model_name", data=cv_df)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
model_nba = BernoulliNB(binarize = .5)

model_nba.fit(X_train,y_train)
y_pred = model_nba.predict(X_test)
print('accuracy:',accuracy_score(y_test,y_pred))
#kiem tra do chinh xac

print('training r^2 :',model_nba.score(X_train,y_train))

print('test r^2 :',model_nba.score(X_test,y_test))
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
from sklearn.ensemble import AdaBoostClassifier
boosting = AdaBoostClassifier(n_estimators=100,

                             base_estimator=model_nba,

                             learning_rate=1)
model_new = boosting.fit(X_train,y_train)
print('training r^2 :',model_new.score(X_train,y_train))
print('test r^2 :',model_new.score(X_test,y_test))
df_new = X.iloc[:2,:]

df_new.iloc[:,:] = 0
df_new.loc[0,['cumin','fish']] = 1

df_new.loc[1,['cumin']] = 1
df_new
X_new = df_new

y_new = model_nba.predict(X_new)
y_new
labels = model_nba.classes_

labels
print(confusion_matrix(y_test,y_pred,labels=model_nba.classes_))
sns.heatmap(confusion_matrix(y_test,y_pred,labels=model_nba.classes_))