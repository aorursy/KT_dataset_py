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
#import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/bank.csv', delimiter= ";", header = 'infer')

df.head()
sns.heatmap(df.isnull(),yticklabels = False, cmap = 'viridis')
sns.pairplot(df)
df.corr()
sns.heatmap(df.corr())
df.dtypes
df_new = pd.get_dummies(df, columns=['job', 'marital', 'education','default',

                                    'housing','loan','contact','month','poutcome'] )

df_new.y.replace(('yes','no'),(1,0),inplace = True)
df_new.dtypes
#Whole dataset's shape (ie (rows, cols))

print(df.shape)
#Unique education values

df.education.unique()
#Crosstab to display education stats with respect to y ie class variable

pd.crosstab(index=df["education"], columns=df["y"])
df.education.value_counts().plot(kind = 'barh')
#Feature selection



y = pd.DataFrame(df_new['y'])

X = df_new.drop(['y'],axis=1)
# data divide on training and testing sets



from sklearn.model_selection import train_test_split



X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3, random_state = 2)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
from sklearn.metrics import accuracy_score



acc_logmodel = round(accuracy_score(y_pred, y_test) * 100)

print(acc_logmodel)
from sklearn.metrics import confusion_matrix, classification_report



print(confusion_matrix(y_test, y_pred))



print('\n')



print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy Score {}".format(accuracy_score(y_test, y_pred)))
print("Confusion Matrix \n{}".format(confusion_matrix(y_test, y_pred)))
print("Classification Report \n{}".format(classification_report(y_test, y_pred)))