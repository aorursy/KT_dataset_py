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
df = pd.read_csv("../input/HR-Employee-Attrition.csv")
df.info()
import warnings

warnings.filterwarnings('ignore')

df.Attrition.replace({"Yes":1,"No":0}, inplace=True)
df.drop(columns=['EmployeeCount','StandardHours'], inplace=True)

df.columns
cat_col = df.select_dtypes(exclude=np.number).columns

num_col = df.select_dtypes(include=np.number).columns
for i in cat_col:

    print(df[i].value_counts())

    print("------------------------------------")
encoded_cat_col = pd.get_dummies(df[cat_col])
final_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report
x = final_model.drop(columns="Attrition")

y = final_model["Attrition"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

train_Pred = model.predict(x_train)
metrics.confusion_matrix(y_train,train_Pred)
Accuracy_percent_train = (metrics.accuracy_score(y_train,train_Pred))*100

Accuracy_percent_train
test_Pred = model.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
Accuracy_percent_test = (metrics.accuracy_score(y_test,test_Pred))*100

Accuracy_percent_test
print(classification_report(y_test, test_Pred))