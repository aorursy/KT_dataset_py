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
import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics
# Data Preprocessing - Data import



glass_data = pd.read_csv('/kaggle/input/glass/glass.csv')

glass_data.head()
print(glass_data.describe())

print(glass_data.dtypes)

glass_data.Type.value_counts()
mval = glass_data.isnull()

mval.head()
sns.heatmap(data = mval, yticklabels=False, cbar=False, cmap='viridis')
glass_data['glass_type'] = glass_data.Type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})

glass_data.head()
# create "Glass correlation Marxix"

columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'glass_type']

matrix = np.zeros_like(glass_data[columns].corr(), dtype=np.bool) 

matrix[np.triu_indices_from(matrix)] = True 

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Glass Correlation Matrix',fontsize=25)

sns.heatmap(glass_data[columns].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", 

            linecolor='b',annot=True,annot_kws={"size":8},mask=matrix,cbar_kws={"shrink": .9});


y = glass_data.glass_type

X = glass_data.loc[:,['Al','Ba', 'Na']]

logistic_model = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=100)

output_model = logistic_model.fit(X_train, y_train)

output_model
score = output_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

logistic_model.fit(X_train,y_train)

y_predict = logistic_model.predict(X_test)

y_predict
cm = confusion_matrix(y_test,y_predict)

print(cm)
print(classification_report(y_test,y_predict))
conf_matrix = metrics.confusion_matrix(y_test,y_predict)

conf_matrix

%matplotlib inline

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
y = glass_data.glass_type

X = glass_data.loc[:,['Ba', 'Al']]

logistic_model1 = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=100)

output_model1 = logistic_model.fit(X_train, y_train)

output_model1
score = output_model1.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

logistic_model1.fit(X_train,y_train)

y_predict = logistic_model1.predict(X_test)

y_predict
cm = confusion_matrix(y_test,y_predict)

print(cm)

print(classification_report(y_test,y_predict))
conf_matrix = metrics.confusion_matrix(y_test,y_predict)

conf_matrix

%matplotlib inline

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')