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
# importing all modules

import pandas as pd

import numpy as np

import os

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/glass/glass.csv')

df
# count of unique rows for each TYPE of Glass

df['Type'].value_counts()
# checking for null values

missing_value = df.isnull()

missing_value
# heat map for checking any null values

sns.heatmap(data = missing_value , yticklabels=False , cbar=False , cmap='viridis')
#Convert the target feature into a binary feature



df['Type'] = df.Type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})
# splitting dataset into features and target variable 

feature_columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']

X = df[feature_columns]

y = df[['Type']]
# splitting X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=100)
# importing Logistic Model and its Object creation and training it

from sklearn.linear_model import LogisticRegression

logr_model = LogisticRegression()

logr_model.fit(X_train,y_train)

y_pred=logr_model.predict(X_test)
# import the metrics class

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
# ROC Curve

y_pred_proba = logr_model.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()