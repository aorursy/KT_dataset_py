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
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/loan-approval/01Exercise1.csv')

df.head()
print('Rows     :',df.shape[0])

print('Columns  :',df.shape[1])

print('\nFeatures :\n     :',df.columns.tolist())

print('\nMissing values    :',df.isnull().values.sum())

print('\nUnique values :  \n',df.nunique())
plt.figure(figsize=(10,10))

sns.heatmap(df.isnull(),cbar=False,cmap='YlGnBu')

plt.ioff()
df.isnull().sum()
df.shape
df_prep =df.copy() 

df_prep.head()
df_prep = df_prep.dropna()

df_prep.shape
df_prep = df_prep.drop(['gender'],axis=1)
df_prep = pd.get_dummies(df_prep,drop_first=True)

df_prep.head()
from sklearn.preprocessing import StandardScaler 

sc_X=StandardScaler()

df_prep['income'] = sc_X.fit_transform(df_prep[['income']])

df_prep['loanamt'] = sc_X.fit_transform(df_prep[['loanamt']])
df_prep.columns
df_prep['status_Y'].value_counts()
X = df_prep.drop(labels='status_Y',axis=1)

y = df_prep['status_Y']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state= 1234,stratify=y)
from sklearn.linear_model import LogisticRegression

lr =LogisticRegression(random_state=0)

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

#y_pred


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_test,y_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

pass
score = lr.score(X_test,y_test)

print("Accuracy score of the model is:",score)
cr =classification_report(y_test,y_pred)

print("Classification Report")

print(cr)
y_prob = lr.predict_proba(X_test)[:,1]

#y_prob
# Classification based on the probability values



y_new_pred = []

threshold = 0.8 



for i in range(0,len(y_prob)):

    if y_prob[i]> threshold:

        y_new_pred.append(1)

    else:

        y_new_pred.append(0)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm1=confusion_matrix(y_test,y_new_pred)

import seaborn as sns

import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm1,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

pass
score = lr.score(X_test,y_new_pred)

print("Accuracy score of the model is:",score)
cr =classification_report(y_test,y_new_pred)

print("Classification Report")

print(cr)
from sklearn.metrics import roc_curve,roc_auc_score



fpr,tpr,threshold = roc_curve(y_test,y_prob)



auc = roc_auc_score(y_test,y_prob)     
import matplotlib.pyplot as plt 

plt.plot(fpr,tpr,linewidth=4)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve for Loan Prediction")

plt.grid()
cm
auc