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
df=pd.read_csv('/kaggle/input/train-cleaned/train_cleaned.csv')

df.head()
X=df.drop(columns='Survived')

y=df['Survived']

print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

print(X_train.shape)

print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier().fit(X_train, y_train)

y_predict=model.predict(X_test)
from sklearn.metrics import classification_report

report=classification_report(y_predict, y_test)

print(report)
from sklearn import metrics

precision, recall, thresholds=metrics.precision_recall_curve(y_test, y_predict)

fpr, tpr, thresholds= metrics.roc_curve(y_test, y_predict)
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(8,4))

plt.subplot(121)

plt.plot(recall, precision)

plt.xlabel('recall')

plt.ylabel('precision')

plt.title('Precision-Recall Curve')

plt.subplot(122)

plt.plot(tpr,fpr)

plt.xlabel('TPR')

plt.ylabel('FPR')

plt.title('ROC Curve')

plt.subplots_adjust(wspace = 0.5 )

auc=metrics.auc(precision, recall)

print("Area under Precision-Recall Curve:",auc)