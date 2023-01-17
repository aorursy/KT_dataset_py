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
data=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.shape
data.isnull().sum()
data.head()
data.DEATH_EVENT.value_counts().plot(kind='bar')
X=data.copy()

X.drop(columns=['DEATH_EVENT'],axis=1,inplace=True)



y=data['DEATH_EVENT']
#SMOTE Technique

from imblearn.over_sampling import SMOTE

# transform the dataset

oversample = SMOTE()

X, y = oversample.fit_resample(X, y)
from sklearn.model_selection import train_test_split



# split data into train and test sets

seed = 7

test_size = 0.40

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
from xgboost import XGBClassifier



model = XGBClassifier(max_depth=10,random_state=1,learning_rate=0.05)

model.fit(X_train, y_train)
# make predictions for test data

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



print("Accuracy",accuracy_score(y_pred,y_test))

print("Precission",precision_score(y_pred,y_test))

print("Recall",recall_score(y_pred,y_test))

print("F1 Score",f1_score(y_pred,y_test))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred,y_test)
from sklearn.metrics import classification_report



print(classification_report(y_pred,y_test))
from sklearn.metrics import roc_curve,auc



fpr, tpr, threshold = roc_curve(y_test, y_pred)



roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.metrics import log_loss

log_loss(y_test,y_pred)