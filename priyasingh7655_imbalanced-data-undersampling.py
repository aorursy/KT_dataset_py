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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

from imblearn.under_sampling import NearMiss

from collections import Counter
dataset=pd.read_csv('../input/creditcardfraud/creditcard.csv')

dataset.head()
#Plot graph to check the number of records in each category

import seaborn as sns

sns.countplot(data=dataset,x='Class')
#Counting number of fraud and non fraud transactions

fraud = dataset[dataset['Class']==1]

non_fraud= dataset[dataset['Class']==0]



print('Number of fraud transactions : {}'.format(fraud['Class'].count()))

print('Number of Non-fraud transactions : {}'.format(non_fraud['Class'].count()))
#Splitting the datset to dependent and independent features and dividing to train test data

X=dataset.iloc[:,:-1]

Y=dataset.iloc[:,-1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30,random_state=50)
#Creating logistic regression model



model=LogisticRegression()

model.fit(X_train,Y_train)

pred=model.predict(X_test)
#Function to check the performance of the model created

def model_performance(Y_test,pred):

    print('Model Accuracy score : {}'.format(accuracy_score(Y_test,pred)))

    print('Model Precision score : {}'.format(precision_score(Y_test,pred)))

    print('Model Recall score : {}'.format(recall_score(Y_test,pred)))

    print('Model F1 score score : {}'.format(f1_score(Y_test,pred)))

    print('Model ROC_AUC_SCORE score : {}'.format(roc_auc_score(Y_test,pred)))



#Check model performance

model_performance(Y_test,pred)
#Using Undersampling to balance the dataset

nm = NearMiss()

X_res,y_res=nm.fit_sample(X,Y)
print('Original dataset shape {}'.format(Counter(Y)))

print('Resampled dataset shape {}'.format(Counter(y_res)))
#Again creating model on the balanced dataset

X_res_train,X_res_test,Y_res_train,Y_res_test=train_test_split(X_res,y_res,test_size=0.30,random_state=50)



new_model=LogisticRegression()

new_model.fit(X_res_train,Y_res_train)

new_pred=new_model.predict(X_res_test)
#Checking model performance on the balanced dataset

model_performance(Y_res_test,new_pred)