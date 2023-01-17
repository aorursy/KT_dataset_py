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
#load the dataset into the dataframe
data_frame = pd.read_excel('../input/Cryotherapy.xlsx')
#inspect the data_frame
data_frame.info()
#visualization of the feature variables
import seaborn as sns
sns.pairplot(data_frame)
#splitting the data_frame into features and target
target = data_frame['Result_of_Treatment']
features = data_frame.drop('Result_of_Treatment',axis=1)
#function for evaluating the performance of the model, here we will be using an fbeta_score as performance metric
from sklearn.metrics import fbeta_score
def performance_metric(y_true,y_pred):
    
    score = fbeta_score(y_true,y_pred,beta=0.5)
    
    return score
#importing an adaboost classifier
from sklearn.ensemble import AdaBoostClassifier

#train and test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,train_size=0.8,random_state=10)
# creating an instance of the ada boost classifier
clf = AdaBoostClassifier()

# fitting the model using the training data
clf.fit(X_train,y_train)

#predicting against the input data
clf_pred = clf.predict(X_test)

# estimating the performance of the  metric
clf_score = performance_metric(y_test,clf_pred)
print('AdaBoost Classifier has a performance of {:.3f}'.format(clf_score))