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
import numpy as np 
import pandas as pd

credit = pd.read_csv("../input/creditcard.csv") 
print("no.of fraud transaction:", len(credit[credit['Class']==1]))
print("no.of normal transaction:", len(credit[credit['Class']==0]))
print("no.of total transaction:",len(credit))
from sklearn.preprocessing import StandardScaler
credit['norm_amount'] =  StandardScaler().fit_transform(credit['Amount'].reshape(-1, 1))
from sklearn.model_selection import train_test_split
training_features,test_features,training_target,test_target = train_test_split(credit.drop(['Class'],axis=1),credit['Class'],test_size=0.1)
#over sampling only on training data
x_train_os,x_test_os,y_train_os,y_test_os = train_test_split(training_features,training_target,test_size=0.1) 
# now apply SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio=1)
x_train,y_train = sm.fit_sample(x_train_os,y_train_os)
from sklearn.ensemble import RandomForestClassifier
rfc  = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
pred = rfc.predict(test_features)
from sklearn.metrics import confusion_matrix,classification_report,recall_score

print(recall_score(test_target,pred))
print(classification_report(test_target,pred))
print(confusion_matrix(test_target,pred))
