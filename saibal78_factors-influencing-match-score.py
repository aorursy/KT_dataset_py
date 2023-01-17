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
import os
import pandas as pd
import numpy as np
print(os.listdir("../input"))
Data=pd.read_csv("../input/StudentsPerformance.csv",sep=',',header=0)
Data.head()
Data.describe()
#Checking missing values
Data.isnull().sum()
#Since there is no missing values in the dataset we can move to next step to do some predictive analysis
#Here I am doing analysis using RandomForestClassifier to derive factors that are influencing the scores

##################Predicting Math Score using Random Forest Classifier################

#Creating predictor matrix,here math score need to be predicted
X=Data.drop(["math score","reading score","writing score"],axis=1)

#Converting non numeric data fields into numeric
X=pd.get_dummies(X)
#Creating target vector
Y=Data['math score']

#Splitting data into test and training components
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=200)

#Importing Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
for w in range (10,100,10):
    clf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=200)
    clf.fit(X_train,Y_train)
    oob=clf.oob_score_
    print(w)
    print(oob)
    print('**********************')
    
#n_estimators 40 has given the best oob score, hence resetting the model using best estimators as 40
clf=RandomForestClassifier(n_estimators=40,oob_score=True,n_jobs=-1,random_state=200)
clf.fit(X_train,Y_train)

#list out all feature importance directly in an array format, no need to run a loop like bagged classifier
clf.feature_importances_
#list out all feature importance directly in a list format
imp_feat=pd.Series(clf.feature_importances_,index=X.columns.tolist())
#Sorting feature importance values and showing in a Bar diagram
%matplotlib inline
imp_feat.sort_values(ascending=False).plot(kind='bar')
