# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.preprocessing import StandardScaler

from sklearn.grid_search import GridSearchCV

print(check_output(["ls", "../input"]).decode("utf8"))

train_csv = pd.read_csv("../input/train.csv")

test_csv = pd.read_csv("../input/test.csv")

train_csv.head()

# Any results you write to the current directory are saved as output.



#print(ytrain.tolist())
xtrain = train_csv.drop(["id","species"],axis = 1)

xtrain.head()

ytrain = train_csv["species"]

xtest = test_csv.drop("id",1)

xtest.ix[:,:]=StandardScaler().fit_transform(xtest)

testid = test_csv["id"]

xtrain.ix[:,:]=StandardScaler().fit_transform(xtrain)

xtrain
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

#from sklearn.cross_validation import cross_val_score  

from sklearn.decomposition import PCA
#logic = LogisticRegression()

#logic.fit(xtrain,ytrain)

#Ytest = logic.predict_proba(xtest)

#print(Ytest)
forest = RandomForestClassifier(n_estimators=50)

param_grid = { 

    'n_estimators': [100,200,300,400,500],

    'max_features': ['auto', 'log2']

}

CV_forest= GridSearchCV(estimator=forest, param_grid=param_grid, cv=5,scoring='neg_log_loss')

CV_forest.fit(xtrain,ytrain)

Ytest = CV_forest.predict_proba(xtest)

print(Ytest)
#组合成结果

result = pd.DataFrame(Ytest,columns = sorted(list(set(ytrain.tolist()))))

result.insert(0,"id",testid.tolist())

result.head()

result.to_csv('classfysamples.csv', index=False)