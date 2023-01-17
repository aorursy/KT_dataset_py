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
dataset=pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv")

dataset.head()

#dataset.isnull().sum()

#void of null values
X=dataset.iloc[:,0:-1]

Y=dataset.iloc[:,-1]

X.head()

Y.head()

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)





dataset.head()

#classification using SVM

from sklearn.svm import SVC

classifier=SVC(kernel='rbf')

classifier.fit(xtrain,ytrain)

ypred=classifier.predict(xtest)

print(ypred)

from sklearn.metrics import accuracy_score



from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=300)

classifier.fit(xtrain,ytrain)

ypred1=classifier.predict(xtest)

print(accuracy_score(ypred1,ytest)*100)

#85 percent of accuracy in randomforest