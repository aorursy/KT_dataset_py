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
import pandas as pd

import numpy as np

data=pd.read_csv("/kaggle/input/titanic/train.csv")

data.info()
from sklearn.impute import SimpleImputer

si=SimpleImputer(strategy="mean",missing_values=np.nan)

data["Age"]=si.fit_transform(data[["Age"]])


data=data.drop(columns="Cabin")
data=data.drop(columns=["Name","Ticket"])
data=data.drop(columns="Embarked")

data=data.drop(columns="PassengerId")
data.head()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lbE=LabelEncoder()

data["Sex"]=lbE.fit_transform(data["Sex"])

data.head()
OHE=OneHotEncoder(categorical_features=[3],handle_unknown='ignore')

data1=OHE.fit_transform(data)

#data.info()
X=data.iloc[:,1:]

y=data.iloc[:,:1]

X.shape,y.shape
import matplotlib.pyplot as plt

import seaborn as sns
#sns.pairplot(data)
sns.heatmap(data.corr())
from sklearn.linear_model import LogisticRegression,LinearRegression

lr=LogisticRegression(solver='lbfgs')



lr.fit(X,y)

y_pred=lr.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
from sklearn.linear_model import LogisticRegression,LinearRegression

lr=LinearRegression()



lr.fit(X,y)

y_pred=lr.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
from sklearn.tree import DecisionTreeClassifier

trees=DecisionTreeClassifier()



trees.fit(X,y)

y_pred=trees.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))



from sklearn.tree import DecisionTreeClassifier

DecisionTreeClassifier()

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



trees=DecisionTreeClassifier()

svm=SVC(gamma="auto",degree=2)



vcm=VotingClassifier(estimators=[("svm",svm),("tree",trees)],voting="hard")

vcm.fit(X,y)



y_pred=vcm.predict(X)

y_pred=y_pred.round()





from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
from sklearn.ensemble import BaggingClassifier

bag=BaggingClassifier(DecisionTreeClassifier())



bag.fit(X,y)

y_pred=bag.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(learning_rate=1)



ada.fit(X,y)

y_pred=ada.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()



rfc.fit(X,y)

y_pred=rfc.predict(X)

y_pred=y_pred.round()



from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

print("Accuracy:%.2f"%(100* accuracy_score(y,y_pred)))

print("R2 :%.3f"%r2_score(y,y_pred))

print("Counfusion :\n",confusion_matrix(y,y_pred))
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")

data_test.info()
data_test=data_test.drop(columns="Cabin")

data_test=data_test.drop(columns=["Name","Ticket"])

data_test=data_test.drop(columns="Embarked")

data_test=data_test.drop(columns="PassengerId")


data_test.info()
from sklearn.impute import SimpleImputer

si=SimpleImputer(strategy="mean",missing_values=np.nan)

data_test["Age"]=si.fit_transform(data_test[["Age"]])

data_test["Fare"]=si.fit_transform(data_test[["Fare"]])



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

lbE=LabelEncoder()

data_test["Sex"]=lbE.fit_transform(data_test["Sex"])
'''

OHE=OneHotEncoder(categorical_features=[3])

data_test=OHE.fit_transform(data_test)'''
data_test.info()
from sklearn.tree import DecisionTreeClassifier

tr=DecisionTreeClassifier()



tr.fit(X,y)

data_pred=tr.predict(data_test)



data_test["Survived"]=data_pred
data.head(10),data_test.head(10)