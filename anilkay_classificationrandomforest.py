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
data=pd.read_csv("/kaggle/input/run-or-walk/dataset.csv")

data.head()
set(data["username"])
del data["username"]
set(data["date"])
set(data["wrist"])
set(data["activity"])
import seaborn as sns


sns.relplot(data=data,x="gyro_x",y="gyro_z",hue="activity")
newframe=pd.DataFrame({

    "gyrox":data["gyro_x"].abs(),

    "gyroz":data["gyro_z"].abs(),

    "activity":data["activity"]

})

sns.relplot(data=newframe,x="gyrox",y="gyroz",hue="activity")
x=data.iloc[:,4:]

xadd=data["wrist"]

x["wrist"]=xadd

y=data["activity"]
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

print(cross_val_score(rfc, x, y, cv=10)) 
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2000)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)
print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))
rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

print(confusion_matrix(y_pred=ypred,y_true=y_test))

print(accuracy_score(y_pred=ypred,y_true=y_test))

print(classification_report(y_pred=ypred,y_true=y_test))