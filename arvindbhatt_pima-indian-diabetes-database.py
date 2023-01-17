import matplotlib.pyplot as plt

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/diabetes.csv')
print(data.head())

#correlation between the features
print(data.corr())

#seems the data is unrelated

#visualization of data
data.hist(figsize=(10,10))
plt.show()
#featuring out important features out
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=0)
X=data[data.columns[:8]]
Y=data['Outcome']
model.fit(X,Y)
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
data=data[['Glucose','BMI','Age','DiabetesPedigreeFunction','Outcome']]
from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.25,random_state=12,stratify=data['Outcome'])
X_train=train.iloc[:,0:4]
y_train=train.iloc[:,4]
X_test=test.iloc[:,0:4]
y_test=test.iloc[:,4]



from sklearn.preprocessing import StandardScaler
sdt=StandardScaler()
sdt.fit(X_train)
X_train=sdt.transform(X_train)
X_test=sdt.transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

X_train.hist(figsize=(10,10))
plt.show()
from sklearn.svm import SVC
model=SVC()
model.fit(X_train,y_train)
model.score(X_test,y_test)