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
import numpy as np

import pandas as pd

data = pd.read_csv("/kaggle/input/petrol-consumption/petrol_consumption.csv")

data
data.head()
x=data.iloc[:,0:4].values

y=data.iloc[:, 4].values
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=0.2 , random_state= 0)
from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor

regressor=RandomForestRegressor(n_estimators=20,random_state= 0)

regressor.fit(X_train,Y_train)

Y_Pre=regressor.predict(X_test)
from sklearn import metrics

print(' Mean Absolute Error',metrics.mean_absolute_error(Y_test,Y_Pre))

print(' Mean Absolute Error',metrics.mean_squared_error(Y_test,Y_Pre))

print(' Mean Absolute Error',np.sqrt(metrics.mean_squared_error(Y_test,Y_Pre)))
print("* 2nd part*")
dt = pd.read_csv("/kaggle/input/bill_authentication/bill_authentication.csv")

dt
dt.head()
X=dt.iloc[:,0:4].values

Y=dt.iloc[:,4].values
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2 , random_state= 0)
from sklearn.preprocessing import StandardScaler

sci= StandardScaler()

x_train=sci.fit_transform(x_train)

x_test=sci.transform(x_test)
from sklearn.ensemble import RandomForestClassifier

regresso=RandomForestClassifier(n_estimators=20,random_state= 0)

regresso.fit(x_train,y_train)

pre=regresso.predict(x_test)
print(classification_report(y_test,pre))
confusion_matrix(y_test,pre)
print('accuracy score :',accuracy_score(y_test,pre)*100)