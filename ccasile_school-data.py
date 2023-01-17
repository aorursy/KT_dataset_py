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



data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
y = data['test preparation course']

X = data

X.drop(columns=['test preparation course'], inplace=True)

y = y.map(dict(completed=1, none=0))

print(y)
data.isna().sum()
X = pd.get_dummies(X, drop_first=True)

X.head()
X.corr()
X.drop(columns=['math score', 'reading score'], inplace=True)
from sklearn import preprocessing



cols = X.columns

x = X.values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, index=X.index, columns=cols)

X.head()
from sklearn.model_selection import train_test_split

X = X.astype(float)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=0)



logreg.fit(X_train,y_train)



predictions = logreg.predict(X_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_test, predictions)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, predictions)
import statsmodels.api as sm

from scipy import stats



X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())