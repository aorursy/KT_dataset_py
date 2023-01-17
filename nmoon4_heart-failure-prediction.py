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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.shape
df.dtypes
df['DEATH_EVENT'].value_counts()
X = df.drop('DEATH_EVENT', axis=1)
X.head()
y = df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
import statsmodels.api as sm
logit_model = sm.Logit(y_train, X_train).fit()
logit_model.summary()
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
LR.score(X_test, y_test)
X = df[['age', 'ejection_fraction', 'serum_creatinine']]
X.corr()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logit_model = sm.Logit(y_train, X_train, axis=1).fit()
logit_model.summary()
LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
LR.score(X_test, y_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, yhat)
df.head()
df['DEATH_EVENT'].value_counts()
X = df.drop('DEATH_EVENT', axis=1).values
y = df['DEATH_EVENT'].values

# for the ML try with all variables vs only the good variables

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 7
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
k_range = 30
mean_acc = np.zeros((k_range-1))
ConfustionMx = [];
for n in range(1, k_range):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

mean_acc
plt.plot(range(1,k_range),mean_acc,'g')
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (k)')
plt.tight_layout()
plt.show()
