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

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



%matplotlib inline

df = pd.read_csv("../input/training-set/train.csv",header=0, delimiter=r",")
df.head()


df = df.loc[:,'col_0':]



#df = df.drop_duplicates()

X_train = df.loc[:, df.columns != 'target']

Y_train = df.loc[:, df.columns == 'target']

X_train = X_train.to_numpy()

Y_train.values.ravel()

X_train.shape



from imblearn.under_sampling import NearMiss 

nr = NearMiss() 







from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler



X_train.shape

#scaler = preprocessing.MinMax()

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)     



scaled_X_train.shape

#Undersampling



from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler(sampling_strategy=0.01)

#X_train_miss, Y_train_miss = under.fit_sample(scaled_X_train, Y_train) 



from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 2) 

X_train, Y_train = sm.fit_sample(X_train, Y_train) 

scaled_X_train.shape

df1 = pd.read_csv("../input/test-data/test.csv",header=0, delimiter=r",")

df1 = df1.loc[:,'col_0':]
df1.head()
X_test = pd.DataFrame(df1).to_numpy()

 

    





scaled_X_train.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



#logreg = LogisticRegression()

logreg = LogisticRegression(max_iter=1000, C=1000,solver='newton-cg',random_state=42,n_jobs=-1)

parameter_grid = {'C': [0.01, 0.1, 1, 2, 10, 100] }

logreg_cv=GridSearchCV(logreg, parameter_grid, scoring = 'roc_auc' )

Y_train = np.ravel(Y_train)

#logreg_cv.fit(X_train, Y_train )

logreg.fit(X_train, Y_train)
y_pred = logreg.predict_proba(X_test)



#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier()

#clf.fit(X_train, Y_train)

#y_pred = clf.predict(X_test)

#from sklearn.ensemble import AdaBoostClassifier

# Create adaboost classifer object

#abc =AdaBoostClassifier(n_estimators=50,learning_rate=1)



# Train Adaboost Classifer

#model = abc.fit(X_train, Y_train)



#Predict the response for test dataset

#y_pred = model.predict(X_test)



target = pd.DataFrame(y_pred)

target.to_csv("pred.csv")

