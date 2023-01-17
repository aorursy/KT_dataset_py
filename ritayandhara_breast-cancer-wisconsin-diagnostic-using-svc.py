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
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
df.sample(4)
df.info()
# Checking for missing values

df.isnull().sum()
# dropping column "Unnamed: 32" as it is empty

df.drop(columns="Unnamed: 32", inplace=True)
df["diagnosis"].unique()
# labelling diagnosis values with 0 and 1 for trainning

df.replace({"M":0,"B":1}, inplace=True)
# Extracting input inputs and label

X = df.iloc[:,2:]

y = df.iloc[:,1:2].iloc[:,-1]
# Scalling data



from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=506)
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train,y_train)
# Accuracy of non tuned SV Classifier

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, svc.predict(X_test)))
# Hyper-parameter tunning using GridSearchCV



from sklearn.model_selection import GridSearchCV



svc = SVC()



k = ['rbf', 'linear','poly','sigmoid']

c = range(1,5)

param_grid=dict(kernel=k, C=c)





grid_svc=GridSearchCV(svc,param_grid=param_grid, cv=10, n_jobs=-1)



grid_svc.fit(X_train,y_train)



print(grid_svc.best_estimator_,"\n")

print(grid_svc.best_params_,"\n")

print(grid_svc.best_score_,"\n")
# Accuracy of tunned model

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, grid_svc.predict(X_test)))