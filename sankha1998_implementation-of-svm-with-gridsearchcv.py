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
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.corr()
import seaborn as sns

sns.heatmap(df.corr())
corr = df.corr()
## corr value less than 0.8 selected

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.8:

            if columns[j]:

                columns[j] = Falseselected_columns = df.columns[columns] = df[selected_columns]
df.head()
y = df.Outcome.values
X=df.drop(columns=['Outcome'],axis=1).values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, test_size=0.2)


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(X_train)
X_train.shape
y.shape
from sklearn.svm import SVC

model = SVC(kernel='linear',C=10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

matrix = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test, y_pred))

print (matrix)


def cparameter(i):

    model = SVC(kernel='linear',C=i)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)

    print(accuracy_score(y_test, y_pred))

    print (matrix)

l = [0.1,1,10]

for i in l:

    print("for the value of c = ", i)

    cparameter(i)
# # for the value of c=10 accuracy 
param_grid = [

  {'C': [0.1,1,10,100], 'gamma': [0.01,0.001], 'kernel': ['rbf','sigmoid']},

 ]
from sklearn.model_selection import GridSearchCV

grid=GridSearchCV(model,param_grid=param_grid, cv=10, n_jobs=-1)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test, y_pred))

print (matrix)
grid.best_score_
grid.best_params_
grid.best_estimator_