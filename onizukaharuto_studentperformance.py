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
df_train=pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/train.csv',index_col=0)

df_test=pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/test.csv',index_col=0)
df_train.isnull().sum()
df_test.isnull().sum()
import seaborn as sns

from matplotlib import pyplot



sns.set_style("darkgrid")

pyplot.figure(figsize=(20, 20))

sns.heatmap(df_train.corr(), square=True, annot=True)
df_train=df_train[['Medu','Fedu','failures','higher','studytime','age','traveltime','internet','romantic','Dalc','G3']]

df_test=df_test[['Medu','Fedu','failures','higher','studytime','age','traveltime','internet','romantic','Dalc']]
df_train['higher']=pd.get_dummies(df_train['higher'], drop_first=True)

df_test['higher']=pd.get_dummies(df_test['higher'], drop_first=True)

df_train['internet']=pd.get_dummies(df_train['internet'], drop_first=True)

df_test['internet']=pd.get_dummies(df_test['internet'], drop_first=True)

df_train['romantic']=pd.get_dummies(df_train['romantic'], drop_first=True)

df_test['romantic']=pd.get_dummies(df_test['romantic'], drop_first=True)
X=df_train.drop('G3',axis=1).values

y=df_train['G3'].values
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

len(X_train),len(X_valid),len(y_train),len(y_valid)
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR

model = SVR(gamma='auto')

model.fit(X_train, y_train)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(predict, y_valid))
import optuna

def objective(trial):

    #kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])

    svr_c = trial.suggest_loguniform('svr_c', 1e0, 1e2)

    epsilon = trial.suggest_loguniform('epsilon', 1e-1, 1e1)

    #gamma = trial.suggest_loguniform('gamma', 1e-3, 3e1)

    model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)

    return mean_squared_error(y_valid, y_pred)



study = optuna.create_study()

study.optimize(objective, n_trials=100)

study.best_params
svr_c = study.best_params['svr_c']

epsilon = study.best_params['epsilon']

model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')

model.fit(X_train, y_train)

predict = model.predict(X_valid)

np.sqrt(mean_squared_error(predict, y_valid))
X=df_train.drop('G3',axis=1).values

y=df_train['G3'].values

#model = RandomForestRegressor(criterion='mse', max_depth=max_depth, n_estimators=n_estimators, random_state=0,n_jobs=-1)

model = SVR(kernel='rbf', C=svr_c, epsilon=epsilon, gamma='auto')

#model = DecisionTreeRegressor(criterion='mse',splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, random_state=0)

model.fit(X, y)
X_test = df_test.values

predict = model.predict(X_test)
submit = pd.read_csv('/kaggle/input/1056lab-student-performance-prediction/sampleSubmission.csv')

submit['G3'] = predict

submit.to_csv('submission.csv', index=False)