# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer

from sklearn.model_selection import GridSearchCV
df= pd.read_csv('../input/eval-lab-2-f464/train.csv')

plt.figure(figsize=(15,15))

sns.heatmap(data=df.corr(),annot=True)
x= df.drop(['id','chem_4','chem_6','class'], axis=1)

y = df['class']





from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.33,random_state=0)
#scaler = StandardScaler()

scaler = RobustScaler()

#scaler = Normalizer()

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.transform(X_test) 
# feature_importances = pd.DataFrame(reg_best.feature_importances_,index = range(0,9),columns=['importance']).sort_values('importance',ascending=False)

# feature_importances
rfc=RandomForestClassifier(random_state=42)
X_test.shape
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 10000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=420, n_jobs = -1)



rf_random.fit(X_train, Y_train)
rf_random.best_params_
random_forest = RandomForestClassifier(n_estimators= 3466, min_samples_split=2, min_samples_leaf=1,max_features='sqrt', max_depth=None, bootstrap= True)

random_forest.fit(X_train, Y_train)

y_pred = random_forest.predict(X_test)
from sklearn.metrics import accuracy_score

#accuracy_score(Y_test,y_pred)
import xgboost as xgb

xg_reg = xgb.XGBClassifier( n_estimators = 600)

xg_reg.fit(X_train,Y_train)

y_pred = xg_reg.predict(X_test)
from sklearn.metrics import accuracy_score

#accuracy_score(Y_test,y_pred)
reg_best = ExtraTreesRegressor(max_depth=100,n_estimators=1400,max_features='sqrt')

reg_best.fit(X_train,Y_train)

y_pred = reg_best.predict(X_test)

#accuracy_score(Y_test,np.round(y_pred))
# starting evaluation on testing data
train_df= pd.read_csv('../input/eval-lab-2-f464/train.csv')

test_df= pd.read_csv('../input/eval-lab-2-f464/test.csv')
x_train = train_df.drop(['id','class'], axis=1)

y_train = train_df['class']

x_test = test_df.drop(['id'], axis=1)
rf_best_1 = RandomForestClassifier(n_estimators= 3466, min_samples_split=2, min_samples_leaf=1,max_features='sqrt', max_depth=None, bootstrap= True)

rf_best_1.fit(x_train, y_train)

y_pred = rf_best_1.predict(x_test)
submission = pd.DataFrame({'id':test_df['id'],'class':y_pred})

#submission.head(40)
filename = 'result1.csv'



submission.to_csv(filename,index=False)
x_train = train_df.drop(['id','class','chem_4','chem_6'], axis=1)

y_train = train_df['class']

x_test = test_df.drop(['id','chem_4','chem_6'], axis=1)
rf_best_2 = RandomForestClassifier(n_estimators= 3466, min_samples_split=2, min_samples_leaf=1,max_features='sqrt', max_depth=None, bootstrap= False)

rf_best_2.fit(x_train, y_train)

y_pred = rf_best_2.predict(x_test)
submission = pd.DataFrame({'id':test_df['id'],'class':y_pred})

#submission.head(40)
filename = 'result2.csv'



submission.to_csv(filename,index=False)