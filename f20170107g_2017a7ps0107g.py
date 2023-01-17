import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')
df.head()
df.info()
df.describe()
df.describe(include='object')
y=df['rating'].copy()
y
df.drop(['id','rating'], axis=1, inplace=True)

df.head()
y
misscc=df.isnull().sum()

misscc[misscc>0]
df.fillna(value=df.mean(),inplace=True)

df.head()
df.fillna(value=df.mode().loc[0],inplace=True)

df.head()
misscc1=df.isnull().sum()

misscc1[misscc1>0]
numerical_features=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features=['type']

X=df[numerical_features+categorical_features]
X=pd.get_dummies(data=X,columns=['type'])

X.head()
X.info()
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import ShuffleSplit

from sklearn import datasets

from sklearn.metrics import mean_squared_error



from sklearn import ensemble
params={'n_estimators':500, 'max_depth': 4,'min_samples_split':2,'learning_rate':0.01, 'loss':'ls'}



model=ensemble.GradientBoostingRegressor(**params)



model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error, r2_score



model_score=model.score(X_train,y_train)



print('R2 sq: ',model_score)

y_pred=model.predict(X_val)



print("Mean squared error: %.2f"% mean_squared_error(y_val, y_pred))



print('Test Variance score: %.2f' % r2_score(y_val, y_pred))
from sklearn.model_selection import GridSearchCV

def GradientBooster(param_grid, n_jobs):

    estimator = ensemble.GradientBoostingRegressor()

    

    cv = ShuffleSplit(X_train.shape[0], test_size=0.2)

    

    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs)

    

    classifier.fit(X_train, y_train)

    

    print ("Best Estimator learned through GridSearch") 

    print ('classifier.best_estimator_' )

    

    return cv,classifier.best_estimator_
param_grid={'n_estimators':[100],

            'learning_rate': [0.1],

            'max_depth':[6],

            'min_samples_leaf':[3],

             'max_features':[1.0]} 



n_jobs=4



cv,best_est= GradientBooster(param_grid, n_jobs)
print("Best Estimator Parameters")

print("---------------------------" )

print("n_estimators: %d" %best_est.n_estimators) 

print("max_depth: %d" %best_est.max_depth) 

print("Learning Rate: %.1f" %best_est.learning_rate) 

print("min_samples_leaf: %d" %best_est.min_samples_leaf) 

print("max_features: %.1f" %best_est.max_features) 



print()

print("Train R-squared: %.2f" %best_est.score(X_train,y_train))
estimator = best_est

estimator.fit(X_train, y_train)



print ("Train R-squared: %.2f" %estimator.score(X_train, y_train))

print ("Test R-squared: %.2f" %estimator.score(X_val, y_val))
tt=estimator.predict(X_val)
tt.round(0)
df2=pd.DataFrame()

df2['rating']=tt.round(0)
df2['rating'].value_counts()
df1=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df1.head()
df1.drop('id',axis=1, inplace=True)

df1.head()
df1.fillna(value=df1.mean(),inplace=True)

df1.head()
df1.fillna(value=df1.mode().loc[0],inplace=True)

df1.head()
df1.isnull().sum()
numerical_features1=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11']

categorical_features1=['type']

X1=df1[numerical_features1+categorical_features1]
X1=pd.get_dummies(data=X1,columns=['type'])

X1.head()
estimator = best_est

estimator.fit(X_train, y_train)



print ("Train R-squared: %.2f" %estimator.score(X_train, y_train))

print ("Test R-squared: %.2f" %estimator.score(X_val, y_val))
tt1=estimator.predict(X1)
tt1.round(0)
df3=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df3.head()
df4=pd.DataFrame(index=df3['id'])

df4['rating']=tt1.round(0)

df4['rating'].value_counts()
df4.to_csv('sub1.csv')
#for second best solution -
from sklearn.model_selection import train_test_split



X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.33,random_state=42)
params={'n_estimators':500, 'max_depth': 4,'min_samples_split':2,'learning_rate':0.01, 'loss':'ls'}



model=ensemble.GradientBoostingRegressor(**params)



model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error, r2_score



model_score=model.score(X_train,y_train)



print('R2 sq: ',model_score)

y_pred=model.predict(X_val)



print("Mean squared error: %.2f"% mean_squared_error(y_val, y_pred))



print('Test Variance score: %.2f' % r2_score(y_val, y_pred))
y_pred.astype('int64')