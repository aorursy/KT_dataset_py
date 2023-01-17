import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import random

np.random.seed(0)

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
data=pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head().T
del data['#']
data.head(10)
data.isnull().sum()
data=data.dropna(subset=['Name'])
data.isnull().sum()
data.loc[data['Type 2'].isnull()]
plt.figure(figsize=(10,6))

sns.countplot(data['Type 2'])

plt.show()
data=data.fillna('#####')

type_02=['Poison', 'Dragon', 'Ground', 'Fairy', 'Grass',

       'Fighting', 'Psychic', 'Steel', 'Ice', 'Rock', 'Dark', 'Water',

       'Electric', 'Fire', 'Ghost', 'Bug', 'Normal']
data['Type 2']=data['Type 2'].apply(lambda x: x if x!='#####' else random.choice(type_02))
plt.figure(figsize=(10,6))

sns.countplot(data['Type 2'])

plt.show()
data.head()
data['First_pokemon']=le.fit_transform(data['Name'])
data.head()
fight=pd.read_csv('/kaggle/input/pokemon-challenge/combats.csv')
print(fight.shape)

fight.head()
data_merged=data.merge(fight,on='First_pokemon',how='inner')
del data_merged['Name']
cat_cols=data_merged.select_dtypes(include='object')

cat_cols.head()
data_merged=pd.concat([data_merged.drop(cat_cols,axis=1),cat_cols.apply(le.fit_transform)],axis=1)
X=data_merged.drop('Winner',axis=1)

y=data_merged['Winner']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=5)
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
param={'n_estimators':np.arange(1,10),'min_samples_split':np.arange(2,5),'min_samples_leaf':np.arange(1,10)}
search=GridSearchCV(estimator=rf,param_grid=param,return_train_score=True).fit(x_train,y_train)
search.best_params_
plt.figure(figsize=(10,7))

pd.DataFrame(search.cv_results_).set_index('params')['mean_test_score'].plot.line()

pd.DataFrame(search.cv_results_).set_index('params')['mean_train_score'].plot.line()

plt.xticks(rotation=90)

plt.show()
y_pred=search.predict(x_test)
r2_score(y_test,y_pred)
from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(x_train,y_train)
y_pred=xgb.predict(x_test)
r2_score(y_test,y_pred)