import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder 

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from xgboost.sklearn import XGBRegressor
df=pd.read_csv('../input/air-quality-data-in-india/city_hour.csv')
df.head()
df.describe()
df.info()
df.isnull().sum()
df.dropna(how="any",inplace=True)
df.describe()
df.info()
#df
df["City"].unique()
df['Datetime'].unique()
#sns.pairplot(df,hue="AQI_Bucket",vars=['PM2.5','PM10','NO','Benzene'],palette='husl')
a=pd.get_dummies(df['City'],drop_first=True)
a.describe()
#type(df['City'])
#df.iloc[:3]
#pd.get_dummies(data=df,columns=['City','Datetime'])
#df.info()
df=df.drop('Datetime',axis=1)
df=pd.get_dummies(data=df,columns=['City'])
#df.head()
df.describe()
sns.heatmap(df.corr(),linewidths=1)
# in this we can see that AQI is highly correlated with PM 2.5 and PM 10
df.info()
#x=df.drop(["AQI_Bucket","AQI","City_Amaravati","City_Amritsar","City_Chandigarh","City_Delhi","City_Gurugram","City_Hyderabad","City_Kolkata","City_Patna","NOx","NH3","CO","SO2","O3","Benzene","Toluene","Xylene"],axis=1)
#y=df["AQI_Bucket"]
#model=LogisticRegression()
#model.fit(x_train,y_train)
#predictions=model.predict(x_test)
#print(confusion_matrix(y_test, predictions))

#print(accuracy_score(y_test, predictions))
x=df.drop(['AQI_Bucket'],axis=1)

x.head()
y=df.iloc[:,13]

y.head()

y.unique()
params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]  

}
grid_size_per_parameter  = [len(i) for i in params.values()]

print(grid_size_per_parameter)

np.prod(grid_size_per_parameter)
classifier =xgb.XGBClassifier()
random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='accuracy',n_jobs=-1,cv=5,verbose=3)
df.head()
type(random_search)
#df.info()
random_search.estimator
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=1)
random_search.fit(x_train,y_train)
random_search.best_estimator_
#random_search.best_params_
classifier=xgb.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.5, gamma=0.2, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.05, max_delta_step=0, max_depth=10,

              min_child_weight=1, missing=None, monotone_constraints=None,

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='multi:softprob', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=None, subsample=1,

              tree_method=None, validate_parameters=False, verbosity=None)
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier,x_test,y_test,cv=10)
score
score.mean()*100
score.std()*100