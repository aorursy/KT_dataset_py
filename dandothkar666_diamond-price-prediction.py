# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV,train_test_split

from sklearn.linear_model import Lasso,Ridge,ElasticNet

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/diamond.csv')
df.head()
df=df.drop(['Unnamed: 0'],axis=1)
df.head()
df.dtypes
df.select_dtypes(include='object').isnull().sum()
df.select_dtypes(exclude='object').isnull().sum()
for i in df.dtypes[df.dtypes=='object'].index:

    sns.countplot(y=i,data=df)

    plt.show()
sns.boxplot(x='price',y='cut',data=df)

plt.show()
sns.boxplot(x='price',y='clarity',data=df)

plt.show()
sns.boxplot(x='price',y='color',data=df)

plt.show()
import plotly.express as px

fig=px.scatter_3d(df,x='color',y='price',z='cut',color='price')

fig.show()
df.groupby('clarity').agg(['mean','std'])
df.groupby('cut').agg(['mean','std'])
df.groupby('color').agg(['mean','std'])
df=df[df.x!=0]

df=df[df.z!=0]

df=df.rename(columns={'x':'length'})

df=df.rename(columns={'y':'width'})

df=df.rename(columns={'z':'depth'})
df=pd.get_dummies(df,columns=['clarity','cut','color'])
df.head()
pipelines={

    'lasso':make_pipeline(StandardScaler(),Lasso(random_state=123)),

    'ridge':make_pipeline(StandardScaler(),Ridge(random_state=123)),

    'enet':make_pipeline(StandardScaler(),ElasticNet(random_state=123)),

    'rf':make_pipeline(StandardScaler(),RandomForestRegressor(random_state=123)),

    'gb':make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state=123))

}
lasso_hyperparameters={

    'lasso__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]

}

ridge_hyperparameters={

    'ridge__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10]

}

enet_hyperparameters={

    'elasticnet__alpha':[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10],

    'elasticnet__l1_ratio':[0.1,0.3,0.5,0.7,0.9]

}
rf_hyperparameters={

    'randomforestregressor__n_estimators':[100,200],

    'randomforestregressor__max_features':['auto','sqrt',0.33]

}



gb_hyperparameters={

    'gradientboostingregressor__n_estimators':[100,200],

    'gradientboostingregressor__learning_rate':[0.005,0.1,0.2],

    'gradientboostingregressor__max_depth':[1,3,5]

}
hyperparameters={

    'rf':rf_hyperparameters,

    'gb':gb_hyperparameters,

    'lasso':lasso_hyperparameters,

    'ridge':ridge_hyperparameters,

    'enet':enet_hyperparameters

}
X=df.drop('price',axis=1)

y=df.price
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
fitted_models={}

for name,pipeline in pipelines.items():

    model=GridSearchCV(pipeline,hyperparameters[name],cv=10,n_jobs=-1)

    model.fit(X_train,y_train)

    fitted_models[name]=model

    print(name,'Model has been fitted. ')
for name,model in fitted_models.items():

    pred=model.predict(X_test)

    print(name)

    print('--------')

    print('R^2 Score:  ',r2_score(y_test,pred))

    print('MAE:  ',mean_absolute_error(y_test,pred))