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
#Reading a csv file

df=pd.read_csv("/kaggle/input/dphi-amsterdam-airbnb-data/airbnb_listing_train.csv")
#Checking for NaN in DataFrame 

df.isnull().sum()
#Printing the first five rows  

df.head()
#importing libraries

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
#Creating a transformer for numerical and categorical columns

nt=SimpleImputer(strategy="mean")

ct=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])

#import libraries to create models(RandomForestRegressor works best)

#from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor

#from sklearn.tree import DecisionTreeRegressor

#Separating the input and target variables

X=df.drop(["price"],axis=1)

y=df.price
#train test split

from sklearn.model_selection import train_test_split

# Break off validation set from training data

X_test_full=pd.read_csv("../input/dphi-amsterdam-airbnb-data/airbnb_listing_validate.csv")

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Select categorical columns 

c = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]



# Select numerical columns

n = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = c + n

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
#Creating a column transformer for a DataFrame

preprocessor=ColumnTransformer(transformers=[("num",nt,n),("cat",ct,c)])


pipe=Pipeline(steps=[("preprocessor",preprocessor),("rfr",RandomForestRegressor())])

pipe.fit(X_train,y_train)

#The hyperparamter tuning did not improve the output

#from sklearn.model_selection import RandomizedSearchCV

#param_grid={"rfr__n_estimators":[i for i in range(100,1000,100)],"rfr__max_depth":[i for i in range(3,11)]}

#search=RandomizedSearchCV(pipe,param_grid,n_jobs=-1,cv=5,verbose=2,scoring="neg_mean_absolute_error")

#search.fit(X_train,y_train)
#pipe.get_params().keys()
#search.best_params_
#search.best_score_


pred=pipe.predict(X_valid)

print(pred)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_valid,pred))
preds=pipe.predict(X_test)
output=pd.DataFrame({"id":X_test.id,"Price":preds})

output.to_csv("submission.csv",index=False)