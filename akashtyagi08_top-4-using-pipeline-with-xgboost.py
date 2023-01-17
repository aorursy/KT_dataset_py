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
import pandas as pd

import numpy as np



train = pd.read_csv("../input/home-data-for-ml-course/train.csv")

test = pd.read_csv("../input/home-data-for-ml-course/test.csv")
train.shape
print(train.head())

print(test.head())
train.describe()
train.columns
from sklearn.model_selection import train_test_split



train = train.iloc[:,1:] # we do not need ID in training data

train_df, valid_df = train_test_split(train,test_size=0.2)
X_train = train_df.iloc[:,:-1]

y_train = train_df.iloc[:,-1]



X_test = valid_df.iloc[:,:-1]

y_test = valid_df.iloc[:,-1]
y_train.describe()
X_train.isnull().any()
X_train.info()
import seaborn as sns

sns.heatmap(X_train.isnull(),cbar=False)
X_train.drop(['Alley','PoolQC','MiscFeature'], axis=1, inplace=True)

X_test.drop(['Alley','PoolQC','MiscFeature'], axis=1, inplace=True)
X_train.info()
# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == 'object']
categorical_cols
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



missing_values = SimpleImputer(strategy="most_frequent")



categorical_values = Pipeline(steps=

                            [('missing_vals',missing_values),

                             ('Categorical',OneHotEncoder(handle_unknown="ignore"))])





preprocessing_pipeline = ColumnTransformer(transformers=

                                          [('numerical',missing_values,numerical_cols),

                                           ('categorical',categorical_values,categorical_cols)])



from xgboost import XGBRegressor



xgb_model = XGBRegressor(max_depth=7, n_estimators=800, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.01)



# Adding model to pipeline



final_pipeline = Pipeline(steps=

                         [('preprocessing',preprocessing_pipeline),

                          ('model',xgb_model)])



# Any model specific parameter can be added by model__

final_pipeline.fit(X_train, y_train

#                    model__early_stopping_rounds=5

                  )



y_predict = final_pipeline.predict(X_test)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test,y_predict)

print(mae)
test_id = test["Id"]
test.drop(['Id','Alley','PoolQC','MiscFeature'], axis=1, inplace=True)
predictions = final_pipeline.predict(test)
predictions
x = {"Id":test_id,"SalePrice":predictions}

submission = pd.DataFrame(x)
submission.head()
submission.to_csv('submission.csv', index=False)