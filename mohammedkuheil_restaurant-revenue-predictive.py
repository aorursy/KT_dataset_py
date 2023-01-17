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
import zipfile

with zipfile.ZipFile('../input/restaurant-revenue-prediction/train.csv.zip', 'r') as zip_obj1:

   # Extract all the contents of zip file in current directory

   zip_obj1.extractall('/kaggle/working/')
train_data = pd.read_csv('./train.csv',index_col='Id',parse_dates=['Open Date'])

train_data
with zipfile.ZipFile('../input/restaurant-revenue-prediction/test.csv.zip', 'r') as zip_obj2:

   # Extract all the contents of zip file in current directory

   zip_obj2.extractall('/kaggle/working/')

test_data = pd.read_csv('./test.csv',index_col='Id',parse_dates=['Open Date'])

test_data
df= train_data

df
print ('  Data Types        ')

print(df.dtypes)

print ('----------------------------------')

print ('         counts of Missing values')

print (df.isna().sum())

print ('----------------------------------')

print ('         Numbers of unique values')

print ([[col,len (df[col].unique())] for col in df.columns])
for col in df.columns:

    print (col,' : ', len (df[col].unique()))
object_columns = df.select_dtypes(include='object')

object_columns
# convert all data (any df) in one step for uniform data

lable_columns = ['City', 'City Group', 'Type']

label_df = df[lable_columns]

from sklearn.preprocessing import LabelEncoder

label_df = label_df.apply(LabelEncoder().fit_transform)

label_df
df_final = pd.concat([label_df,df.drop(columns=['City', 'City Group', 'Type'])],axis=1)

df_final
y = df_final['revenue']

X = df_final.drop(columns=['revenue','Open Date'])

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.9, test_size=0.1,random_state=0)
from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)

my_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)

from sklearn.metrics import mean_absolute_error



predictions = my_model.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
# convert all data (any df) in one step for uniform data

label_df1 = test_data[lable_columns]

from sklearn.preprocessing import LabelEncoder

label_df1 = label_df1.apply(LabelEncoder().fit_transform)

label_df1
df_final_test = pd.concat([label_df1,test_data.drop(columns=['City', 'City Group', 'Type','Open Date'])],axis=1)

df_final_test.dtypes
df_final_test
preds = my_model.predict(df_final_test)

print(preds.shape)

print(df_final_test.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),

                              ('model', RandomForestRegressor(n_estimators=500,

                                                              random_state=0))

                             ])

from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("MAE scores:\n", scores)

my_pipeline.fit(X_train, y_train)
PRES = my_pipeline.predict(df_final_test)

print(PRES.shape)

print(df_final_test.shape)
submission = pd.DataFrame({

        "Id": df_final_test.index,

        "Prediction": PRES

    })

submission.to_csv('CrossValidation.csv',header=True, index=False)

print('Done')
