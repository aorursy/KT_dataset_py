# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from learntools.ml_intermediate.ex2 import *

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor



from sklearn.metrics import mean_absolute_error

from sklearn import preprocessing





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH_TRAIN = '/kaggle/input/covid19-global-forecasting-week-5/train.csv'

train =pd.read_csv(PATH_TRAIN).fillna('.')

train_c = pd.read_csv(PATH_TRAIN).fillna('.')

PATH_TEST ='/kaggle/input/covid19-global-forecasting-week-5/test.csv'

test_c =pd.read_csv(PATH_TEST).fillna('.')

submission_c = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv').fillna('_')

train_c.head()
train_c.info()
train_c.columns
train_c.tail()
print(train_c.shape)

missing_value_columns = (train_c.isnull().sum())

print(missing_value_columns)

pd.unique(train_c.Target)
train_c.Target.unique()
# coverting the string to date

redate = pd.to_datetime(train_c['Date'], errors='coerce')



train_c['Date'] =redate.dt.strftime('%Y%m%d').astype(int)

# this an alternative to the 'commented codes below' that I used for preprocessing even the yielded same result.



c_tags = train_c["Target"].unique()

for index in range(0, len(c_tags)):

    train_c["Target"].replace(c_tags[index], index, inplace=True)

    

feature_cols_2 = ['Population', 'Weight', 'Date', 'Target']

train_c[feature_cols_2].head()
# preprocessing of the message.



# train_c['Target'].map({'Fatalities': 0, 'ConfirmedCases': 1}) # this can be used to encode Target.



#from sklearn.preprocessing import LabelEncoder



#cat_feature = ['Target']

#encoder = LabelEncoder()



# Apply the label encoder to each column

#encoded = train_c[cat_feature].apply(encoder.fit_transform)



'''The aboved code was used instead of the label encoder. the code in this cell did not contain Target but yielded same'''



# selecting features



#feature_cols = ['Population', 'Weight', 'Date']

#feature_cols_1 = train_c[feature_cols].join(encoded)

#feature_cols_2 = train_c[feature_cols].join(cat_train_c)





y = train_c.TargetValue



X = train_c[feature_cols_2]

# splitting the dataset



from sklearn.model_selection import train_test_split 



X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state =0, train_size=0.8, test_size=0.2)

                                                      

                                                      



from xgboost import XGBRegressor



my_model = XGBRegressor()

my_model.fit(X_train, y_train)

prediction = my_model.score(X_valid, y_valid)

print(str(prediction))
# test data



redate = pd.to_datetime(test_c['Date'], errors='coerce')



test_c['Date'] =redate.dt.strftime('%Y%m%d').astype(int)





for index in range(0, len(c_tags)):

    test_c["Target"].replace(c_tags[index], index, inplace=True)
feature_cols_3 = ['Population', 'Weight', 'Date', 'Target']
# getting data for prediction

data_test = test_c[feature_cols_3]
prediction_test = my_model.predict(data_test)

prediction_test
submission_c = pd.DataFrame({"Id": data_test.index, "TargetValue":prediction_test  })

submission_c.to_csv("submission.csv", index=False)