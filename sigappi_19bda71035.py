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
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing


train_data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv')

test_data = pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')





train_data.head(10)
train_data.describe()
col_name_X = [col_name for col_name in train_data.columns]

col_name_X


train_X_all = train_data[col_name_X]



train_X_all=train_X_all.drop(['flag'],axis=1)

train_Y_all = train_data.flag

train_X_all
#split the data into test and train

train_x, valid_x, train_y, valid_y = train_test_split(train_X_all,train_Y_all,random_state=13)

valid_x
rf_model = RandomForestClassifier(random_state=0)

rf_model.fit(train_x,train_y)
feature_importances = pd.DataFrame(rf_model.feature_importances_,

                                   index = train_x.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances
features = ['currentBack','currentFront','trackingDeviationBack','positionBack','trackingDeviationFront','positionFront','motorTempBack','velocityFront','velocityBack']
train_data_1 = train_data[features]

train_data_1


train_x_1, valid_x_1, train_y_1, valid_y_1 = train_test_split(train_data_1,train_Y_all,random_state=13)

rf_model_1 = RandomForestClassifier(random_state=0)

rf_model_1.fit(train_x_1,train_y_1)
#Predict value and get f1-score

valid_pred = rf_model_1.predict(valid_x_1)

f1_score(valid_y_1,valid_pred)

confusion_matrix(valid_y_1,valid_pred )


test_data_1 = test_data[features]

test_data_1
pred_test = rf_model_1.predict(test_data_1)
sample = pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')

sample['flag'] = pred_test

sample.to_csv('submit_1.csv',index=False)