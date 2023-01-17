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
linkedin_data_original= pd.read_csv('/kaggle/input/linkedin-profiles-and-jobs-data/dump.csv')

linkedin_data_original
linkedin_data_original.describe()
linkedin_data_new= linkedin_data_original.fillna(axis=0, value=linkedin_data_original.mean())

linkedin_data_new.describe()
from sklearn.model_selection import train_test_split





y= linkedin_data_new.followersCount



features=['ageEstimate', 'companyFollowerCount', 'companyStaffCount', 'avgMemberPosDuration' , 'avgCompanyPosDuration']

X= linkedin_data_new[features]



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
from sklearn.ensemble import RandomForestClassifier



model_1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model_1.fit(train_X, train_y)

predictions_rfc= model_1.predict(val_X)

predictions_rfc
from sklearn.metrics import mean_absolute_error as mae



mae_val_rfc= mae(val_y, predictions_rfc)

print(mae_val_rfc)
from sklearn.tree import DecisionTreeClassifier



model_2= DecisionTreeClassifier(max_depth=5, random_state=1)

model_2.fit(train_X, train_y)

predictions_dtc=model_2.predict(val_X)

predictions_dtc
mae_val_dtc= mae(val_y, predictions_dtc)

print(mae_val_dtc)