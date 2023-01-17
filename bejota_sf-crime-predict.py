# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # viz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv",parse_dates=['Dates'])
# get rid of bogus lat/long values
train_df = train_df[(train_df.X != -120.5) & (train_df.Y != 90)]
# add columns which may be useful
train_df['Year'] = train_df.Dates.dt.year
train_df['Hour'] = train_df.Dates.dt.hour
train_df['Month'] = train_df.Dates.dt.month
train_df['DayOfMonth'] = train_df.Dates.dt.day
train_df['DayOfWeekInt'] = train_df.Dates.dt.dayofweek
# convert category to codes
train_df['PdDistrict'] = train_df['PdDistrict'].astype("category")
train_df['PdDistrictInt'] = train_df.PdDistrict.cat.codes
#train_df['Category'] = train_df['Category'].astype("category")
#train_df['CategoryInt'] = train_df.Category.cat.codes
train_df.columns
train_df.isnull().values.any()
from sklearn.cross_validation import train_test_split
sub_train_df, sub_test_df = train_test_split(train_df, test_size = 0.5)
predictors = ['PdDistrictInt', 'X', 'Y', 'Year', 'Hour', 'Month', 'DayOfMonth', 'DayOfWeekInt']
target = ['Category']
X = sub_train_df[predictors]
Y = sub_train_df[target]
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion='entropy',min_samples_split=20,random_state=99,max_depth=10)
dt.fit(X,Y)
dt.score(sub_test_df[predictors],sub_test_df[target])
dt.feature_importances_
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
y = sub_train_df.Category.ravel()
rf.fit(X,y)
rf.score(sub_test_df[predictors],sub_test_df[target])
rf.feature_importances_
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
kn.fit(X,y)
kn.score(sub_test_df[predictors],sub_test_df[target])
test_df = pd.read_csv("../input/test.csv",parse_dates=['Dates'])
# get rid of bogus lat/long values
test_df = test_df[(test_df.X != -120.5) & (test_df.Y != 90)]
# add columns which may be useful
test_df['Year'] = test_df.Dates.dt.year
test_df['Hour'] = test_df.Dates.dt.hour
test_df['Month'] = test_df.Dates.dt.month
test_df['DayOfMonth'] = test_df.Dates.dt.day
test_df['DayOfWeekInt'] = test_df.Dates.dt.dayofweek
# convert category to codes
test_df['PdDistrict'] = test_df['PdDistrict'].astype("category")
test_df['PdDistrictInt'] = test_df.PdDistrict.cat.codes
#test_df['Category'] = test_df['Category'].astype("category")
#test_df['CategoryInt'] = test_df.Category.cat.codes
test_df.columns
predictors = ['PdDistrictInt', 'X', 'Y', 'Year', 'Hour', 'Month', 'DayOfMonth', 'DayOfWeekInt']
target = ['Category']
X = train_df[predictors]
Y = train_df[target]
dt = DecisionTreeClassifier(criterion='entropy',min_samples_split=20,random_state=99,max_depth=10)
dt.fit(X,Y)
result = pd.DataFrame(dt.predict_proba(test_df[predictors]), index=test_df.Id, columns=dt.classes_)
result
#result.to_csv('bejota_SFC_submission.csv', index=True)
