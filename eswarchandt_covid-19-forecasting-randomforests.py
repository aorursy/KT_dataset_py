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
import numpy as np

import pandas as pd
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

train.tail()

# Format date

train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

train["Date"]  = train["Date"].astype(int)

train.head()
# drop nan's

train = train.drop(['Province_State'],axis=1)

train = train.dropna()

train.isnull().sum()
# Do same to Test data

test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

test["Date"]  = test["Date"].astype(int)

# deal with nan's for lat and lon

#test = test.dropna()

test.isnull().sum()



x = train[[ 'Date']]

y1 = train[['ConfirmedCases']]

y2 = train[['Fatalities']]

x_test = test[['Date']]
from sklearn.ensemble import RandomForestClassifier

Tree_model = RandomForestClassifier(max_depth=200, random_state=0)
##

Tree_model.fit(x,y1)

pred1 = Tree_model.predict(x_test)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]
pred1.head()




##

Tree_model.fit(x,y2)

pred2 = Tree_model.predict(x_test)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Death_prediction"]





Sub = pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

sub_new = Sub[["ForecastId"]]

sub_new
# submit



submit = pd.concat([pred1,pred2,sub_new],axis=1)

submit.head()

# Clean

submit.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

submit = submit[['ForecastId','ConfirmedCases', 'Fatalities']]



submit["ConfirmedCases"] = submit["ConfirmedCases"].astype(int)

submit["Fatalities"] = submit["Fatalities"].astype(int)


submit.describe()

Sub = submit

Sub.to_csv('submission.csv', index=False)
Sub.head()