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

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import numpy as np

from sklearn.metrics import mean_squared_log_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import LabelEncoder
train_df=pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

test_df=pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
train_df.head()
train_df.info()
test_df.info()
test_df.head()
print("Min train date: ",train_df["Date"].min())

print("Max train date: ",train_df["Date"].max())

print("Min test date: ",test_df["Date"].min())

print("Max test date: ",test_df["Date"].max())
train_df.isnull().sum()
test_df.isnull().sum()
train_df.rename(columns={'Country_Region':'Country'}, inplace=True)

test_df.rename(columns={'Country_Region':'Country'}, inplace=True)



train_df.rename(columns={'Province_State':'State'}, inplace=True)

test_df.rename(columns={'Province_State':'State'}, inplace=True)
mo = train_df['Date'].apply(lambda x: x[5:7])

da = train_df['Date'].apply(lambda x: x[8:10])

mo_test = test_df['Date'].apply(lambda x: x[5:7])

da_test = test_df['Date'].apply(lambda x: x[8:10])

train_df['day_from_jan_first'] = (da.apply(int)

                               + 31*(mo=='02') 

                               + 60*(mo=='03')

                               + 91*(mo=='04')  

                              )

test_df['day_from_jan_first'] = (da_test.apply(int)

                               + 31*(mo_test=='02') 

                               + 60*(mo_test=='03')

                               + 91*(mo_test=='04')  

                              )
train_df["Date"] = train_df["Date"].apply(lambda x:x.replace("-",""))

train_df["Date"]  = train_df["Date"].astype(int)
train_df.info()
test_df["Date"] = test_df["Date"].apply(lambda x:x.replace("-",""))

test_df["Date"]  = test_df["Date"].astype(int)
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
train_copy = train_df.copy()
train_copy['State'].fillna(EMPTY_VAL, inplace=True)

train_copy['State'] = train_copy.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
train_copy.head()
test_copy = test_df.copy()
test_copy['State'].fillna(EMPTY_VAL, inplace=True)

test_copy['State'] = test_copy.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
test_copy.head()
labelencoder = LabelEncoder()
train_copy['Country'] = labelencoder.fit_transform(train_copy['Country'])

train_copy['State'] = labelencoder.fit_transform(train_copy['State'])
test_copy['Country'] = labelencoder.fit_transform(test_copy['Country'])

test_copy['State'] = labelencoder.fit_transform(test_copy['State'])
train_copy.info()
train_copy.columns
X=train_copy[['State', 'Country', 'Date', 'day_from_jan_first']]
y1=train_copy["ConfirmedCases"] #Confirmed Case

y2=train_copy["Fatalities"]     #Fatalities
#Confirmed Cases

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(X, y1, test_size = .20, random_state = 42)
dt1=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)
dt1.fit(X_train_confirmed, y_train_confirmed)
y_pred_dt_confirmed=dt1.predict(X_test_confirmed)
np.sqrt(mean_squared_log_error( y_test_confirmed, y_pred_dt_confirmed ))
#Fatalities

X_train_fatal, X_test_fatal, y_train_fatal, y_test_fatal = train_test_split(X, y2, test_size = .20, random_state = 42)
dt2=DecisionTreeRegressor(criterion="friedman_mse",max_depth=20,random_state=42)
dt2.fit(X_train_fatal, y_train_fatal)
y_pred_dt_fatal=dt2.predict(X_test_fatal)
np.sqrt(mean_squared_log_error( y_test_fatal, y_pred_dt_fatal ))
test_copy.head()
X_test=test_copy[['State','Country','Date','day_from_jan_first']]
y_confirmed=dt1.predict(X_test)
y_fatal=dt2.predict(X_test)
submission=pd.DataFrame({'ForecastId': test_copy["ForecastId"], 'ConfirmedCases': y_confirmed, 'Fatalities': y_fatal})
submission.head()
submission.to_csv('submission.csv', index=False)