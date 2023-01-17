# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 150)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_filepath = '../input/covid19-global-forecasting-week-2/train.csv'

test_filepath = '../input/covid19-global-forecasting-week-2/test.csv'
train_df = pd.read_csv(train_filepath, index_col="Id")

test_df = pd.read_csv(test_filepath, index_col="ForecastId")
train_df.rename(columns={'Country_Region':'Country'}, inplace=True)

test_df.rename(columns={'Country_Region':'Country'}, inplace=True)   

train_df.rename(columns={'Province_State':'State'}, inplace=True)

test_df.rename(columns={'Province_State':'State'}, inplace=True)



EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



# %% [code]

train_df['State'].fillna(EMPTY_VAL, inplace=True)

train_df['State'] = train_df.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



train_df['Date'] = pd.to_datetime(train_df['Date'], infer_datetime_format=True)



train_df.loc[:, 'Date'] = train_df.Date.dt.strftime("%m%d")

train_df["Date"]  = train_df["Date"].astype(int)



train_df.head()



# %% [code]

test_df['Date'] = pd.to_datetime(test_df['Date'], infer_datetime_format=True)

test_df['State'].fillna(EMPTY_VAL, inplace=True)

test_df['State'] = test_df.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)

test_df.loc[:, 'Date'] = test_df.Date.dt.strftime("%m%d")

test_df["Date"]  = test_df["Date"].astype(int)

test_df.head()
y_train_cc = np.array(train_df['ConfirmedCases'].astype(int))

y_train_ft = np.array(train_df['Fatalities'].astype(int))

cols = ['ConfirmedCases', 'Fatalities']



full_df = pd.concat([train_df.drop(cols, axis=1), test_df])

index_split = train_df.shape[0]

full_df = pd.get_dummies(full_df, columns=full_df.columns)



x_train = full_df[:index_split]

x_test= full_df[index_split:]

#x_train.shape, x_test.shape, y_train_cc.shape, y_train_ft.shape
full_df.head()
from sklearn.tree import DecisionTreeClassifier



dtcla = DecisionTreeClassifier()

# We train model

dtcla.fit(x_train, y_train_cc)
predictions = dtcla.predict(x_test)
dtcla.fit(x_train,y_train_ft)
predictions1 = dtcla.predict(x_test)
submission = pd.DataFrame({'ForecastId': test_df.index,'ConfirmedCases':predictions,'Fatalities':predictions1})

filename = 'submission.csv'



submission.to_csv(filename,index=False)