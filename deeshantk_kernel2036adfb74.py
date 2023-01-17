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
path = '/kaggle/input/covid19-global-forecasting-week-2/'

train = pd.read_csv(path + 'train.csv')

test = pd.read_csv(path + 'test.csv')
def drop(df, to_drop):

    df.drop(to_drop, axis = 1, inplace = True)

    return df
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train['Countries'] = label.fit_transform(train['Country_Region'])

test['Countries'] = label.fit_transform(test['Country_Region'])
train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 

train['dayofyear'] = train['Date'].dt.dayofyear

test['dayofyear'] = test['Date'].dt.dayofyear
y = train['ConfirmedCases']

y1 = train['Fatalities']
drop(train, 'Id')

drop(train, 'Province_State')

drop(train, 'Country_Region')

drop(train, 'Date')

drop(train, 'Fatalities')

drop(train, 'ConfirmedCases')

drop(test, 'ForecastId')

drop(test, 'Province_State')

drop(test, 'Date')

drop(test, 'Country_Region')
train.head()
from sklearn.tree import DecisionTreeRegressor

mel_model = DecisionTreeRegressor(random_state=1)



print(mel_model.fit(train,y))
pred = mel_model.predict(test)
ConfirmedCases = pred.astype(int)

train['ConfirmedCases'] = y

test['ConfirmedCases'] = ConfirmedCases

mel_model.fit(train,y1)

pred = mel_model.predict(test)

Fatalities = pred.astype(int)
res = pd.DataFrame([ConfirmedCases, Fatalities], index = ['ConfirmedCases','Fatalities'], columns= np.arange(1, ConfirmedCases.shape[0] + 1)).T

res.to_csv('submission.csv', index_label = "ForecastId")
#res = pd.DataFrame({'ForecastId':list(range(1, 12643)), 'ConfirmedCases': predictions})
#train['ConfirmedCases'] = y

#test['ConfirmedCases'] = predictions
#mel_model.fit(train,y1)
#pred = mel_model.predict(test)
#predictions = pred.astype(int)
#res['Fatalities'] = predictions
#res.to_csv("submission.csv", index = False)