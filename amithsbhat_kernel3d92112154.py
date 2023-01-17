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
import datetime

from dateutil.parser import parse

from dateutil.tz import gettz



def getTS(dt):

    tzinfos = {'UTC' : gettz('Europe/London')}

    date_str = '{0} 00:00:00'.format(dt)

    str_to_dt = parse(date_str + ' UTC', tzinfos=tzinfos)

    return int(str_to_dt.timestamp())



df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv", index_col = "Id" )

df['Date'] = df['Date'].apply(getTS)



countries = list(set(df['Country_Region']))

countries.sort()

countryDict = {each : idx for idx,each in enumerate(countries,1)}

df['Country_Region'] = df['Country_Region'].map(countryDict)

allProvinces = list(set(df['Province_State']))

#allProvinces.sort()

provinceDict = {each : idx for idx,each in enumerate(allProvinces,1)}

df['Province_State'] = df['Province_State'].map(provinceDict)



df.head()

provinceDict
df.isnull().values.any()
feature_col_names = ['Country_Region', 'Date','Province_State'] 

predicted_class_names1 = ['ConfirmedCases']

predicted_class_names2 = ['Fatalities']



X = df[feature_col_names].values

Y1 = df[predicted_class_names1].values

Y2 = df[predicted_class_names2].values



print("Data cleanup done...")
from sklearn.svm import SVR



regr1 = SVR(kernel='poly', C=100, gamma='auto', degree=4, epsilon=.1)





regr1.fit(X, Y1.ravel())







regr2 = SVR(kernel='poly', C=100, gamma='auto', degree=4, epsilon=.1)



regr2.fit(X, Y2.ravel())
testpath = '/kaggle/input/covid19-global-forecasting-week-3/test.csv'



dft = pd.read_csv(testpath, index_col = "ForecastId" )

dft['Date'] = dft['Date'].apply(getTS)

dft['Country_Region'] = dft['Country_Region'].map(countryDict)

dft['Province_State'] = dft['Province_State'].map(provinceDict)

dft.head()



Xt = dft[feature_col_names].values

predictionsC = regr1.predict(Xt) 

 

dft['ConfirmedCases'] = predictionsC



predictionsF = regr2.predict(Xt)

dft['Fatalities'] = predictionsF



allowedCols =['ForecastId','ConfirmedCases','Fatalities']





for col in dft.columns:

    if col not in allowedCols:

        dft = dft.drop([col], axis = 1)

        print("Dropping {0}".format(col))

        

def normalize(val):

    if val > 0:

        return round(val)

    return 0



dft['ConfirmedCases'] = dft['ConfirmedCases'].apply(normalize)

dft['Fatalities'] = dft['Fatalities'].apply(normalize)

dft.to_csv('submission.csv', index = True)



print("Done...")