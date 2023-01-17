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

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_log_error
df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', index_col='Id')

dtest=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv', index_col='ForecastId')
y1=df['ConfirmedCases']

y2=df['Fatalities']

df.drop('ConfirmedCases', axis=1, inplace=True)

df.drop('Fatalities', axis=1, inplace=True)
df['check']=1

dtest['check']=2

combo=pd.concat([df,dtest])



def date_split(date):

    d=date.str.split('-', n=1, expand=True)

    return d[1]



combo['MM_DD']= date_split(combo['Date'])



combo['Province_State']=combo['Province_State'].fillna(0)
le=LabelEncoder()

combo['MM_DD']=le.fit_transform(combo['MM_DD'])

combo=pd.get_dummies(combo)



df1=combo[combo['check']==1]

dtest1=combo[combo['check']==2]
df1.drop('check', axis=1, inplace=True)

dtest1.drop('check', axis=1, inplace=True)
X_train1, X_valid1, y_train1, y_valid1 = train_test_split(df1, y1, train_size=0.8, test_size=0.2, random_state=0)

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(df1, y2, train_size=0.8, test_size=0.2, random_state=0)
ef= ExtraTreesRegressor(n_estimators=15, random_state=3 )

p2=ef.fit(X_train1, y_train1).predict(X_valid1)

rmsle2=np.sqrt(mean_squared_log_error(y_valid1 , p2))

print(rmsle2)
ef2= ExtraTreesRegressor(n_estimators=29,criterion='friedman_mse', random_state=7)

p3=ef2.fit(X_train2, y_train2).predict(X_valid2)

rmsle3=np.sqrt(mean_squared_log_error(y_valid2 , p3))

print(rmsle3)
pre1=ef.fit(df1,y1).predict(dtest1)

pre2=ef2.fit(df1,y2).predict(dtest1)
output=pd.DataFrame({'ForecastId': dtest.index, 'ConfirmedCases':pre1, 'Fatalities':pre2})

output.to_csv('submission.csv', index=False)