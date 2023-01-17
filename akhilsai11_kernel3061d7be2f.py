# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv',parse_dates=['Date']).drop('Province_State',1)
def convDate(x):

    return x.month*30 +x.day
data['Date']=data['Date'].apply(convDate)

data['ConfirmedCases']=data['ConfirmedCases']+1
data['Fatalities']=data['Fatalities']+1
countries=data['Country_Region'].unique().tolist()
split=[]

for i in range(len(data)-1):

    if(data.iloc[i]['Country_Region'] !=data.iloc[i+1]['Country_Region']):

        split.append(i+1)
df=np.split(data,split,axis=0)
exp=[np.poly1d(np.polyfit(y=np.log(q['ConfirmedCases']),x=q['Date'],deg=3)) for q in df]
print(exp)
def predict(xs):

    index=countries.index(xs['Country_Region'])

    coeff=exp[index]

    return max(math.floor(np.exp(coeff(xs['Date']))),1)
pred=[predict(df[5].iloc[q]) for q in range(len(df[5]))]
plt.scatter(y=(pred),x=df[5]['Date'])
plt.scatter(y=(df[0]['Fatalities']),x=df[1]['Date'])
exp2=[np.poly1d(np.polyfit(y=np.log(q['Fatalities']),x=q['Date'],deg=3)) for q in df]
def predict1(xs):

    index=countries.index(xs['Country_Region'])

    coeff=exp2[index]

    return max(math.floor(np.exp(coeff(xs['Date']))),1)
pred1=[predict1(df[0].iloc[q]) for q in range(len(df[1]))]
plt.scatter(y=(pred1),x=df[1]['Date'])
test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv',parse_dates=['Date'])

test['Date']=test['Date'].apply(convDate)

print(test)
test_pred=[predict(test.iloc[q]) for q in range(len(test))]

deaths=[predict1(test.iloc[q]) for q in range(len(test))]

print(test_pred)
my_submission = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases':test_pred,'Fatalities':deaths})
my_submission.to_csv('submission.csv', index=False)