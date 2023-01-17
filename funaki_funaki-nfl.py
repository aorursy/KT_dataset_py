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
import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv',sep=',')

df.columns
df.isnull().sum()
df.dtypes
df=pd.get_dummies(df,columns=['Team','PlayDirection'])
df['StadiumType'].unique()
df=df[(df['StadiumType']!='Cloudy') & (df['StadiumType']!='Bowl')]
df['StadiumType'].unique()
outdoor=['Outdoor', 'Outdoors','Open','Indoor, Open Roof','Outdoor Retr Roof-Open', 'Oudoor', 'Ourdoor','Retr. Roof-Open','Outdor',

       'Retr. Roof - Open', 'Domed, Open', 'Domed, open', 'Outside','Heinz Field']

indoor=['Indoors', 'Retractable Roof', 'Indoor','Retr. Roof-Closed','Dome', 'Domed, closed','Indoor, Roof Closed', 'Retr. Roof Closed','Closed Dome','Dome, closed','Domed']
df['stadiumtype']=(df['StadiumType'].isin(outdoor)*1)
df['stadiumtype'].unique()
df['GameWeather'].unique()
rain=['Light Rain', 'Showers','Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.','Rain', 'Heavy lake effect snow','Snow', 'Cloudy, Rain','Rain shower','Rainy']
df['weather']=(~df['GameWeather'].isin(rain)*1)
df=pd.get_dummies(df, columns=['OffenseFormation','Position'])
df.columns
df['PlayerHeight']= df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
df['gameclock']=[ pd.Timedelta(val).total_seconds() for val in df['GameClock']]
df=df.groupby('PlayId').mean()
df=df.dropna()
df.corr()
df['DefendersInTheBox'].head()
features=pd.DataFrame(df,columns=['X', 'Y', 'S', 'A', 'Dis','Dir','NflId','YardLine', 'Quarter',

       'gameclock', 'Down', 'Distance','HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'DefendersInTheBox','PlayerHeight',

       'PlayerWeight','Temperature', 'Humidity', 'Team_home', 'stadiumtype', 'weather','PlayDirection_right',

       'OffenseFormation_ACE',

       'OffenseFormation_I_FORM', 'OffenseFormation_JUMBO',

       'OffenseFormation_PISTOL', 'OffenseFormation_SHOTGUN',

       'OffenseFormation_SINGLEBACK', 'OffenseFormation_WILDCAT', 'Position_C',

       'Position_CB', 'Position_DB', 'Position_DE', 'Position_DL',

       'Position_DT', 'Position_FB', 'Position_FS', 'Position_G',

       'Position_HB', 'Position_ILB', 'Position_LB', 'Position_MLB',

       'Position_NT', 'Position_OG', 'Position_OLB', 'Position_OT',

       'Position_QB', 'Position_RB', 'Position_S', 'Position_SAF',

       'Position_SS', 'Position_T', 'Position_TE', 'Position_WR'])
target=pd.Series(df['Yards'])
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X=scaler.fit_transform(features)

print(X.mean(axis=0))

print(X.std(axis=0))
from sklearn.model_selection import train_test_split

train_X,test_X,train_y,test_y,train_playid,test_playid=train_test_split(X,target,df.index,test_size=0.2,shuffle=False)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(train_X,train_y)

r=lr.score(test_X,test_y)

r
yard=['Yards' + str(i) for i in range(-99,100)]
pred_y=lr.predict(test_X)

pred_y=np.round(pred_y)

pred_y
score=np.array([(i >= pred_y)*1 for i in range(-99,100)])

score
prediction = pd.DataFrame(score.T,

                  columns=yard,

                  index=test_playid)
prediction
score_test=np.array([(i >= test_y)*1 for i in range(-99,100)])
test_y=pd.DataFrame(score_test.T,

                  columns=yard,

                  index=test_playid)

test_y
C=((prediction-test_y)**2).sum().sum()/(199*len(prediction.index))

C
prediction.to_csv('submission.csv')