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
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

testdf = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

df.head()
df.info()
df = df.drop(columns= ['County','Province_State'])

testdf = testdf.drop(columns= ['County','Province_State'])

df
df.iloc[:,[1,5]]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le1= LabelEncoder()

df.iloc[:,1] = le.fit_transform(df.iloc[:,1])

df.iloc[:,5] = le1.fit_transform(df.iloc[:,5])

testdf.iloc[:,1] = le.transform(testdf.iloc[:,1])

testdf.iloc[:,5] = le1.transform(testdf.iloc[:,5])
df['Date'] = pd.to_datetime(df['Date'])

df['Dayofweek'] = df['Date'].dt.dayofweek

df['Day'] = df['Date'].dt.day

df['Month'] = df['Date'].dt.month

df1 = df.drop(columns=['Date'])



testdf['Date'] = pd.to_datetime(testdf['Date'])

testdf['Day'] = testdf['Date'].dt.day

testdf['Dayofweek'] = testdf['Date'].dt.dayofweek

testdf['Month'] = testdf['Date'].dt.month

testdf1 = testdf.drop(columns=['Date'])
fid = testdf1['ForecastId']

test = testdf1.drop(columns=['ForecastId'])

test
y_train = df1['TargetValue']

x_train = df1.drop(columns=['TargetValue','Id'])

x_train
test1 = test[['Country_Region', 'Population', 'Weight', 'Target','Dayofweek','Day','Month']]

test1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test1 = sc.transform(test1)
from sklearn.model_selection import train_test_split 



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(x_train , y_train)
prediction = forest.predict(x_test)
check = pd.DataFrame({'pred':prediction})

check
accuracy =forest.score(x_test,y_test)

accuracy
predict =forest.predict(x_test1)
predict[1789]
sub = pd.DataFrame({'id':fid,'pred':predict})

sub
a=sub.groupby(['id'])['pred'].quantile(q=0.05).reset_index()

b=sub.groupby(['id'])['pred'].quantile(q=0.5).reset_index()

c=sub.groupby(['id'])['pred'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()