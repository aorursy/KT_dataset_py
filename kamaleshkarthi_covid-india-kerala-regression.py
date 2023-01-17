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
covid=pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
covid.head()
covid.info()
covid.isnull().sum()
covid.head()
df = covid.loc[(covid['State/UnionTerritory'] == 'Kerala')]
df.head(10)
import seaborn as sns
sns.countplot(x='ConfirmedIndianNational',data=df)
import plotly.offline as py
import plotly.graph_objs as go
Cured_chart = go.Scatter(x=df['Date'], y=df['Cured'], name= 'Cured Rate')
Deaths_chart = go.Scatter(x=df['Date'], y=df['Deaths'], name= 'Deaths Rate')
py.iplot([Cured_chart,Deaths_chart])
df1=df[['Confirmed']]
df1 = df1.values
train_size = int(len(df1) * 0.80)
test_size = len(df1) - train_size
train, test = df1[0:train_size,:], df1[train_size:len(df1),:]
print(len(train), len(test))
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
look_back = 2
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)
#trainX
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(trainX,trainY)
predict1=model.predict(testX)
df = pd.DataFrame({'Actual': testY.flatten(), 'Predicted': predict1.flatten()})
df
df.plot(kind='bar',figsize=(16,10))