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
import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

%matplotlib inline



df=pd.read_csv('../input/temperature-data-for-country-aland/GlobalTemperaturesdata.csv',parse_dates=["dt"],index_col="dt")

df.head()
df.dropna(inplace=True)



newdf=df.resample('A').mean()



newdf.dropna(inplace=True)
plt.hist(newdf.AverageTemperature, bins=20, rwidth=0.8)

plt.show()
newdf.describe()
newdf['Zscore']=(newdf.AverageTemperature - newdf.AverageTemperature.mean())/newdf.AverageTemperature.std()



newdf.head()
df2=newdf[(newdf.Zscore>-2)&(newdf.Zscore<2)]



plt.hist(df2.AverageTemperature,bins=20,rwidth=0.8)

plt.show()
df2.AverageTemperature.plot(figsize=(15,10))
from statsmodels.tsa.stattools import adfuller





def adfuller_test(AverageTemperature):

    result=adfuller(AverageTemperature)

    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):

        print(label+' : '+str(value) )

    if result[1] <= 0.05:

        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")

    else:

        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
adfuller_test(df2['AverageTemperature'])
df.head()
df2['m_3']=df2['AverageTemperature'].rolling(window=3).mean();

df2['m_7']=df2['AverageTemperature'].rolling(window=7).mean();

df2['m_9']=df2['AverageTemperature'].rolling(window=9).mean();

plt.figure(figsize=(15,10))

plt.subplot(4,1,1)

plt.plot(df2.AverageTemperature)

plt.title('Average Temperature original plot')



plt.subplot(4,1,2)

plt.plot(df2.m_3,'g')

plt.title('Average Temperature with Three rolling windows')



plt.subplot(4,1,3)

plt.plot(df2.m_7,'r')

plt.title('Average Temperature with Seven rolling windows')



plt.subplot(4,1,4)

plt.plot(df2.m_9,'y')

plt.title('Average Temperature with Nine rolling windows')



plt.tight_layout(pad=3.0)
df2['lag_1']=df2['AverageTemperature'].shift(1)

df2['lag_2']=df2['AverageTemperature'].shift(2)
df2['Year']= df2.index.year



df2.head()
df2.drop(['AverageTemperatureUncertainty','Zscore'],axis=1,inplace=True)
df2.dropna(inplace=True)



df2.index = range(len(df2))
df2.head()
from sklearn.linear_model import LinearRegression



model = LinearRegression()

from sklearn.model_selection import train_test_split



X = df2[['m_3','m_9','lag_1','lag_2','Year']]

y=df2.AverageTemperature



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model.fit(X_train,y_train)
y_predict= model.predict(X_test)



y_predict
from statsmodels.tools.eval_measures import rmse

error = rmse(y_test,y_predict)

print('Root Mean Squared Error ',error)
db=pd.DataFrame(data=y_test)



db.index= range(len(db))



db['Predicted_value']=y_predict



db.head()
plt.figure(figsize=(20,8))

plt.subplot(3,1,1)

plt.plot(db.AverageTemperature)

plt.title('Actual Avg Temperature')

plt.legend('Actual')







plt.subplot(3,1,2)

plt.plot(db.Predicted_value,'g')

plt.title('Predicted Avg Temperature')

plt.legend('Predict')



plt.subplot(3,1,3)

plt.plot(db.Predicted_value,'g')

plt.title('Combined plot of Actual and Predicted value')

plt.legend('Predict')



plt.subplot(3,1,3)

plt.plot(db.AverageTemperature)

plt.legend('Actual')



plt.tight_layout(pad=3.0)






