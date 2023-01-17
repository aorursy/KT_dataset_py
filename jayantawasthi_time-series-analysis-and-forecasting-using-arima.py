# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/time-series-datasets/Electric_Production.csv")
train.head()
train.info()
train["DATE"]=pd.to_datetime(train["DATE"],infer_datetime_format=True)
train=train.set_index(['DATE'])
train.head()
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6

train.plot()


from statsmodels.tsa.seasonal import seasonal_decompose

res= seasonal_decompose(train, model='multiplicative')

res.plot()

plt.show()
train["IPG2211A2N"].iloc[:60]
from statsmodels.tsa.stattools import adfuller
def stest(x):

    result = adfuller(x)

    print(result[1])
stest(train.iloc[:,0].values)
len(train)

a=[]

a.append(train["IPG2211A2N"].iloc[0])

for i in range(396):

     z=train["IPG2211A2N"].iloc[i+1]-train["IPG2211A2N"].iloc[i]

     a.append(z)
a[:5]
plt.plot(a)
stest(a)
t=train["IPG2211A2N"].to_dict()

x={}

l=0

for i,j in t.items():

       x.update({i:a[l]})

       l=l+1

   
xe=pd.Series(x)
plt.plot(xe)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot=plot_pacf(xe,lags=10)
acf_plot=plot_acf(xe,lags=10)
len(xe)
xtrain=xe[0:330]

xtest=xe[330:397]
len(xtrain)
xtrain.head()
from statsmodels.tsa.arima_model import ARIMA
xee=ARIMA(xtrain,order=(3,0,3))
ttt=xee.fit()
ttt.aic
tt=ttt.forecast(steps=67)[0]
tt[:10]
xtest.values[:10]
xtest.size
xtest[0]
qqq=[]

a=0

for i in range(330):

    a=a+xtrain[i]

    qqq.append(a)
train["IPG2211A2N"].iloc[335]
qqq[329]
def pv(web):

        qqq=[]

        a=101.4

        for i in range(67):

            a=a+web[i]

            qqq.append(a)

        return qqq


qqqt=pv(xtest)
qqqt[:10]
qqqf=pv(tt)
qqqf[:5]
from sklearn.metrics import mean_squared_error 

import numpy as np
error=mean_squared_error(qqqt,qqqf)
error
np.sqrt(error)