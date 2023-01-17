# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install nsepy
from nsepy import get_history
from datetime import date

Infy = get_history(symbol='INFY',

                   start=date(2015,1,1),

                   end=date(2015,12,31))



Tcs= get_history(symbol='TCS',

                   start=date(2015,1,1),

                   end=date(2015,12,31))

nifty = get_history(symbol='NIFTY',

                   start=date(2015,1,1),

                   end=date(2015,12,31))
#unable to retrieve anything when not using index=True

nifty=get_history(symbol='NIFTY',start=date(2015,1,1),end=date(2015,12,31),index=True)
def MA(p,n):

    return p.rolling(window=n*7).mean().dropna()

for x in [4,16,32,52]:

    print('for MA with'+str(x)+" for Infy =")

    print(MA(Infy['Close'],x))

    print('for MA with'+str(x)+" for TCS =")

    print(MA(Infy['Close'],x))

    print('for MA with'+str(x)+" for Nifty =")

    print(MA(Infy['Close'],x))
#Rolling window 10

Infy_RA_10=Infy['Close'].rolling(window=10).mean().shift(-10)

Tcs_RA_10=Tcs['Close'].rolling(window=10).mean().shift(-10)

nifty_RA_10=nifty['Close'].rolling(window=10).mean().shift(-10)



Infy_RA_75=Infy['Close'].rolling(window=75).mean().shift(-75)

Tcs_RA_75=Tcs['Close'].rolling(window=75).mean().shift(-75)

nifty_RA_75=nifty['Close'].rolling(window=75).mean().shift(-75)
# Volume Shockers

def volume_Shockers(x):

    shock=[]

    for y in range(0,len(x)):

        if y==0:

            shock.extend([float('nan')])

        elif x[y]>=1.1*x[y-1]:

            shock.extend([1])

        elif x[y]<=1.1*x[y-1]:

            shock.extend([0])

  

    return shock

Infy['Volume_Shocks']=volume_Shockers(Infy['Volume'])

Tcs['Volume_Shocks']=volume_Shockers(Tcs['Volume'])

#Price shockers, threshhold=2% of present close price

def price_Shockers(x):

    shock=[]

    for y in range(0,len(x)-1):

        if abs(x[y]-x[y+1])>=0.02*x[y]:

            shock.extend([1])

        elif abs(x[y]-x[y+1])<=0.02*x[y]:

            shock.extend([0])

    shock.extend([float('nan')])

    print(len(shock))

    print(len(x))

    return shock

Infy['Price_Shockers']=price_Shockers(Infy['Close'])

Tcs['Price_Shockers']=price_Shockers(Tcs['Close'])
#Price without Volume

def P_without_V(x,y):

    shock=[]

    for x1 in range(0,len(x)):

        if x[x1]==1 and y[x1]==0:

            shock.extend([1])

        else:

            shock.extend([0])

    return shock

Infy['Price_without_Volume']=P_without_V(Infy['Price_Shockers'],Infy['Volume_Shocks'])

Tcs['Price_without_Volume']=P_without_V(Tcs['Price_Shockers'],Tcs['Volume_Shocks'])
from sklearn.linear_model import Ridge,Lasso

from sklearn.ensemble import GradientBoostingRegressor as gbr

from sklearn.model_selection import GridSearchCV as gsc,train_test_split as tts
from bokeh.plotting import figure

from bokeh.io import output_notebook, show

p=figure(x_axis_label='SerialNumber',y_axis_label='Close')

p.line(range(0,len(Infy)),Infy['Close'])

output_notebook()

show(p)
p=figure(x_axis_label='SerialNumber',y_axis_label='Close')

p.line(range(0,len(Tcs)),Tcs['Close'])

output_notebook()

show(p)
p=figure(x_axis_label='Volume',y_axis_label='Date')

p.line(Infy.index,Infy['Volume'])

p.line(Tcs.index,Tcs['Volume'],line_color='red')

output_notebook()

show(p)
from bokeh.layouts import row

p=figure(x_axis_label='Price_shock_without_volume_shock',y_axis_label='Index')

p1=figure(x_axis_label='Price_shock_without_volume_shock',y_axis_label='Index')

p.circle(Tcs.index,Tcs['Price_without_Volume'],line_color='red')

p1.circle(Infy.index,Infy['Price_without_Volume'])

layout=row(p,p1)

output_notebook()

show(layout)
from statsmodels.tsa.stattools import pacf

lag_pacf = pacf(Infy['Close'], nlags=20, method='ols')

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(Infy['Close'])),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(Infy['Close'])),linestyle='--',color='gray')

plt.title('Partial_Autocorrelation Function')
#Supposing Target variable as Close value

Tcs_Target=Tcs['Close']

Infy_Target=Infy['Close']

Tcs=Tcs.fillna(0)

Infy=Infy.fillna(0)

Tcs_train=Tcs.drop(['Close','Series','Symbol'],axis=1)

Infy_train=Infy.drop(['Close','Series','Symbol'],axis=1)
#train test split

Tcs_x,Tcs_y,Tcs_target_x,Tcs_target_y=tts(Tcs_train,Tcs_Target,test_size=0.2)

Infy_x,Infy_y,Infy_target_x,Infy_target_y=tts(Infy_train,Infy_Target,test_size=0.2)
model1=Ridge()

model1.fit(Tcs_x,Tcs_target_x)

model2=Ridge()

model2.fit(Infy_x,Infy_target_x)
predicted_Tcs=model1.predict(Tcs_y)

predicted_Infy=model2.predict(Infy_y)

from sklearn.metrics import mean_absolute_error as mse

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

#Now, by giving data for tomorrow, predictions can be done
model3=gbr()

model3.fit(Tcs_x,Tcs_target_x)

model4=gbr()

model4.fit(Infy_x,Infy_target_x)

predicted_Tcs=model3.predict(Tcs_y)

predicted_Infy=model4.predict(Infy_y)

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

#Tuning Parameters

gbr_params={'learning_rate':[0.1],'n_estimators':[50,60,70]}

ridge_params={'alpha':[0.9,1,1.05,1.1]}

model1=gsc(Ridge(),ridge_params)

model2=gsc(gbr(),gbr_params)
model1.fit(Tcs_x,Tcs_target_x)

predicted_Tcs=model1.predict(Tcs_y)

print(model1.best_params_)

model1.fit(Infy_x,Infy_target_x)

predicted_Infy=model1.predict(Infy_y)

print(model1.best_params_)
print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

model2.fit(Tcs_x,Tcs_target_x)

predicted_Tcs=model2.predict(Tcs_y)

print(model2.best_params_)

model2.fit(Infy_x,Infy_target_x)

predicted_Infy=model2.predict(Infy_y)

print(model2.best_params_)

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

#marginal improvement using tuned Gradient Boosting Machine
#Predition for Volume

Tcs_Target=Tcs['Volume']

Infy_Target=Infy['Volume']

Tcs=Tcs.fillna(0)

Infy=Infy.fillna(0)

Tcs_train=Tcs.drop(['Volume','Series','Symbol'],axis=1)

Infy_train=Infy.drop(['Volume','Series','Symbol'],axis=1)

#train test split

Tcs_x,Tcs_y,Tcs_target_x,Tcs_target_y=tts(Tcs_train,Tcs_Target,test_size=0.2)

Infy_x,Infy_y,Infy_target_x,Infy_target_y=tts(Infy_train,Infy_Target,test_size=0.2)

model1=Ridge()

model1.fit(Tcs_x,Tcs_target_x)

model2=Ridge()

model2.fit(Infy_x,Infy_target_x)

predicted_Tcs=model1.predict(Tcs_y)

predicted_Infy=model2.predict(Infy_y)

from sklearn.metrics import mean_absolute_error as mse

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

model3=gbr()

model3.fit(Tcs_x,Tcs_target_x)

model4=gbr()

model4.fit(Infy_x,Infy_target_x)

predicted_Tcs=model3.predict(Tcs_y)

predicted_Infy=model4.predict(Infy_y)

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

model2.fit(Tcs_x,Tcs_target_x)

predicted_Tcs=model2.predict(Tcs_y)

model2.fit(Infy_x,Infy_target_x)

predicted_Infy=model2.predict(Infy_y)

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))

#marginal improvement using tuned Gradient Boosting Machine

#Tuning Parameters

gbr_params={'learning_rate':[0.1],'n_estimators':[50,60,70]}

ridge_params={'alpha':[0.9,1,1.05,1.1]}

model1=gsc(Ridge(),ridge_params)

model2=gsc(gbr(),gbr_params)

model1.fit(Tcs_x,Tcs_target_x)

predicted_Tcs=model1.predict(Tcs_y)

print(model1.best_params_)

model1.fit(Infy_x,Infy_target_x)

predicted_Infy=model1.predict(Infy_y)

print(model1.best_params_)

print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))



print('for TCS='+str(mse(Tcs_target_y,predicted_Tcs)))

print('for Infy='+str(mse(Infy_target_y,predicted_Infy)))
#As no column is perfectly correlated, hence no multicollinearity 

Tcs.corr()
Infy.corr()