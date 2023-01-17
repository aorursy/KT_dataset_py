import pandas as pd
import matplotlib.pyplot as plt

shampoo=pd.read_csv('C:\\Users\\Anuvrat Shukla\\Desktop\\nw\\shampoo.csv',index_col=[0],parse_dates=[0],squeeze=True)
shampoo.head()

# NOISY curve

shampoo.plot()

# Smoothning

shampoo_smooth=shampoo.rolling(window=10).mean()
shampoo_smooth.plot()

shampoo.describe()

# Base Thrm.

df=pd.concat([shampoo,shampoo.shift(1)],axis=1)
df.columns=['Actual','Predicted']
df.head()

df.dropna(inplace=True)
df.head()

# Mean Square Error

from sklearn.metrics import mean_squared_error as err

error=err(df.Actual,df.Predicted)
error

# Base line error
import numpy as np
np.sqrt(error)

# ARIMA and plot_acf & plot_pacf

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


plot_acf(shampoo)
#q=3,4

plot_pacf(shampoo)
#p=2,3

#p=2,3
#q=3,4
#d=0-2

from statsmodels.tsa.arima_model import ARIMA

# TRAIN TEST SPLIT

train=shampoo[:25]
test=shampoo[25:]
import warnings
warnings.filterwarnings('ignore')

model=ARIMA(train,order=(4,2,1))

model_fit=model.fit()

model_fcst=model_fit.forecast(steps=11)[0]

model_fcst

test.values

model_fit.aic

plt.plot(test)
plt.plot(model_fcst,color='red')

#ARIMA error
np.sqrt(err(test,model_fcst))

# Tuning parameters (2nd way)

p=d=q=range(0,5)
#import iteration tools

import itertools
pdq=list(itertools.product(p,d,q))
pdq

#Ignore Warnings
import warnings
warnings.filterwarnings('ignore')
#Ignore errors
for i in pdq:
    try:
        model=ARIMA(train,order=i)
        model_fit=model.fit()
       
        model_forecast=model_fit.forecast(steps=11)[0]
        print(i,np.sqrt(err(test,model_forecast)),model_fit.aic)
      
    except:
        continue

