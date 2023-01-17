import os
import numpy as np
import pandas as pd
from pandas.tools.plotting import lag_plot,autocorrelation_plot

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
os.listdir("../input")
temp_dic = pd.read_excel('../input/daily-minimum-temperatures-in-me.xlsx',sheet_name=['Temperature'],parse_dates=['Date'])
df = pd.DataFrame(temp_dic['Temperature'])
df.set_index(['Date'],inplace=True)
df.sample(5)
df.info()
df.plot(figsize=(15,5))
lag_plot(df)
#Clearly show some correlation
df_corr = pd.concat([df.shift(1),df],axis=1)
df_corr.columns=['t-1','t+1']
df_corr.corr(method="pearson")
sns.heatmap(df_corr.corr(method="pearson"),cmap="Blues",annot=True)
#Shows strong positive Correlation
plt.figure(figsize=(10,10))
autocorrelation_plot(df)
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
X = df.values
train,test = X[1:len(X)-7],X[len(X)-7:]
print(train.shape)
print(test.shape)
model = AR(train)
model_fit= model.fit()
window = model_fit.k_ar #Variables
coeff = model_fit.params # Coefficients
#Linear Regression - y= bX1 + bX2 ... + bXn
history = train[len(train) - window:]
len(train) - 29
history
history = [history[i] for i in range(len(history))]
history
predictions=[]

for t in test:
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    y = coeff[0]
    for d in range(window):       
        y += coeff[d + 1] * lag[window - d - 1]
        #print(coeff[d + 1] * lag[window - d - 1])
    predictions.append(y)
    history.append(t)
    print(f"Predicted :{y} and expected value:{t}")
mean_squared_error(test,predictions)
plt.plot(test,label='actual')
plt.plot(predictions,label='predicted')
plt.legend()













