import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
%pylab inline
from pylab import rcParams
rcParams['figure.figsize'] = 22,10

import datetime

#SERIE AND ARIMA MODEL LIBRARIES
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import warnings
import itertools
warnings.filterwarnings("ignore") # specify to ignore warning messages

print(os.listdir("../input"))
dataset = pd.read_csv("../input/pollution.csv") #import dataset pollution
print("Dataset has "+str(dataset.shape[0])+" rows and "+str(dataset.shape[1])+" attributes, with "+str(dataset['pm2.5'].isna().sum())+" NaN in pm2.5 column") #print number rows and attribute
dataset.head(3) #print only first 3 DF rows
dataset.hist(figsize=(22,10)) #create all histograms for each attributes
plt.show()
#deleted January month bacause pm2.5 is everytime equals to zero
dataset = dataset.drop(dataset[(dataset["year"]==2010) & (dataset["month"]==1) & (dataset["day"]==1)].index) #take index rows
dataset = dataset.drop(axis=1, labels="No")
dataset.head(2)
#select only values : date, pm2.5 and set datetime how 
dataset = dataset[["year","month","day","hour","pm2.5"]] #take in pollution dataset only 4 columns
date_hour = pd.to_datetime(dataset[["year","month","day","hour"]]) #transform it in datetime
pollution = dataset.set_index(date_hour)#set index equal to datetime
pollution = pollution[["pm2.5"]] #transform dataset in only one column

#transform zero value in NaN and after delete it
pollution = pollution.replace(0, pd.np.nan)
pollution = pollution.dropna()

print(pollution.head(3)) #print first 3 rows
#plot pm2.5 time series
series = Series(pollution["pm2.5"].values,index=pollution.index)
series.plot(figsize=(22,10))
plt.title('PM2.5 FROM 2010 TO 2015')
plt.show()
print(series.describe())
#a function to evaluate the ARIMA model
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.70)
    train, test = X[0:train_size], X[train_size:]#split serie in 70% train set e 30% test
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order) #call ARIMA 
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    result ={}
    result['model_fit']=model_fit
    result['mse']=mse
    result['prediction']=predictions
    result['test']=test
    result['train']=train
    result['acc']= accuracy_score(round(pd.Series((v[0] for v in test))),round(pd.Series((v[0] for v in predictions))))
    return result
    
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    acc = mse['acc']
                    mse = mse['mse']
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA'+str(order)+' MSE= '+str(mse)+' and ACCURACY= '+str(acc))
                except:
                    continue
    print('Best ARIMA %s MSE= %.3f' % (best_cfg, best_score))
# evaluate parameters
p = [0, 1, 2, 4, 6, 8, 10]
d = range(0, 3)
q = range(0, 3)

#I have eliminated 0 value because there was an error "SVG  error"
df = pollution
df = df[:1000] #with this selection, I select only 1.000 records

autocorrelation_plot(df.values)

#evaluate_models(df.values, p, d, q) #il modello migliore ha i parametri ARIMA(1, 0, 0) MSE=3180.066
arima_model_1 = evaluate_arima_model(df.values,(1,0,0))
model_fit = arima_model_1['model_fit']
# save model
#model_fit.save('model.pkl')

print("Mean Square Eerror: "+str(arima_model_1['mse']))
print("Accuracy: "+str(arima_model_1['acc']))#l'accuratezza non potrà mai essere alta in variabili continue
prediction = pd.Series( (v[0] for v in arima_model_1['prediction']))
test = pd.Series((v[0] for v in arima_model_1['test']))
train = pd.Series((v[0] for v in arima_model_1['train']))

fig = plt.figure(figsize=(22,10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(test,label="Test set")
ax.plot(prediction,label="Prediction set")
ax.legend()
plt.show()

#fig.savefig('prediction_test.png')   # save the figure to file
#plt.close(fig)
# evaluate parameters
p = [0, 1, 2, 4, 6, 8, 10]
d = range(0, 3)
q = range(0, 3)

#I have eliminated 0 value because there was an error "SVG  error"
df = pollution
df = df[:1000]
scaler = MinMaxScaler()
scaler.fit(df)
df = scaler.transform(df)

#evaluate_models(df, p, d, q) #ARIMA(1, 0, 0)
arima_model_2 = evaluate_arima_model(df,(1,0,0))
model_fit = arima_model_2['model_fit']
# save model
#model_fit.save('model.pkl')

print("Mean Square Error: "+str(arima_model_2['mse']))
print("Accuracy: "+str(arima_model_2['acc']))#l'accuratezza non potrà mai essere alta in variabili continue
prediction = pd.Series( (v[0] for v in arima_model_2['prediction']))
test = pd.Series((v[0] for v in arima_model_2['test']))
train = pd.Series((v[0] for v in arima_model_2['train']))

fig = plt.figure(figsize=(22,10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(test,label="Test set")
ax.plot(prediction,label="Prediction set")
ax.legend()
plt.show()

#fig.savefig('model2.png')   # save the figure to file
#plt.close(fig)
