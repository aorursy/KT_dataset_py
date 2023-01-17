# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.metrics import median_absolute_error,explained_variance_score

from sklearn.metrics import mean_absolute_error,mean_squared_log_error

        

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

def smape(y_true, y_pred):

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    return np.nanmean(diff)

    #from https://www.otexts.org/fpp/2/5

    #Hyndman and Koehler (2006) recommend that the sMAPE not be used. 

    #It is included here only because it is widely used, although we will not 

    #use it in this book.





#from http://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

def mse(y_true,y_pred):

    #A non-negative floating point value (the best value is 0.0), 

    #or an array of floating point values, one for each individual target.

    return mean_squared_error(y_true, y_pred)

def r2score(y_true, y_pred):

    r2=r2_score(y_true, y_pred)

    r2=max(-.5,r2)*200 # clip from -100 to +200

    return r2

def mdae(y_true, y_pred):

    #A positive floating point value (the best value is 0.0).

    m=median_absolute_error(y_true, y_pred)

    m=min(200,m)

    return m

def mape(y_true, y_pred):

    #MAE output is non-negative floating point. The best value is 0.0.

    m=np.nanmean(np.abs((y_true-y_pred)/y_true) )

    return m*100

def mae(y_true, y_pred):

    #MAE output is non-negative floating point. The best value is 0.0.

    m=mean_absolute_error(y_true, y_pred)*20 #just to scale better

    m=min(200,m)

    return m

def evs(y_true, y_pred):

    #Best possible score is 1.0, lower values are worse.

    m=explained_variance_score(y_true, y_pred)*200

    m=min(0,m)

    return m

def msle(y_true, y_pred):

    #A non-negative floating point value (the best value is 0.0), 

    # or an array of floating point values, one for each individual target.

    m=mean_squared_log_error(y_true, y_pred)*200 

    m=min(200,m)

    return m
y_true = np.array(3)

y_pred = np.ones(1)

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')



res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res3.append(mse(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res4.append(mae(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res5.append(evs(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res6.append(msle(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res7.append(mdae(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

    res8.append(mape(np.array([y_true]*100).reshape(100,), 

                        np.array([y_pred*i]*100).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()
y_true = np.array([1,9])

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue', label='SMAPE')

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res3.append(mse(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res4.append(mae(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res5.append(evs(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res6.append(msle(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res7.append(mdae(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res8.append(mape(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()

print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])

print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 

      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)

y_true = np.random.uniform(1, 9, 100)

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')

plt.legend()

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res3.append(mse(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res4.append(mae(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res5.append(evs(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res6.append(msle(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res7.append(mdae(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res8.append(mape(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()

print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])

print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 

      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)

y_true = np.random.lognormal(1, 1, 100)

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res3.append(mse(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res4.append(mae(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res5.append(evs(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res6.append(msle(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res7.append(mdae(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

    res8.append(mape(np.array(y_true).reshape(100,),

                         np.array(y_pred*i).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()

print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])

print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 

      ' at median %0.2f' % np.nanmedian(y_true))
y_true = np.array([0])

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res3.append(mse(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res4.append(mae(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res5.append(evs(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res6.append(msle(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res7.append(mdae(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

    res8.append(mape(np.array([y_true]*100).reshape(100,),

                         np.array([y_pred*i]*100).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()
np.random.seed(0)

y_true = np.array([0,9])

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res3.append(mse(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res4.append(mae(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res5.append(evs(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res6.append(msle(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res7.append(mdae(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

    res8.append(mape(np.array([y_true]*50).reshape(100,),

                         np.array([y_pred*i]*50).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()

print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])

print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 

      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)

y_true = np.random.lognormal(1, 1, 100)

y_true[y_true < 3] = 0

print('There are %d zeros in the series' % np.sum(y_true == 0))

y_pred = np.ones(len(y_true))

x = np.linspace(0,10,1000)

res = [smape(y_true, i * y_pred) for i in x]

plt.plot(x, res,color='blue',label='SMAPE')

res2,res3,res4,res5,res6,res7,res8=[],[],[],[],[],[],[]

for i in x:

    res2.append(r2score(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res3.append(mse(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res4.append(mae(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res5.append(evs(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res6.append(msle(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res7.append(mdae(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

    res8.append(mape(np.array([y_true]*1).reshape(100,),

                        np.array([y_pred*i]*1).reshape(100,)))

plt.plot(x, res2,color='red',label='R2')

plt.plot(x, res3,color='green',label='MSE')

plt.plot(x, res4,color='yellow',label='MAE')

plt.plot(x, res5,color='magenta',label='EVS')

plt.plot(x, res6,color='cyan',label='MSLE')

plt.plot(x, res7,color='black',label='MdAE')

plt.plot(x, res8,color='orange',label='MAPE')

plt.legend()

plt.show()

print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])

print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 

      ' at median %0.2f' % np.nanmedian(y_true))