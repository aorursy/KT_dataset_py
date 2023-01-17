import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statsmodels.api as sm

import statsmodels.tsa.api as smt

import statsmodels.formula.api as smf



import matplotlib.pyplot as plt

%matplotlib inline



import warnings

import itertools

from sklearn.metrics import mean_squared_error

import gc
#use smaller-size object types eg. int32 to make our dataframe more memory efficient

types_dict = {'id': 'int32',

             'item_nbr': 'int32',

             'store_nbr': 'int8',

             'unit_sales': 'float32'

             }

use_cols = ["id","date","item_nbr","store_nbr","unit_sales"]
#select dates from 10/7/2017

grocery_train = pd.read_csv('../input/train.csv', low_memory=True,usecols=use_cols, dtype=types_dict,parse_dates=['date'],skiprows=range(1, 121688779))
grocery_train.head()
use_cols2 = ["id","date","item_nbr","store_nbr"]
grocery_test = pd.read_csv('../input/test.csv', low_memory=True,usecols=use_cols2, dtype=types_dict,parse_dates=['date'])
items_per_date_and_sales_grp=grocery_train.groupby(['item_nbr'])
#total num of items in our data

len(items_per_date_and_sales_grp)
#this function exhuastiely search for the optimal parameters(amount of AR, MA) and find the best ones with lowest AIC score.



def gridSearch(itemObj,silent):

    # Define the p, d and q parameters to take any value between 0 and 3

    p = d = q = range(0, 3)



    # Generate all different combinations of p, q and q triplets

    pdq = list(itertools.product(p, d, q))



    # Generate all different combinations of seasonal p, q and q triplets

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



    bestAIC = np.inf

    bestParam = None

    bestSParam = None

    

    print('Running GridSearch')

    

    #use gridsearch to look for optimial arima parameters

    for param in pdq:

        for param_seasonal in seasonal_pdq:

            try:

                mod = sm.tsa.statespace.SARIMAX(itemObj,

                                                order=param,

                                                seasonal_order=param_seasonal,

                                                enforce_stationarity=False,

                                                enforce_invertibility=False)



                results = mod.fit()



                #if current run of AIC is better than the best one so far, overwrite it

                if results.aic<bestAIC:

                    bestAIC = results.aic

                    bestParam = param

                    bestSParam = param_seasonal



            except:

                continue

                

    print('the best ones are:',bestAIC,bestParam,bestSParam)

    

    print('proceeding to build a model with best parameter')

    #apply the best parameters on the arima model

    mod = sm.tsa.statespace.SARIMAX(itemObj,

                                    order=bestParam,

                                    seasonal_order=bestSParam,

                                    enforce_stationarity=False,

                                    enforce_invertibility=False)



    results = mod.fit()



    if(silent==False):

        print(results.summary().tables[1])

    

    print("running diagnoistic plots")

    #results.plot_diagnostics(figsize=(15, 12))

    #plt.show()

    

    #do a small test prediction

    predictDateStr = '2017-08-01'

    predictDate = pd.to_datetime(predictDateStr)

    pred = results.get_prediction(start=predictDate,dynamic=True, full_results=True)

    pred_ci = pred.conf_int()

    

    #calculting error scores

    print("Validation mse:", mean_squared_error(itemObj[predictDateStr:], pred.predicted_mean))

    

    if(silent==False):

        #plot the prediction graph out

        plt.plot(itemObj, color='black')

        plt.plot(pred.predicted_mean,color='red', alpha=.7)



        #ax = plt.gca()

        #ax.fill_between(pred_ci.index,

        #            pred_ci.iloc[:, 0],

        #            pred_ci.iloc[:, 1], color='k', alpha=.15)

        plt.show()

    

    #make forecast for next 16 days for submission data

    n_steps = 16

    pred_uc_99 = results.get_forecast(steps=n_steps, alpha=0.01) # alpha=0.01 signifies 99% confidence interval



    # Get confidence intervals 95% & 99% of the forecasts

    pred_ci_99 = pred_uc_99.conf_int()

    

    if(silent==False):

        #plot forecase prediction

        plt.plot(itemObj, color='black')

        plt.plot(pred_uc_99.predicted_mean,color='red', alpha=.7)

        ax = plt.gca()

        ax.fill_between(pred_ci_99.index,

                        pred_ci_99.iloc[:, 0],

                        pred_ci_99.iloc[:, 1], color='k', alpha=.25)

        plt.show()



    print(pred_uc_99.predicted_mean)



    #return forecasted result

    return pred_uc_99.predicted_mean
#turn off warnings before running gridsearch

import warnings

warnings.filterwarnings(action='once')
listOfItems = []

count = 0



#go through each item group and put it through arima for prediction

for name,grp in items_per_date_and_sales_grp:

    count+=1

    

    print('run count',count)

    #further group it by day, averaging the unit sales

    itembyday = grp.groupby('date')['unit_sales'].mean()

    

    #make sure every item has valid data for our training date range(07-10 to 08-15)

    date_index = pd.date_range('2017-07-10', '2017-08-15')

    itembyday = itembyday.reindex(date_index,method='nearest')



    #run modelling use extracted item day sales data

    #show diagnosistic plots only for the first run

    if count==1:

        predictedVal = gridSearch(itembyday,False)

    else:

        predictedVal = gridSearch(itembyday,True)

    

    #create a dataframe from returned predicted values

    predictedDF= pd.DataFrame(columns=['item_nbr','date','unit_sales'])

    predictedDF['unit_sales'] = predictedDF['unit_sales'].astype('float32')

    predictedDF['item_nbr'] = predictedDF['item_nbr'].astype('int32')



    predictedDF['unit_sales']=predictedVal

    predictedDF['date']=predictedVal.index

    predictedDF['item_nbr']=name

    

    #append the dataframe into a list for later concat

    listOfItems.append(predictedDF)

    gc.collect()

    #for testing n-th loop

    #if count==3:

    #    break
predDF = pd.concat(listOfItems)
predDF
grocery_test = pd.merge(grocery_test,predDF,how='left',on=['item_nbr','date'])
grocery_test['unit_sales']=grocery_test['unit_sales'].clip(lower=0)
grocery_test.shape
grocery_test = grocery_test.fillna(0)
grocery_test[['id','unit_sales']].to_csv('grocery_submit.csv', index=False, float_format='%.3f')