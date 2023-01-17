## Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from statsmodels.tsa.ar_model import AR

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX



from scipy import stats

from scipy.stats import normaltest



from sklearn.metrics import mean_squared_error



import warnings

from IPython.display import clear_output
## Load data and convert datatypes

data = pd.read_csv("/kaggle/input/demand-forecasting-kernels-only/train.csv")

data['date'] = pd.to_datetime(data['date'])



## Split into train and testing data

## Use the last 3 month (Oct-Dec 2017) as testing data. The rest is training data.

train = data.loc[data['date'] < '2017-10-01' ]

test = data.loc[data['date'] >= '2017-10-01' ]
## Sales for a store and an item

def get_sales(train, test, store, item):

    # Training

    sales_train = train.loc[train['store'] == int(store)].loc[train['item'] == int(item)]

    dates_train = sales_train['date'].dt.strftime('%Y-%m-%d')

    sales_train = sales_train['sales'].tolist()

   

    # Testing 

    sales_test = test.loc[test['store'] == int(store)].loc[test['item'] == int(item)]

    dates_test = sales_test['date'].dt.strftime('%Y-%m-%d')

    sales_test = sales_test['sales'].tolist()

              

    return sales_train, dates_train, sales_test, dates_test
def plot_sales_per_store_item(df, store=1, item=1):

    '''

    Plots predicted sales for a given store and an item.

    Input: Dataframe df with test and predicted values

    '''

    df = df.loc[df['store'] == store]

    df = df.loc[df['item'] == item]

    

    plt.figure(figsize=[20,10])

    

    plt.plot(df['date'].dt.strftime('%Y-%m-%d'), df['sales'], color="navy", label='truth')

    plt.plot(df['date'].dt.strftime('%Y-%m-%d'), df['predicted ARIMA'], color="green", label='prediction')

    plt.title("Prediction and ground truth")

    plt.xticks(np.arange(0, len(df), 10))

    plt.xlabel("Date")

    plt.ylabel("Sales [-]")

    plt.legend()

    plt.show()
## Remove seasonal component in data

def difference(data, interval=365):

    diff = []

    for i in range(interval, len(data)):

        value = data[i] - data[i - interval]

        diff.append(value)

        

    return np.array(diff)



## Reconstruct seasonality

def inverse_difference(history, yhat, interval=365):

    

    return yhat + history[-interval]
## Get data for store 4 and item 7:

store = 4

item = 7



train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, store, item)



## Remove seasonal component in data

differenced = difference(train_4_7, 365)



## Fit model with arbitray order

model = ARIMA(differenced, order=(7, 0, 1))

model_fit = model.fit(disp=0)



## Predictions

print("=============================\nOne step out-of-sample forecast\n")

### 1. One step out-of-sample forecasts

##### Forecast function

forecast = model_fit.forecast()[0]

forecast = inverse_difference(train_4_7, forecast, 365)



#### Predict function

start = len(differenced)

end = len(differenced)

prediction = model_fit.predict(start, end)

prediction = inverse_difference(train_4_7, prediction, 365)



print("True value: {}\nForecast function: {} | Predict function: {}".format(test_4_7[0], forecast, prediction))

print("=============================\nMultiple step out-of-sample forecast\n")



### 2. Multiple step out-of-sample forecast

#### Forecast function

days = 7



forecast = model_fit.forecast(steps=days)[0]



history = [x for x in train_4_7]

day = 1

for yhat in forecast:

    inverted = inverse_difference(history, yhat, 365)

    history.append(inverted)

    day +=1



#### Predict function

start = len(differenced)

end = start + days-1



prediction = model_fit.predict(start, end)



history_pred = [x for x in train_4_7]

day = 1

for yhat in prediction:

    inverted = inverse_difference(history_pred, yhat, 365)

    history_pred.append(inverted)

    day +=1



## Plots

plt.figure(figsize=[20,10])

plt.title("Forecasts")

plt.plot(test_4_7[:days], color='grey', label='Test')

plt.plot(history[-days:], color='green', label='Forecast')

plt.plot(history_pred[-days:], color='blue', label='Predict')

plt.xlabel("Day")

plt.ylabel("Sales")

plt.legend()

plt.show()
model_fit.summary()
def evaluate_arima_model(train, test, arima_order):

    '''

    Evaluate an ARIMA model of order (p, d, q).

    Returns the MSE.

    '''

    ## Remove seasonal components (year scale)    

    history = [x for x in train]   

    predictions = []

   

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)          

        yhat = model_fit.forecast()[0]         

        predictions.append(yhat)

        history.append(test[t])

        

    error = mean_squared_error(test, predictions)

    

    return error





def evaluate_models(train, test, p_values, d_values, q_values):

    '''

    Grid search on ARIMA models.

    Prints out the best parameter setting.

    '''

    train = np.array(train).astype('float32')

    test = np.array(test).astype('float32')

    

    best_score, best_config = float('inf'), None

    

    for p in p_values:      

        for d in d_values:            

            for q in q_values:

                order = (p,d,q)           

                

                try:

                    mse = evaluate_arima_model(train, test, order)

                    

                    if mse < best_score:

                        best_score, best_config = mse, order

                        

                    print('ARIMA order: {} | MSE: {:.4f}'.format(order, mse))                 

                

                except:

                    continue

    

    print("\n-----------------------------\nBest ARIMA order: {} | Best MSE: {}".format(best_config, mse))
## Grid search for a store and an item

store = 4

item = 7



## Grid parameters

p_values = [6, 7, 8]

d_values = [1]

q_values = [0, 2, 3, 4]



## Get data for a store and an item

train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, store, item)



warnings.filterwarnings("ignore")

#evaluate_models(train_4_7, test_4_7, p_values, d_values, q_values)
## Forecast with best model

train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, store, item)



def predict_arima(train_4_7, dates_train, test_4_7, dates_test):

    days = 90

    store = 1

    item = 15

    arima_order = (8,1,3)





    ## Get data for a store and an item:

    

    differenced = difference(train_4_7, 365)



    ## Fit model

    model = ARIMA(differenced, order=arima_order)

    model_fit = model.fit(disp=0)



    ## Predict

    forecast = model_fit.forecast(steps=days)[0]



    history = [x for x in train_4_7]

    day = 1

    for yhat in forecast:

        inverted = inverse_difference(history, yhat, 365)

        history.append(inverted)

        day +=1

    

    return history

 

arima_res = predict_arima(train_4_7, dates_train, test_4_7, dates_test)





## Plot

plt.figure(figsize=[20,10])

plt.title("Forecasts")

plt.plot(test_4_7[:days], color='grey', label='Test')

plt.plot(arima_res[-days:], color='green', label='Forecast')

plt.xlabel("Day")

plt.ylabel("Sales")

plt.legend()

plt.show()
fig = plt.figure(figsize=(12,8))

ax0 = fig.add_subplot(111)



sns.distplot(model_fit.resid ,fit = stats.norm, ax = ax0)

plt.show()
### All sales



def predict_all_arima(data):

    '''

    Predictions for all combinations of stores and items.

    Reads the dataframe for tests.

    Outputs the dataframe in plus the predicted sales.

    '''

  

    ## Initialize dataframe for output

    cols = ["date", "store", "item", "sales"]

    df_out = pd.DataFrame()

    

    ## Steps slice data

    stores = list(np.arange(1, 11))

    items = list(np.arange(1, 51))



    for store in stores:

        data_store = data.loc[data['store'] == store]

        

        for item in items:

            predictions = []

            clear_output(wait=True)

            print("Processing store {}/{} and item {}/{}".format(store, stores[-1], item, items[-1]))

            

            data_store_item = data_store.loc[data_store['item'] == item]

         

            train = data_store_item.loc[data_store_item['date'] < '2017-10-01' ]

            test = data_store_item.loc[data_store_item['date'] >= '2017-10-01' ]

            

            train_store_item, dates_train, test_store_item, dates_test = get_sales(train, test, store, item)

            

            

            ## Use model for prediction        

            # ARIMA parameters

            days = len(test_store_item)

            arima_order = (8,1,3)

            

            # Remove seasonal component

            differenced = difference(train_store_item, 365)



            # Fit model

            model = ARIMA(differenced, order=arima_order)

            model_fit = model.fit(disp=0)



            # Predict

            forecast = model_fit.forecast(steps=days)[0]



            history = [x for x in train_store_item]

            day = 1

            for yhat in forecast:

                inverted = inverse_difference(history, yhat, 365)

                #history.append(inverted)

                predictions.append(inverted)

                day +=1

                  

            ## Add prediction to output dataframe

            

            test["predicted ARIMA"] = predictions           

            df_out = df_out.append(test, ignore_index=True)

    

    ## Generate and save output 

    df_out.to_csv("Predictions_ARIMA.csv")

    print("Done!")



    return df_out



## Predict

data = pd.read_csv("./data/train.csv")

data['date'] = pd.to_datetime(data['date'])

#df_arima = predict_all_arima(data)
plot_sales_per_store_item(df_arima, 8, 4)
## Get data for store 4 and item 7:

train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, 4, 7)



def predict_sarima(train_4_7, dates_train, test_4_7, dates_test):

    days = 92

    

    ## Multiple step out-of-sample forecast with predict function



    test_4_7 = test_4_7[:days]



    start = len(train_4_7)

    end = start + days-1



    history = [x for x in train_4_7]

    predictions = []



    for i in range(len(test_4_7)):   

        # Fit model and make forecast for history

        model = SARIMAX(history, order=(6, 1, 0), seasonal_order=(0, 0, 0, 7), trend='n')

        model_fit = model.fit(disp=False)

        yhat = model_fit.predict(len(history), len(history))



        # Store forecast in list of predictions

        predictions.append(yhat)



        # Add actual observation to history for the next loop

        history.append(test_4_7[i])

        

    return predictions



sarima_res = predict_sarima(train_4_7, dates_train, test_4_7, dates_test)



## Plots

plt.figure(figsize=[20,10])

plt.title("Forecasts")

plt.plot(test_4_7[:days], color='grey', label='Test')

plt.plot(sarima_res, color='blue', label='Predict')

plt.xlabel("Day")

plt.ylabel("Sales")

plt.legend()

plt.show()
def evaluate_sarima_model(train, test, arima_order, seasonal_order, trend):

    '''

    Evaluate an SARIMA model of order (p, d, q) (P, D, Q, m) with trends.

    Returns the MSE.

    '''

    ## Remove seasonal components (year scale)    

    history = [x for x in train]   

    predictions = []

   

    for t in range(len(test)):

        model = SARIMAX(history, order=arima_order, seasonal_order=seasonal_order, trend=trend)

        model_fit = model.fit(disp=0)               

        yhat = model_fit.predict(len(history), len(history))        

        predictions.append(yhat)

        history.append(test[t])

        

    error = mean_squared_error(test, predictions)

    

    return error





def evaluate_all_sarima_models(train, test, p_values, d_values, q_values, P_values, D_values, Q_values, m_values):

    '''

    Grid search on SARIMA models.

    Inputs:

    train, test: lists of sales for a given store and an item

    grid parameters as lists

    Prints out the best parameter setting.

    '''

    

    train = np.array(train).astype('float32')

    test = np.array(test).astype('float32')

    

    best_score, best_config, best_seasonal, best_trend = float('inf'), None, None, None

    

    for p in p_values:      

        for d in d_values:            

            for q in q_values:

                for P in P_values:

                    for D in D_values:

                        for Q in Q_values:

                            for m in m_values:

                                    arima_order = (p,d,q)

                                    seasonal_order = (P,D,Q,m)

                                    trend = "c"

                                    try: 

                                        print("Fitting model ARIMA order {} seasonal order {} trend {}".format(arima_order, seasonal_order, trend))

                                        mse = evaluate_sarima_model(train, test, arima_order, seasonal_order, trend)

                                        print(mse)

                                        if mse < best_score:

                                            best_score, best_config, best_seasonal, best_trend = mse, arima_order, seasonal_order, trend



                                        print('SARIMA order: {} | Seasonal order: {} | Trend: {} | MSE: {:.4f}'.format(order, seasonal_order, trend, mse))                               

                                    except:

                                        continue

    

    print("\n-----------------------------\nBest SARIMA order: {} | Best seasonal order: {} | Best Trend: {} | Best MSE: {}".format(best_score, best_config, best_seasonal, best_trend))
#### Grid search

p_values = [1, 8, 9]

d_values = [1]

q_values = [3, 7]



P_values = [1]

D_values = [1]

Q_values = [0]

m_values = [7]

#trends = ['n', 'c']



store = 4

item = 7





## Get data for a store and an item

train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, store, item)





#evaluate_sarima_model(train_4_7, test_4_7, (1,0,1), (1,1,2,7), 'c')



warnings.filterwarnings("ignore")

#evaluate_all_sarima_models(train_4_7, test_4_7, p_values, d_values, q_values, P_values, D_values, Q_values, m_values)
#### Prediction with best model



def predict_all_sarima(data):

    '''

    Predictions for all combinations of stores and items.

    Reads the dataframe for tests.

    Outputs the dataframe in plus the predicted sales.

    '''

  

    ## Initialize dataframe for output

    cols = ["date", "store", "item", "sales"]

    df_out = pd.DataFrame()

    

    ## Steps slice data

    stores = list(np.arange(1, 11))

    items = list(np.arange(1, 51))



    for store in stores:

        data_store = data.loc[data['store'] == store]

        

        for item in items:

            predictions = []

            clear_output(wait=True)

            print("Processing store {}/{} and item {}/{}".format(store, stores[-1], item, items[-1]))

            

            data_store_item = data_store.loc[data_store['item'] == item]

         

            train = data_store_item.loc[data_store_item['date'] < '2017-10-01' ]

            test = data_store_item.loc[data_store_item['date'] >= '2017-10-01' ]

            

            train_store_item, dates_train, test_store_item, dates_test = get_sales(train, test, store, item)

            

            

            ## Use model for prediction

            

            # SARIMA parameters

            days = len(test_store_item)

            arima_order = (6,1,0)

            seasonal_order = (0,0,0,7)

        



            # Fit model

            start = 1

            end = 1

            

            predict = model_fit.predict(start, end)



            history = [x for x in train_store_item]

            day = 1

            for yhat in forecast:

                inverted = inverse_difference(history, yhat, 365)

                #history.append(inverted)

                predictions.append(inverted)

                day +=1

                  

            ## Add prediction to output dataframe

            

            test["predicted SARIMA"] = predictions           

            df_out = df_out.append(test, ignore_index=True)

    

    ## Generate and save output 

    df_out.to_csv("Predictions_SARIMA.csv")

    print("Done!")



    return df_out



## Predict

data = pd.read_csv("./data/train.csv")

data['date'] = pd.to_datetime(data['date'])

#df_sarima = predict_all_sarima(data)
#### Plot

plot_sales_per_store_item(df_sarima, 8, 4)
## Compare ARIMA and SARIMA

train_4_7, dates_train, test_4_7, dates_test = get_sales(train, test, store, item)



arima_res = predict_arima(train_4_7, dates_train, test_4_7, dates_test)

sarima_res = predict_sarima(train_4_7, dates_train, test_4_7, dates_test)



def plot_results(test, arima_res, sarima_res, dates_test):

       

    plt.figure(figsize=[20,10])

    plt.title("Both models for store 4 and item 7")

    plt.plot(dates_test, test[:92], color="black", label="Truth")

    plt.plot(dates_test, arima_res[-92:], color="green", label="ARIMA")

    plt.plot(dates_test, sarima_res, color="orange", label="SARIMA")

    plt.xticks(np.arange(0, len(test), 14))

    plt.legend()

    

plot_results(test_4_7, arima_res, sarima_res, dates_test)