import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import plotly.graph_objects as go 

from plotly.subplots import make_subplots

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller
# Reading data

DATA_DIR= "../input/m5-forecasting-accuracy/"

CALENDAR= DATA_DIR + "calendar.csv"

# SALES_TRAIN_VALID= DATA_DIR + "sales_train_validation.csv"

SAMPLE_SUB= DATA_DIR + "sample_submission.csv"

SELL_PRICES= DATA_DIR + "sell_prices.csv"

FULL_TRAIN_DF= DATA_DIR + "sales_train_evaluation.csv"



calendar= pd.read_csv(CALENDAR)

# stv= pd.read_csv(SALES_TRAIN_VALID)

sub= pd.read_csv(SAMPLE_SUB)

sell_prices= pd.read_csv(SELL_PRICES)

full_df= pd.read_csv(FULL_TRAIN_DF)



print(f"Calendar Dataframe shape: {calendar.shape}")

# print(f"Sales Train Validation Dataframe shape: {stv.shape}")

print(f"Submission Dataframe shape: {sub.shape}")

print(f"Sell Prices Dataframe shape: {sell_prices.shape}")

print(f"Full training Dataframe shape: {full_df.shape}")
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/134072

def rmsse(y_true, y_pred, y_hist): 

    h, n= len(y_true), len(y_hist)

    error= np.sum((y_true - y_pred)**2)

    deviation= (1/(n-1)) * np.sum((y_hist[1:] - y_hist[:-1])**2)

    rmsse = np.sqrt((1/h) * (error/deviation))

    return rmsse
# store of the sales data columns

d_cols = full_df.columns[full_df.columns.str.contains("d_")]



# group columns by store_id

df= full_df.groupby(full_df["store_id"]).sum()[d_cols].T

df.head()
df.shape
# adding calendar.csv

df= df.reset_index().rename(columns= {"index": "d"}).merge(calendar, how= "left", validate="1:1")
df.head()
# store the store columns

stores= []

for word in df.columns:

    if word.isupper():

        stores.append(word)

stores
# plotting sales over time figure

fig = go.Figure(data= [{

        "x":df.date ,

        "y": df[col],

        "name": col} for col in stores])



fig.update_layout(

    title="Total sales per store",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig.show()
df_weekend= df[-7:].groupby("weekday").sum()[stores]

# plotting the figure

fig = go.Figure(data= [{

        "x":df_weekend.index,

        "y": df_weekend[col],

        "name": col} for col in df_weekend])



fig.update_layout(

    title="Total sales by each store per Day of the Week of the last week of data",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig.show()
# plotting the figure

df_monthly= df.groupby("month").sum()[stores]



fig = go.Figure(data= [{

        "x":df_monthly.index,

        "y": df_monthly[col],

        "name": col} for col in df_monthly])



fig.update_layout(

    title="Monthly sales per store",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)



fig.show()
# Dickey-fuller statistical test 

def ad_fuller(timeseries: pd.DataFrame, significance_level= 0.05):

    

    non_stationary_cols= []

    stationary_cols= []

    

    for col in timeseries.columns: 

        dftest= adfuller(df[col], autolag="AIC")

        if dftest[1] <= significance_level:

            stationary_cols.append({col:{"Test Statistic": dftest[0],

                                         "p-value": dftest[1],

                                         "# Lags": dftest[2],

                                         "# Observations": dftest[3],

                                         "Critical Values": dftest[4],

                                         "Stationary": True}})

        else: 

            non_stationary_cols.append({col:{"Test Statistic": dftest[0],

                                         "p-value": dftest[1],

                                         "# Lags": dftest[2],

                                         "# Observations": dftest[3],

                                         "Critical Values": dftest[4],

                                         "Stationary": False}})

    return non_stationary_cols, stationary_cols

            
non_stationary_cols, stationary_cols= ad_fuller(df[stores])



len(non_stationary_cols), len(stationary_cols)
non_stationary_cols[0]
rolling_mean= df["CA_1"].rolling(window=28, center=False).mean()

rolling_std= df["CA_1"].rolling(window=28, center=False).std() 



fig= go.Figure(data=

               [go.Scatter(x= df["date"],

                           y= df["CA_1"],

                           name= "original", 

                           showlegend=True,

                           marker=dict(color="blue"))])

fig.add_traces([

    go.Scatter(x= df["date"],

                         y=rolling_mean,

                         name= "rolling mean",

                         showlegend= True, 

                         marker=dict(color="red")),

    go.Scatter(x= df["date"],

                         y=rolling_std,

                         name= "rolling std",

                         showlegend= True, 

                         marker=dict(color="black"))])

fig.update_layout(

    title="Store CA_1 Total Sales",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()
# making the data stationary

df["lag-1_CA_1"]= df["CA_1"].diff().fillna(df["CA_1"])



# visualizing stationary data

rolling_mean= df["lag-1_CA_1"].rolling(window=28, center=False).mean()

rolling_std= df["lag-1_CA_1"].rolling(window=28, center=False).std() 



fig= go.Figure(data=

               [go.Scatter(x= df["date"],

                           y= df["lag-1_CA_1"],

                           name= "original", 

                           showlegend=True,

                           marker=dict(color="blue"))])

fig.add_traces([

    go.Scatter(x= df["date"],

                         y=rolling_mean,

                         name= "rolling mean",

                         showlegend= True, 

                         marker=dict(color="red")),

    go.Scatter(x= df["date"],

                         y=rolling_std,

                         name= "rolling std",

                         showlegend= True, 

                         marker=dict(color="black"))])

fig.update_layout(

    title="Store first difference CA_1 Total Sales",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()
# adding new col to stores

stores.append("lag-1_CA_1")
stores
# check for stationarity (our new col is the only stationary col)

_, stationary= ad_fuller(df[stores])

stationary
_, ax= plt.subplots(1, 2, figsize= (10,8))

plot_acf(df["lag-1_CA_1"], lags=10, ax=ax[0]), plot_pacf(df["lag-1_CA_1"], lags=10, ax=ax[1])

plt.show()
model= ARIMA(df["lag-1_CA_1"], order=(8,1,0))

results= model.fit(disp=-1)



fig= go.Figure(data=

               [go.Scatter(x= df["date"],

                           y= df["lag-1_CA_1"],

                           name= "original", 

                           showlegend=True,

                           marker=dict(color="blue"))])

fig.add_trace(

    go.Scatter(x= df["date"],

               y=results.fittedvalues,

               name= "fitted values",

               showlegend= True, 

               marker=dict(color="red")))

fig.update_layout(

    title="Fitted values",

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()

# a closer look

_, ax= plt.subplots(figsize=(12,8))

results.plot_predict(1799, 1940, dynamic=False, ax=ax)

plt.show()
compare_df= pd.DataFrame({"actual": df["CA_1"],

                          "predictions": pd.Series(results.fittedvalues.cumsum(), copy=True),

                          "d": df["d"]}).set_index("d")

compare_df.loc["d_1", "predictions"]= 0
fig= go.Figure(data=

               [go.Scatter(x= compare_df.index[-90:],

                           y= compare_df.iloc[-90:, 0],

                           name= "actual", 

                           showlegend=True,

                           marker=dict(color="blue"))])

fig.add_traces([

                go.Scatter(x= compare_df.index[-90:],

                           y=compare_df.iloc[-90:, 1],

                           name= "predictions",

                           showlegend= True, 

                           marker=dict(color="red"))])

fig.update_layout(

    title="Actual vs Predicted; RMSE %5f" % np.sqrt(sum((compare_df["actual"] - compare_df["predictions"])**2)/len(compare_df)),

    xaxis_title="Dates",

    yaxis_title="Units Sold",

    font=dict(

        family="Arial, monospace",

        size=14,

        color="#7f7f7f"

    )

)

fig.show()