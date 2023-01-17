import math as math

import random as random

import itertools as it

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import warnings

from statsmodels.tsa import holtwinters as hw

from statsmodels.tsa.arima.model import ARIMA as ARIMA

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from numpy.linalg import LinAlgError



# Import

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")

sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")



# Clean so day is number only

sales.rename({"d_{}".format(i) : i for i in range(1,1942)}, axis = 1, inplace = True)

calendar.loc[:,"d"] = calendar.loc[:,"d"].str.extract("d_(\d+)").iloc[:,0].astype(int)

# Set prices in sales format

## Note not all prices available, some na

prices.loc[:,"id"] = prices.loc[:,"item_id"]+"_"+prices.loc[:,"store_id"]+"_evaluation"

prices.drop(["store_id","item_id"],axis=1,inplace=True)

prices = prices.merge(calendar.loc[:,["wm_yr_wk","d"]], on="wm_yr_wk",how="left").drop("wm_yr_wk",axis=1)

prices = prices.pivot("id","d","sell_price").rename_axis(columns=None)

prices = prices.merge(sales.loc[:,["id","store_id","state_id"]],left_index=True,right_on="id")



# Highest day number in sales evaluation data

max_pred = sales.select_dtypes("number").columns.max()



#

warnings.simplefilter('error', ConvergenceWarning)
begin = 1

start = 1942



# Total sales over time

sales_total = sales.loc[:,range(begin,start)].sum(axis=0).to_frame("total")

sales_total["mean"] = sales_total["total"].rolling(7).mean()



fig = plt.figure(figsize = (8,4), dpi = 100)

ax = fig.add_subplot(1,1,1)



ax.plot(range(begin,start), sales_total["total"], linewidth = 0.5)

ax.plot(range(begin,start), sales_total["mean"])



def labels(axis, title="Total Sales") :

    axis.set_title(title)

    axis.set_xlabel("Day")

    axis.set_ylabel("Sales")

labels(ax)

ax.legend(("Total","Rolling 7-day average"))



plt.show()



# Total sales over time by state and store

sales_total = sales.loc[:,["store_id"]+list(range(begin,start))]

sales_total = sales_total.groupby("store_id").sum()



fig = plt.figure(figsize = (8,4), dpi = 100)

ax = fig.add_subplot(1,1,1)



colours_state = {"CA" : "green", "TX" : "red", "WI" : "blue"}

lines = []

lines_names = []

linestyles = ["-","--",":"]

linestyles = linestyles + ["-."] + linestyles*2

num = 0

for store_id in sales_total.index :

    line = ax.plot(range(begin,start),

                   sales_total.loc[store_id,:].rolling(7).mean(),

                   color = colours_state[store_id[0:2]],

                   linestyle = linestyles[num],

                   linewidth = 0.5)

    lines.append(line)

    lines_names.append(store_id)

    

    num += 1

    

labels(ax, title="Store Total Sales 7-day Rolling Average")

def legend(ax, names = None) : ##? lines,

    ax.legend(names, loc = "center left", bbox_to_anchor = [1.01,0.5])

legend(ax, names=lines_names)



plt.show()



# Total sales over time by category and department

sales_total = sales.loc[:,["state_id","dept_id"]+list(range(begin,start))]

sales_total = sales_total.groupby(["state_id","dept_id"]).sum().reset_index()



fig = plt.figure(figsize = (10,3), dpi = 100)

fig.suptitle("Department Total Sales 7-day Rolling Average",y=1)



colours_dept = {"HOBBIES" : "green", "HOUSEHOLD" : "red", "FOODS" : "blue"}

lines = []

lines_names = []

linestyles = ["-","--"]

linestyles = linestyles + [":"] + linestyles*2

count=1

for state_id in sales.loc[:,"state_id"].unique() :

    ax = fig.add_subplot(1,3,count)

    ax.set_title(state_id)

    ax.set_xlabel("Day")

    ax.set_ylabel("Sales")

    

    num = 0

    for dept_id in sales_total.loc[sales_total.loc[:,"state_id"]==state_id,"dept_id"].unique() :

        line = ax.plot(range(begin,start), 

                       sales_total.loc[(sales_total.loc[:,"dept_id"]==dept_id)

                                       & (sales_total.loc[:,"state_id"]==state_id),

                                       range(begin,start)]

                                  .iloc[0,:]

                                  .rolling(7)

                                  .mean(), 

                       color = colours_dept[dept_id.split("_")[0]],

                       linestyle = linestyles[num],

                       linewidth = 0.5)

        lines.append(line)

        lines_names.append(dept_id)



        num += 1

    

    count += 1



ax.legend(lines_names, 

          loc = "center", 

          bbox_to_anchor = [-0.7,-0.25], 

          ncol=len(lines_names),

          prop={"size" : 7})



plt.show()



del sales_total
begin = 1942-28

start = 1942



# Total sales over time

sales_total = sales.loc[:,range(begin,start)].sum(axis=0).to_frame("total")



fig = plt.figure(figsize = (8,4), dpi = 100)

ax = fig.add_subplot(1,1,1)



ax.plot(range(begin,start), sales_total, linewidth = 2)



def labels(axis, title="Total Sales") :

    axis.set_title(title)

    axis.set_xlabel("Day")

    axis.set_ylabel("Sales")

labels(ax)



plt.show()



# Total sales over time by state and store

sales_total = sales.loc[:,["store_id"]+list(range(begin,start))]

sales_total = sales_total.groupby("store_id").sum()



fig = plt.figure(figsize = (8,4), dpi = 100)

ax = fig.add_subplot(1,1,1)



colours_state = {"CA" : "green", "TX" : "red", "WI" : "blue"}

lines = []

lines_names = []

linestyles = ["-","--",":"]

linestyles = linestyles + ["-."] + linestyles*2

num = 0

for store_id in sales_total.index :

    line = ax.plot(range(begin,start),

                   sales_total.loc[store_id,range(begin,start)],

                   color = colours_state[store_id[0:2]],

                   linestyle = linestyles[num],

                   linewidth = 0.5)

    lines.append(line)

    lines_names.append(store_id)

    

    num += 1

    

labels(ax, title="Store Total Sales")

def legend(ax, names = None) : ##? lines,

    ax.legend(names, loc = "center left", bbox_to_anchor = [1.01,0.5])

legend(ax, names=lines_names)



plt.show()



# Total sales over time by category and department

sales_total = sales.loc[:,["state_id","dept_id"]+list(range(begin,start))]

sales_total = sales_total.groupby(["state_id","dept_id"]).sum().reset_index()



fig = plt.figure(figsize = (10,3), dpi = 100)

fig.suptitle("Department Total Sales", y=1)



colours_dept = {"HOBBIES" : "green", "HOUSEHOLD" : "red", "FOODS" : "blue"}

lines = []

lines_names = []

linestyles = ["-","--"]

linestyles = linestyles + [":"] + linestyles*2

count=1

for state_id in sales.loc[:,"state_id"].unique() :

    ax = fig.add_subplot(1,3,count)

    ax.set_title(state_id)

    ax.set_xlabel("Day")

    ax.set_ylabel("Sales")

    

    num = 0

    for dept_id in sales_total.loc[sales_total.loc[:,"state_id"]==state_id,"dept_id"].unique() :

        line = ax.plot(range(begin,start), 

                       sales_total.loc[(sales_total.loc[:,"dept_id"]==dept_id)

                                       & (sales_total.loc[:,"state_id"]==state_id),

                                       range(begin,start)]

                                  .iloc[0,:], 

                       color = colours_dept[dept_id.split("_")[0]],

                       linestyle = linestyles[num],

                       linewidth = 0.5)

        lines.append(line)

        lines_names.append(dept_id)



        num += 1

    

    count += 1



ax.legend(lines_names, 

          loc = "center", 

          bbox_to_anchor = [-0.7,-0.25], 

          ncol=len(lines_names),

          prop={"size" : 7})



plt.show()



del sales_total
# Naive method

## array: 1d pandas array

## begin, start, end: model input [begin,start), prediction [start,end], begin used for the naive day ##?

## return: 1d pandas array, only predictions [start,end], same format as array

def predict_naive(predictor, begin=None, start=None, end=None) :    

    if begin==None : begin = predictor.index.max()

    if start==None : start = begin+1

    if end==None : end = start+28-1

        

    return pd.Series([predictor.loc[begin]]*(end-start+1),

                     range(start,end+1))



# Apply naive to sales

sales_naive = sales.loc[:,[max_pred]].apply(

    lambda x : predict_naive(x, begin=max_pred, start=max_pred+1, end=max_pred+28),

    axis=1

)

sales_naive = pd.merge(sales.loc[:,["id"]],

                       sales_naive,

                       how="right",

                       left_index=True,

                       right_index=True)



# Mean method

def predict_mean(predictor, begin=None, start=None, end=None) :    

    if begin==None : begin = predictor.index.min()

    if start==None : start = predictor.index.max()+1

    if end==None : end = start+28-1

    

    mean = predictor.loc[range(begin,start)].mean()

    mean = round(mean)

    

    return pd.Series([mean]*(end-start+1),

                     range(start,end+1)).astype(int)



# Apply to mean sales

start = max_pred+1

begin = start-28

end = start+28-1



sales_mean = sales.apply(

    lambda x : predict_mean(x, begin, start, end),

    axis=1

)

sales_mean = pd.merge(sales.loc[:,["id"]],

                      sales_mean,

                      left_index=True,

                      right_index=True)
# Plot example prediction

fig = plt.figure(figsize = (8,2))

ax = fig.add_subplot(1,1,1)



ax.plot(range(begin,start),

        sales.loc[3,range(begin,start)])

ax.plot(range(start,end+1),

        sales_naive.loc[3,range(start,end+1)],

        linestyle = "--")

ax.plot(range(start,end+1),

        sales_mean.loc[3,range(start,end+1)],

        linestyle = "--")



ax.set_title("Example: {}".format(sales.loc[3,"id"]))

ax.legend(["Observed", "Naive", "Mean"], loc = "center left", bbox_to_anchor = [1.01,0.5])



plt.show()
# Time series rolling cross-validation

## return: pandas DataFrame with columns of

##         averaged mean error, MAE, MSE if output="aggregate"

##       : pandas DataFrame with rows kth fold

##         and columns of k, start, residual days if output="residuals"

### Note ts cv done on all or some of predictors

def cross_validation(

        function = None, # function: kwargs predictor, begin, start, end.  MUST return df

        predictor = None, # pandas dataframe, series x days, MUST HAVE UNIQUE INDEX

        window_prior = None, # Days as input to function, i.e.start-begin

        window_post = None, # Days as output of function, i.e. end-start+1

        k_folds = 10) :

    

    #

    out_array = pd.DataFrame()



    # Interval between each cv fold

    if k_folds == "all" :

        k_folds = predictor.shape[-1]-window_prior-window_post+1

        interval = 1

    else :

        interval = int((predictor.shape[-1]-window_prior-window_post+1)

                       / k_folds)

    if predictor.shape[-1] < window_prior+window_post :

        raise Exception("Not enough days with given window sizes.")

    

    # For each k_fold,

    for k in range(1, k_folds+1) :

        # Set begin, start, end

        begin = predictor.columns[0] + interval*(k-1)

        start = begin + window_prior

        end = start + window_post - 1



        # Prediction

        prediction = function(predictor = predictor,

                              begin = begin,

                              start = start,

                              end = end)



        # Residuals

        residuals = (prediction - predictor.loc[:,range(start,end+1)]).T.reset_index(drop=True).T

        

        # Append residuals to out_array

        residuals = pd.merge(pd.DataFrame({"k" : [k]*residuals.shape[0],

                                           "start" : [start]*residuals.shape[0]}, 

                                          index = residuals.index),

                             residuals,

                             left_index=True,

                             right_index=True)



        out_array = out_array.append(residuals)



    out_array.loc[:,"k"] = out_array.loc[:,"k"].astype(int)



    return out_array



# Calculate cross-validated residuals. 

## DataFrame columns of fold, start, day. Index is id.

cv_mean = cross_validation(

    lambda predictor,begin,start,end : 

        predictor.apply(lambda x : predict_mean(x,begin,start,end),

                        axis=1),

    sales.set_index("id").loc[:,range(max_pred-28*10,max_pred)],

    window_prior = 28,

    window_post = 28,

    k_folds=10

)
# Statistics

array = cv_mean.drop(["k","start"],axis=1).stack()

print("Mean error: {}\nMAE: {}\nRMSE: {}".format(

    array.mean(),

    array.abs().mean(),

    np.sqrt((array**2).mean())

))



# Plot

fig = plt.figure(figsize = (12,4), dpi = 100)

ax = fig.add_subplot(1,2,1)



##

ax.hist(array,

        bins = range(int(array.min()),

                     int(array.max())+1,

                     max(int((array.max()-array.min())/100),1)), 

        log=True)



ax.set_title("Residuals Histogram")

ax.set_xlabel("Residual")

ax.set_ylabel("Number of predictions")



plt.show()
#

def hp_tuner(

        params_list, # list of parameters for which t-s c-v each run on function

        function = None, # function: kwargs predictor, begin, start, end, parameters

        predictor = None, # 2d pandas array series x days

        window_prior = None, # List of days, each as input to function, i.e.start-begin

        window_post = None, # List of days to predict, each as output of function, i.e. end-start+1

        k_folds = 10

    ) :

    

    #

    array = pd.DataFrame()

    

    #

    for parameter in params_list :

        #

        out = cross_validation(lambda predictor,begin,start,end : 

                                   function(predictor=predictor, 

                                            begin=begin, 

                                            start=start, 

                                            end=end, 

                                            parameter=parameter),

                               predictor=predictor,

                               window_prior=window_prior,

                               window_post=window_post,

                               k_folds=k_folds)

        

        #

        out = out.T.append(pd.DataFrame([[parameter for i in range(0,out.index.shape[0])]], 

                                        columns=out.index, 

                                        index=["parameter"])).T

        

        array = array.append(out)



    return array



#

params_list = [7*i for i in range(0,10)]

#

hp = hp_tuner(

    params_list,

    lambda predictor,begin,start,end,parameter :

        predictor.apply(lambda x : predict_mean(x,begin+parameter,start,end),

                        axis=1),

    sales.set_index("id").loc[:,range(max_pred-28*10,max_pred)],

    window_prior = max(params_list)+7,

    window_post = 28,

    k_folds=10

)

hp_mean = (hp.drop(["start"],axis=1)

             .groupby(["parameter","k"])

             .apply(lambda x : x.drop(["parameter","k"],axis=1)

                                .abs()

                                .stack()

                                .mean())

             .reset_index()

             .drop("k",axis=1)

             .groupby("parameter",as_index=False)

             .mean()

             .rename({0:"MAE"},axis=1)

)



# Plot

fig = plt.figure(figsize = (6,2), dpi = 100)



ax = fig.add_subplot(1,1,1)

ax.plot(max(params_list)+7-hp_mean.loc[:,"parameter"],

        hp_mean.loc[:,"MAE"])

ax.set_xlabel("Hyper-parameter")

ax.set_ylabel("Mean Absolute Error")



plt.show()
def submission(array, file_name_add="") :

    sales_prediction = (array.loc[:,["id"]+list(range(1914,1914+28))]

                             .rename({1914+i-1 : "F"+str(i) for i in range(1,29)},

                                     axis=1)

                       )

    sales_prediction.loc[:,"id"] = sales_prediction.loc[:,"id"].str.replace("evaluation", "validation")

    

    sales_prediction.append(

        array.loc[:,["id"]+list(range(1914+28,1914+28*2))]

             .rename({1914+28+i-1 : "F"+str(i) for i in range(1,29)},

                     axis=1)

    ).to_csv(path_or_buf = "M5_accuracy_submission_{}.csv".format(file_name_add),

             index=False)



# Validation days

## Apply to naive sales

sales_naive_val = sales.apply(

    lambda x : predict_naive(x, max_pred-28, max_pred-28+1, max_pred-28+28),

    axis=1

)

sales_naive_val = pd.merge(sales.loc[:,["id"]],

                           sales_naive_val,

                           left_index=True,

                           right_index=True)

## Apply to mean sales

sales_mean_val = sales.apply(

    lambda x : predict_mean(x, begin-28, start-28, end-28),

    axis=1

)

sales_mean_val = pd.merge(sales.loc[:,["id"]],

                          sales_mean_val,

                          left_index=True,

                          right_index=True)



# Merge

sales_naive = sales_naive_val.merge(sales_naive, on="id", how="outer")

sales_mean = sales_mean_val.merge(sales_mean, on="id", how="outer")



# Save submission

submission(sales_naive, file_name_add = "naive")

submission(sales_mean, file_name_add = "mean")



## Description:

## The naive forecasting model.

## The mean forecasting model.
#

start = max_pred+1

begin= start-28*4#d 1

end = start+28-1



# Smoothing

sales_total = sales.loc[:,["store_id"]+list(range(1,start))].groupby("store_id").sum()



warnings_SHW = []



# Apply seasonal Holt-Winters

## array: 2d pandas DataFrame 1 x days

## begin, start, end: model input [begin,start), prediction [start,end]

## return: pandas Series, only predictions [start,end], same format as array

def predict_SHW(predictor, begin=None, start=None, end=None, warnings=None) :

    if begin==None : begin = predictor.columns.min()

    if start==None : start = predictor.columns.max()+1

    if end==None : end = start+28-1

    

    # Parameters, begin at 0

    hw_train = hw.ExponentialSmoothing(predictor.iloc[0].loc[range(begin,start)].tolist(),

                                       trend = "add",

                                       seasonal = "add",

                                       seasonal_periods = 7)

    try : 

        # Fit

        hw_train = hw_train.fit()



        # Prediction - first time at zero

        hw_prediction = hw_train.predict(start = start-begin, 

                                         end = end-begin)

        # shift time to first time at 1 (i.e. back to start)

        hw_prediction = pd.Series(list(hw_prediction),

                                  range(start,end+1))

    except ConvergenceWarning :

        hw_prediction = pd.Series([np.nan]*(end-start+1), index=range(start,end+1))

        if warnings is not None :

            warnings.append(predictor.index[0])

        else : 

            print("ConvergenceWarning")

    

    return hw_prediction



shw_prediction = sales_total.apply(lambda x : predict_SHW(pd.DataFrame(x).T, 

                                                          warnings=warnings_SHW), 

                                   axis=1)

print("Did not converge: {}".format(warnings_SHW))



# Plot

## observed, prediction: 2d arrays, must have same indices, indices used as plot titles

## begin, start, end: observed plotted [begin,start), prediction plotted [start,end]

## fig_title: whole figure title (suptitle)

def plot_forecasts(observed, 

                   prediction=None,

                   fig_title=None) :    

    fig = plt.figure(figsize = (8,1*len(observed.index)), dpi = 100)

    fig.suptitle(fig_title, y=1.015)

    

    count = 1

    for index in observed.index :

        ax = fig.add_subplot(math.ceil(len(observed.index)/2),

                             min(observed.shape[0],2),

                             count)

        

        ax.plot(observed.columns, 

                observed.loc[index,:],

                linewidth = 0.5,

                color = "gray")

        if prediction is not None :

            ax.plot(prediction.columns, 

                    prediction.loc[index,:], 

                    linewidth = 0.5,

                    color = "red")

            maximum = max(observed.loc[index,:].max(),

                          prediction.loc[index,:].max())

            minimum = min(observed.loc[index,:].min(),

                          prediction.loc[index,:].min())

            ax.vlines(start, ymin = minimum, ymax = maximum, linewidth = 0.5, linestyle = "--")

        

        ax.set_title(index)

        ax.set_xlabel("Day")

        ax.set_ylabel("Total Sales")

        

        count += 1

    

    if len(observed.index) >= 2 : fig.tight_layout()



    plt.show()



plot_forecasts(sales_total.drop(warnings_SHW).loc[:,range(start-28*4,start)], 

               shw_prediction.dropna())
# Cross-validation

cv_shw = cross_validation(

    lambda predictor,begin,start,end : 

        predictor.apply(lambda x : predict_SHW(pd.DataFrame(x).T,begin,start,end),

                        axis=1),

    sales_total.loc[:,range(1,max_pred+1)],

    window_prior = 1600,

    window_post = 28,

    k_folds=10

)



# Statistics

print("Mean error: {}\nMAE: {}\nRMSE: {}".format(

    cv_shw.drop(["k","start"],axis=1).stack().mean(),

    cv_shw.drop(["k","start"],axis=1).stack().abs().mean(),

    np.sqrt((cv_shw.drop(["k","start"],axis=1).stack()**2).mean())

))



# Plot errors

# Plot

## Output: Prints histogram or line/point plots of residuals

def plot_residuals(

        array, # Pandas DataFrame containing residuals of series x days

        group=None, # group for plotting lines if graph="line"

        title=None, # Plot title

        quantile=0.95

    ) :

    

    #? outliers + non_outliers = array ???

    outliers = (array.reset_index(drop=True)

                     .apply(lambda x : 

                                x.loc[x.abs().nlargest(int((1-quantile)*x.shape[0])).index]

                                 .reset_index(drop=True)

                           )

               )

    wedge = array.quantile([1-quantile,quantile],interpolation="nearest")

    

    fig = plt.figure(figsize = (8,4), dpi = 100)

    ax = fig.add_subplot(1,1,1)

    

    ax.fill_between(x = wedge.columns.to_list(),

                    y1 = wedge.loc[1-quantile,:].to_list(),

                    y2 = wedge.loc[quantile,:].to_list())

    ax.scatter(outliers.stack().reset_index(1).loc[:,"level_1"], 

               outliers.stack().reset_index(drop=True),

               s=2**2)



    MAE = array.stack().reset_index(1).sort_values("level_1").groupby("level_1",as_index=False).mean()

    ax.plot(MAE.loc[:,"level_1"], 

            MAE.iloc[:,1],

            c="#76a834",

            linewidth=3)

    ax.legend(["Mean absolute error","95th percentile range", "Outlier residual"], 

              loc = "center left", 

              bbox_to_anchor = [1.01,0.5])

    #print(MAE.loc[:,"level_1"], 

    #      MAE.iloc[:,1])

    

    ax.set_title(title)

    ax.set_xlabel("Forecast Days Ahead")

    ax.set_ylabel("Residual")

    

    plt.show()

plot_residuals(cv_shw.drop(["k","start"], axis=1),quantile=0.95)
# Calculate hierarchical distribution 1 level down using mean method

## Returns: prediction under hierarchical mean model with group column

def hierarchical_mean(level0, # pandas DataFrame, prediction of higher aggregate sales,

                              # only columns of index and predictions

                      observed, # pandas DataFrame, prior observations, only columns

                                # of index, group and observed values

                      group="store_id", # column in observed containing group that is in the index of level0

                      begin=None, 

                      start=None, 

                      end=None) :    

    if begin==None : begin = observed.select_dtypes('number').columns.column.min()

    if start==None : start = level0.select_dtypes('number').column.min()

    if end==None : end = level0.select_dtypes('number').column.max()

    

    # Mean prediction in "p"

    sales_prediction = observed.loc[:,[group]+list(range(begin,start))] ##d [index,group]

    sales_prediction.loc[:,"p"] = (

        sales_prediction.loc[:,range(begin,start)].apply(lambda x : x.mean(),

                                                         axis=1)

    )

    sales_prediction.drop(range(begin,start), axis=1, inplace=True)

    # Calculate p from mean

    sums = sales_prediction.loc[:,[group,"p"]].groupby(group).sum().loc[:,"p"]

    sales_prediction.loc[:,"p"] = sales_prediction.apply(lambda x : x["p"] / sums[x[group]],

                                                         axis=1)

    # Merge with level0 and *p

    sales_prediction = pd.merge(sales_prediction, level0, on = group, how = "left")

    sales_prediction.loc[:,range(start,end+1)] = (

        sales_prediction.apply(lambda x : x.loc[range(start,end+1)] * x.loc["p"],

                               axis=1)

    ).applymap(lambda x : np.nan if np.isnan(x)

                          else round(x))



    return sales_prediction.drop("p",axis=1)



# Replace non-converged with mean

sales_total_mean = sales_mean.merge(sales.loc[:,["id","store_id"]],on="id",how="left")

sales_total_mean = sales_total_mean.drop("id",axis=1).groupby("store_id").sum()

shw_prediction = shw_prediction.fillna(sales_total_mean)



# Caluclate sales from shw_prediction

sales_SHW = hierarchical_mean(shw_prediction.reset_index()

                                            .rename({"index" : "store_id"},axis=1)

                                            .fillna(sales_mean), 

                              sales,

                              group="store_id",

                              begin=start-28,

                              start=start,

                              end=end)

sales_SHW = sales.loc[:,["id"]].merge(sales_SHW, left_index=True, right_index=True)
# Validation

# Apply to mean sales

warnings_SHW_val = []

shw_prediction_val = sales_total.apply(lambda x : predict_SHW(pd.DataFrame(x).T,

                                                              begin=begin,

                                                              start=start-28,

                                                              end=end-28,

                                                              warnings=warnings_SHW_val),

                                       axis=1)

print("Did not converge: {}".format(warnings_SHW_val))

# Replace non-converged with mean

shw_prediction_val = shw_prediction_val.fillna(sales_total_mean)



# Hierarchical for sales

sales_SHW_val = hierarchical_mean(shw_prediction_val.reset_index()

                                                    .rename({"index" : "store_id"},axis=1),

                                  sales,

                                  group="store_id",

                                  begin=max_pred+1-28-28,

                                  start=start-28,

                                  end=max_pred)

sales_SHW_val = sales.loc[:,["id"]].merge(sales_SHW_val, left_index=True, right_index=True)



# Merge with evaluation prediction

sales_SHW = sales_SHW_val.drop("store_id",axis=1).merge(sales_SHW.drop("store_id",axis=1),

                                                        on="id",

                                                        how="outer")



# Submission - validation

submission(sales_SHW, file_name_add="SHW")



# Description:

# Seasonal Holt-Winters of store total sales with hierarchical distribution factor given by mean of last 112 days in training data.



warnings.simplefilter('default')
from statsmodels.tsa import stattools as stats



#

start = max_pred+1

begin= start-28*4

end = start+28-1



#

sales_total = sales.loc[:,["store_id"]+list(range(1,max_pred+1))].groupby("store_id").sum()
# Difference array entries d times and/or with difference m (x_m-x_1)

## Returns: pandas DataFrame, differenced with ALL na columns removed

### Note ALL columns containing na removed.

def diff(array, d, D, m=0) :

    new_array = array.copy()

    # differenced d

    for i in range(0, d) :

        new_array = new_array - new_array.shift(1, axis=1)

    # seasonal m

    if m > 0 :

        # differenced D

        for i in range(0, D) :

            new_array = new_array - new_array.shift(m, axis=1)

    return new_array.dropna(axis=1)



#

# plot_forecasts(sales_total.loc[:,range(begin,start)], fig_title="No Differencing")

#

plot_forecasts(diff(sales_total,d=0,D=1,m=7).loc[:,range(begin,start)], fig_title="d=0, D=1, m=7")

# plot_forecasts(diff(sales_total,d=1,D=1,m=7).loc[:,range(begin,start)], fig_title="d=1, D=1, m=7")

#

plot_forecasts(diff(sales_total.loc[["WI_2","WI_3"],:], d=0, D=1,m=28).loc[:,range(begin,start)], 

               fig_title="d=0, D=1, m=28")
# ACF and PACF graphs

## array: 

def corr_graphs(array, fig_title = None) :

    if len(array.shape) < 2 :

        new_array = pd.DataFrame([array.copy()])

    else :

        new_array = array.copy()

        

    #

    fig = plt.figure(figsize = (12,2*new_array.shape[0]), dpi = 125)

    fig.suptitle(fig_title, y=1.015, size=18)

    

    nrows = new_array.shape[0]

    count = 1

    for index in new_array.index :        

        row = new_array.loc[index,:]

        #

        n_lags = 20

        # ACF

        acf, acf_conf, q_stat, q_stat_p = stats.acf(row, nlags = n_lags, alpha = 0.05, qstat=True, fft=True)

        # PACF

        pacf, pacf_conf= stats.pacf(row, nlags = n_lags, alpha = 0.05)

        

        # Plot ACF

        ax1 = fig.add_subplot(nrows,2,1+(count-1)*2)

        ax1.bar(range(0,n_lags+1), acf, width = 0.25, color = "blue")

        ax1.fill_between(range(0,n_lags+1), acf_conf[:,0], acf_conf[:,1], linewidth = 0.5, color = "gray", alpha = 0.25)

        ax1.hlines(y = [1.96/math.sqrt(new_array.shape[1]),

                        -1.96/math.sqrt(new_array.shape[1])], 

                   xmin = 0, xmax = n_lags, 

                   linestyles="dashed")

        

        ax1.set_ylim(-1,1)

        ax1.set_title("Store {store}".format(store=index))

        ax1.set_xlabel("Lag")

        ax1.set_ylabel("ACF")

        plt.grid(True)    

        

        # Plot PACF

        ax2 = fig.add_subplot(nrows,2,2+(count-1)*2)

        ax2.bar(range(0,n_lags+1), pacf, width = 0.25, color = "blue")

        ax2.fill_between(range(0,n_lags+1), pacf_conf[:,0], pacf_conf[:,1], linewidth = 0.5, color = "gray", alpha = 0.25)

        ax2.hlines(y = [1.96/math.sqrt(new_array.shape[1]),

                        -1.96/math.sqrt(new_array.shape[1])], 

                   xmin = 0, xmax = n_lags, 

                   linestyles="dashed")

        

        ax2.set_ylim(-1,1)

        ax2.set_title("Store {store}".format(store=index))

        ax2.set_xlabel("Lag")

        ax2.set_ylabel("PACF")

        plt.grid(True)

        

        count += 1

    

    fig.tight_layout()

    

    plt.show()

    

corr_graphs(diff(sales_total.loc[:,range(begin,start)],d=0,D=1,m=7), 

            fig_title = "d=0, D=1, m=7")

corr_graphs(diff(sales_total.loc[["WI_2","WI_3"],range(begin,start)],d=0,D=1,m=28), 

            fig_title = "d=0, D=1, m=28")
# Predict with SARIMA model

## predictors: pandas DataFrame 1 x days

## order, seasonal_order: see statsmodels

## return: pandas Series containing predictions and index of days

def predict_SARIMA(predictor, 

                   order, 

                   seasonal_order, 

                   begin=None, 

                   start=None, 

                   end=None, 

                   AICc=None,

                   warnings=None) :

    if begin==None : begin = predictor.columns.min()

    if start==None : start = predictor.columns.max()+1

    if end==None : end = start+28-1

        

    try : 

        # Fit SARIMA

        model = ARIMA(predictor.iloc[0].loc[range(begin,start)].tolist(), 

                      order = order, 

                      seasonal_order = seasonal_order)

        fit = model.fit()



        # Predict

        SARIMA_prediction = pd.Series(fit.predict(start=start-begin,

                                      end=end-begin)

                                     )

        if AICc is not None :

            AICc.append([predictor.index[0],

                         str([order,seasonal_order]),

                         str((order[1],seasonal_order[1])),

                         fit.aicc])

    except (ConvergenceWarning, LinAlgError) as e :

        SARIMA_prediction = pd.Series([np.nan]*(end-start+1))

        if warnings is not None :

            warnings.append([predictor.index[0], str([order, seasonal_order])])

        else :

            print(e)

    

    # Re-index

    SARIMA_prediction.rename({i : start+i for i in range(0,end-start+1)},

                             inplace=True)#, axis=1

    SARIMA_prediction.name = predictor.index[0]



    return SARIMA_prediction



# List of parameters to try

# Index of stores (unique) with list of many lists [order, seasonal_order]

## With 7 day period

params_list_7 = it.product(it.product([0,1],repeat=3),

                           it.product([0,1],repeat=3))

params_list_7 = pd.Series(params_list_7)

params_list_7 = params_list_7.apply(

    lambda x : [x[0],tuple(list(x[1])+[7])]

)

params_list_7 = params_list_7.reset_index().rename({"index":"param_code",

                                                    0:"parameter"},

                                                   axis=1)

## With 28 day period

params_list_28 = it.product(it.product([0,1],repeat=3),

                            it.product([0,1],repeat=3))

params_list_28 = pd.Series(params_list_28)

params_list_28 = params_list_28.apply(

    lambda x : [x[0],tuple(list(x[1])+[28])]

)

params_list_28 = params_list_28.reset_index().rename({"index":"param_code",

                                                      0:"parameter"},

                                                     axis=1)

# All params

params_list = (pd.concat([params_list_7,params_list_28])

                 .drop("param_code",axis=1)

                 .reset_index(drop=True)

                 .reset_index()

                 .rename({"index":"param_code"},axis=1))



#

AICc=[]

warnings_SARIMA = []

hp_7 = hp_tuner(

    params_list_7.loc[:,"parameter"],

    lambda predictor,begin,start,end,parameter : 

        predictor.apply(lambda x : predict_SARIMA(pd.DataFrame(x).T,

                                                  parameter[0],

                                                  parameter[1],

                                                  begin,

                                                  start,

                                                  end,

                                                  AICc=AICc,

                                                  warnings=warnings_SARIMA),

                        axis=1),

    sales_total.loc[:,range(max_pred-28*10,max_pred)],

    window_prior = 28*4,

    window_post = 28,

    k_folds=10

)

# With 28 days to WI_2 and WI_3

hp_28 = hp_tuner(

    params_list_28.loc[:,"parameter"],

    lambda predictor,begin,start,end,parameter : 

        predictor.apply(lambda x : predict_SARIMA(pd.DataFrame(x).T,

                                                  parameter[0],

                                                  parameter[1],

                                                  begin,

                                                  start,

                                                  end,

                                                  AICc=AICc,

                                                  warnings=warnings_SARIMA),

                        axis=1),

    sales_total.loc[["WI_2","WI_3"],range(max_pred-28*10,max_pred)],

    window_prior = 28*4,

    window_post = 28,

    k_folds=10

)

# Join

hp = pd.concat([hp_7.reset_index(),hp_28.reset_index()])

del hp_28



# Include param_code

hp.loc[:,"parameter_key"] = hp.loc[:,"parameter"].astype(str)

params_list.loc[:,"parameter_key"] = params_list.loc[:,"parameter"].astype(str)

hp = hp.merge(params_list.drop("parameter",axis=1),

              on="parameter_key", 

              how="left")

hp.drop("parameter_key", axis=1, inplace=True)

params_list.drop("parameter_key", axis=1, inplace=True)



# Print number of warnings for each store_id and parameter

def count_warnings(warnings_list) :

    # Count warnings

    warnings_count = (

        pd.DataFrame(warnings_list, columns=["store_id","parameter"])

          .groupby("store_id")

          .apply(lambda x : x.loc[:,"parameter"]

                             .value_counts())

          .reset_index()

    )

    if warnings_count.shape[1]==2 : 

        warnings_count.loc[:,"parameter"] = warnings_count.columns[1]

        warnings_count.columns = ["store_id","count", "parameter"]

        warnings_count = warnings_count.loc[:,["store_id","parameter","count"]]

    else :

        warnings_count = warnings_count.rename({"level_1":"parameter","parameter":"count"},

                                               axis=1)

    

    # Include param_code

    params_list.loc[:,"parameter_key"] = params_list.loc[:,"parameter"].astype(str)

    warnings_count = (warnings_count.merge(params_list.drop("parameter",axis=1), 

                                           left_on="parameter",

                                           right_on="parameter_key",

                                           how="left")

                                    .drop("parameter_key",axis=1)

                     )

    params_list.drop("parameter_key", axis=1, inplace=True)

    

    warnings_count = warnings_count.sort_values(["store_id","count","param_code"])

    

    return warnings_count



if len(warnings_SARIMA) > 0 :     

    # Count warnings

    warnings_count_SARIMA = count_warnings(warnings_SARIMA)

    print("Did not converge/complete:\n{}\n".format(warnings_count_SARIMA))
# For MAE, AICc averaged over folds

hp_stats = hp.copy()



# MAE

## Mean over days

hp_stats.loc[:,"MAE"] = (hp_stats.drop(["store_id","parameter","param_code","start","k"],

                                       axis=1)

                                 .abs()

                                 .mean(axis=1))

hp_stats = hp_stats.loc[:,["store_id","param_code","MAE"]]

## Mean over folds

hp_stats = (hp_stats.groupby(["store_id","param_code"])

                    .mean()

                    .reset_index())

hp_stats = hp_stats.merge(params_list, on="param_code", how="left")



# AICc

## Store AICc

params_list.loc[:,"parameter_key"] = params_list.loc[:,"parameter"].astype(str)

hp_AICc = (pd.DataFrame(AICc,columns=["store_id","parameter","diff","AICc"])

             .merge(params_list.drop("parameter",axis=1), 

                     left_on="parameter",

                     right_on="parameter_key",

                     how="left")

              .drop(["parameter_key","parameter"],axis=1))

params_list.drop("parameter_key", axis=1, inplace=True)

## Mean over folds

hp_AICc = hp_AICc.groupby(["store_id","diff","param_code"]).mean().reset_index()

## Merge with hp_stats

hp_stats = hp_stats.merge(hp_AICc, on=["store_id","param_code"],how="left")

hp_stats = hp_stats.loc[:,["store_id","parameter","diff","MAE","AICc","param_code"]]
# For comparison of approach only.

# # Top MAE

# hp_top_MAE = (hp_stats.sort_values("MAE")

#                       .groupby("store_id")

#                       .head(1)

#                       .sort_values("store_id")

#                       .reset_index(drop=True))

# # Top AICc

# hp_top_AICc = (hp_stats.loc[hp_stats.loc[:,"diff"]=="(0, 1)",:]

#                        .sort_values("AICc")

#                        .groupby(["store_id"])

#                        .head(1)

#                        .sort_values("store_id")

#                        .reset_index(drop=True))



# Top MAE of top AICc for each diff

hp_top_MAE_AICc = (hp_stats.sort_values("AICc")

                           .groupby(["store_id","diff"])

                           .head(1)

                           .sort_values("MAE")

                           .groupby("store_id")

                           .head(1)

                           .sort_values("store_id")

                           .reset_index(drop=True))



#

for top in [hp_top_MAE_AICc] :

    print("Top parameters:\n{}\n".format(

        top.drop("diff",axis=1)

    ))



    # Plot residuals

    top_residuals = top.drop("parameter",axis=1).merge(

        hp,

        on=["store_id","param_code"],

        how="left"

    ).drop(list(top.columns)+["k","start"],axis=1)

    print("Mean error: {}\nMAE: {}\nRMSE: {}\n".format(

        top_residuals.stack().mean(),

        top_residuals.stack().abs().mean(),

        math.sqrt((top_residuals.stack()**2).mean())

    ))

    plot_residuals(top_residuals.astype(float),

                   quantile=0.95)
# Choose hp_top

hp_top = hp_top_MAE_AICc.set_index("store_id")



# Apply final parameters

params_list = pd.DataFrame()

for store_id in hp_top.index :

    params_list = params_list.append(

        pd.Series(

            {"order" : hp_top.loc[store_id, "parameter"][0],

             "seasonal_order" : hp_top.loc[store_id, "parameter"][1]},

            name = store_id

        )

    )



# Final predictions on evaluation data

start = max_pred+1

begin= start-28*4

end = start+28-1

warnings_SARIMA = []

SARIMA_prediction = sales_total.apply(

    lambda x : predict_SARIMA(pd.DataFrame(x.loc[range(begin,start)]).T,

                              order=params_list.loc[x.name,"order"],

                              seasonal_order=params_list.loc[x.name,"seasonal_order"],

                              begin=begin,

                              start=start,

                              end=end,

                              warnings=warnings_SARIMA),

    axis=1

)



# Print number of warnings for each store_id and parameter

if len(warnings_SARIMA) > 0 :     

    # Count warnings

    warnings_count_SARIMA = count_warnings(warnings_SARIMA)

    print("Did not converge/complete:\n{}\n".format(warnings_count_SARIMA))



# Replace non-converged with mean

SARIMA_prediction = SARIMA_prediction.fillna(sales_total_mean)



#

sales_SARIMA = hierarchical_mean(SARIMA_prediction.reset_index().rename({"index" : "store_id"},axis=1),

                                 sales,

                                 group="store_id",

                                 begin=start-28,

                                 start=start,

                                 end=end)

sales_SARIMA = sales.loc[:,["id"]].merge(sales_SARIMA, left_index=True,right_index=True)
# Validation

warnings_SARIMA = []

SARIMA_prediction_val = sales_total.apply(

    lambda x : predict_SARIMA(pd.DataFrame(x).T,

                              order=params_list.loc[x.name,"order"],

                              seasonal_order=params_list.loc[x.name,"seasonal_order"],

                              begin=begin-28,

                              start=start-28,

                              end=end-28, 

                              warnings=warnings_SARIMA),

    axis=1

)



# Print number of warnings for each store_id and parameter

if len(warnings_SARIMA) > 0 :     

    # Count warnings

    warnings_count_SARIMA = count_warnings(warnings_SARIMA)

    print("Did not converge/complete:\n{}\n".format(warnings_count_SARIMA))

    

# Replace non-converged with mean

SARIMA_prediction_val = SARIMA_prediction_val.fillna(sales_total_mean)



sales_SARIMA_val = hierarchical_mean(SARIMA_prediction_val.reset_index().rename({"index" : "store_id"},axis=1),

                                     sales,

                                     begin=start-28-28,

                                     start=start-28,

                                     end=end-28).drop("store_id",axis=1)

sales_SARIMA_val = sales.loc[:,["id"]].merge(sales_SARIMA_val, left_index=True,right_index=True)



# Merge

sales_SARIMA = sales_SARIMA_val.merge(sales_SARIMA, on="id", how="outer")



# Submission - validation

submission(sales_SARIMA, file_name_add="SARIMA")



# Description:

# SARIMA model of store total sales with hierarchical distribution factor given by mean of last 112 days in training data.