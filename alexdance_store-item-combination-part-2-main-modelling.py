import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error , mean_absolute_error
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from sklearn.metrics import accuracy_score
class color:  # Testing to make the heading look a liitle more impressive
   BOLD = '\033[1m'
df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
df.head()
split = "2017-01-01"
df['date'] =  pd.to_datetime(df['date'])
split = "2017-01-01"
df['ItemStoreCombined'] = df['item'].map(str) + '-' + df['store'].map(str) 
# this is used in particular to ensure the rolling forecast data does not leak from 1 item / store combination to the next
df.head()

df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear
df_roll=df.copy() # for the rolling forecast
# for rolling forecast
df_roll['sales-1'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(1).mean().reset_index(0,drop=True)
df_roll['sales-2'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(2).mean().reset_index(0,drop=True)
df_roll['sales-3'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(3).mean().reset_index(0,drop=True)
df_roll['sales-4'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(4).mean().reset_index(0,drop=True)
df_roll['sales-5'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(5).mean().reset_index(0,drop=True)
df_roll['sales-6'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(6).mean().reset_index(0,drop=True)
df_roll['sales-7'] = df_roll.groupby('ItemStoreCombined')['sales'].rolling(7).mean().reset_index(0,drop=True)
df_roll.head(10)
# ConsideredLooking forward but chose not to
df_roll_1_1= df_roll[(df_roll.store==1) & (df_roll.item==1)]
df_roll_2_2 = df_roll[(df_roll.store==2) & (df_roll.item==2)]
df_roll_2_2.head() # to check rolling mean worked. As this is product 2 in store 2 and as Sales--3 has Nan then the rolling mean is not bleeding from earlier data
df_roll_1_1.head()  
df_roll = df_roll.dropna()  
df = df.set_index('date')
df.head()
def calculate_error(test_sales, train_sales , test_prediction, train_prediction):
    # https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    MSE_test = mean_squared_error(y_true=test_sales,  y_pred=test_prediction) # Mean Square Error (MAE)
    MSE_train = mean_squared_error(y_true=train_sales,  y_pred=train_prediction)
    MAE_test = mean_absolute_error(y_true=test_sales,  y_pred=test_prediction) # Mean Absolute Error (MAE)
    MAE_train = mean_absolute_error(y_true=train_sales,  y_pred=train_prediction)
    MAPE = np.mean(np.abs(test_prediction - test_sales  ) **2)  # Mean Absolute Percentage Error (MAPE)
    RMSE  = np.mean(np.sqrt((test_prediction - test_sales) ** 2))    
    return{'MSE_test': MSE_test, 'MSE_train':MSE_train ,'MAE_test': MAE_test, 'MAE_train':MAE_train, 'MAPE':MAPE, 'RMSE':RMSE}
def calculate_errorb(test_sales,  test_prediction):
    # https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
    MSE_test = mean_squared_error(y_true=test_sales,  y_pred=test_prediction) # Mean Square Error (MAE)
    MAE_test = mean_absolute_error(y_true=test_sales,  y_pred=test_prediction) # Mean Absolute Error (MAE)
    MAPE = np.mean(np.abs(test_prediction - test_sales  ) **2)  # Mean Absolute Percentage Error (MAPE)
    RMSE  = np.mean(np.sqrt((test_prediction - test_sales) ** 2))    
    return{'MSE_test': MSE_test ,'MAE_test': MAE_test,  'MAPE':MAPE, 'RMSE':RMSE}
df_roll_store_item = df.groupby(["store","item"]).rolling('7D').sales.mean() 
print(df_roll_store_item)
df_roll_store_item =df_roll_store_item.reset_index()
df_roll_store_item.head()
df_roll_store_item.sample(5)
df_roll_store_item =df_roll_store_item.rename(columns={"sales":"Mean_Amount_7D"})
df_roll_store_item.head()
df_roll_final = df_roll.merge(df_roll_store_item, left_on=['date','store','item'], right_on=['date','store','item'] )
df_roll_final_7days = df_roll_final[(df_roll_final.date >= '2017-01-01') & (df_roll_final.date < '2017-01-08')]
df_roll_final_7days.head()
df_roll_final_7days.sum()
df_roll_final.head()
df_roll_final = df_roll_final.drop (['sales-1', 'sales-2','sales-3', 'sales-4', 'sales-5', 'sales-6',  'sales-7'],axis=1)
df_train = df[ :split ] 
df_test = df[split : ] 
# df_test_final will be the collated way of comparing the sales and all the forecasting options. 
# Every time a new model is run it will be added to this
df_test_final = df_test.copy()
df_test_final =df_test_final.drop (['dayofweek', 'quarter','month', 'year', 'dayofyear', 'weekofyear'],axis=1)
df_test.head()
y_train = df_train.loc[:,'sales']
y_test= df_test.loc[:,'sales']
X_train = df_train.drop (['sales'],axis=1) 
X_test = df_test.drop (['sales'],axis=1)
print(y_train.shape)
print(y_test.shape)
print(X_train.shape)
print(X_test.shape)
X_train.head()
y_train.head()
df_test_final = df_test_final.merge(df_roll_store_item, left_on=['date','store','item'], right_on=['date','store','item'] )
df_test_final.head()
df_roll.head()
df_weighted = df_roll.copy() 
df_weighted['date'] =  pd.to_datetime(df_weighted['date'])
df_weighted = df_weighted.set_index('date')
df_weighted.head()
weights = np.arange(1,11) #this creates an array with integers 1 to 10 included
weights
wma10 = df_weighted['sales'].rolling(10).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
wma10.head(20)
df_weighted['10-day-WMA'] = wma10
df_weighted.head()
wma10.sample(5)
df_weighted.info()
df_weighted.tail()
df_weighted_7days = df_weighted[(df_weighted.index >= '2017-01-01')] 
df_weighted_7days = df_weighted_7days[(df_weighted_7days.index < '2017-01-08')]
df_weighted_7days.head()
RMSE_Weighted_10 =  np.mean(np.sqrt((df_weighted_7days['10-day-WMA'] - df_weighted_7days['sales']) ** 2))    
sma10 = df_weighted['sales'].rolling(10).mean()

df_weighted['sma10'] = sma10
df_weighted_short = df_weighted[split : "2017-03-30"] 
df_weighted_short_1_1 =  df_weighted_short[(df_weighted_short.store==1) & (df_weighted_short.item==1)]
df_weighted_short_1_1.head()
plt.figure(figsize = (12,6))
plt.plot(df_weighted_short_1_1['sales'], label="sales")
plt.plot(df_weighted_short_1_1['10-day-WMA'], label="10-Day WMA")
plt.plot(df_weighted_short_1_1['sma10'], label="10-Day SMA")
plt.xlabel("Date")
plt.ylabel("sales")
plt.legend()
plt.show()
XG_model = xgb.XGBRegressor(n_estimators=1000) 
X_test = X_test.drop (['ItemStoreCombined'],axis=1)
X_train = X_train.drop (['ItemStoreCombined'],axis=1)
X_test.head()
y_test.head()
y_test.sum()
%%time
XG_model.fit(X_train, y_train,eval_set=[(X_test, y_test)],early_stopping_rounds=50,verbose=False)

_ = plot_importance(XG_model, height=0.9)
XG_test_prediction = XG_model.predict(X_test)
XG_test_all =X_test.copy()
XG_train_all =X_train.copy()
XG_test_all['XG prediction'] = XG_model.predict(X_test)
XG_train_all['XG prediction'] =XG_model.predict(X_train)
XG_test_all['sales'] = y_test
XG_train_all['sales'] = y_train
df_xg_all = pd.concat([XG_test_all, XG_train_all], sort=False)
RMSE_XG_initial  = np.mean(np.sqrt((XG_test_all['XG prediction'] - XG_test_all['sales']) ** 2)) 
print(RMSE_XG_initial)
_ = df_xg_all[['sales','XG prediction']].plot(figsize=(15, 5))
# too many stores and products for graph to be useful apart form seeing the outliers
# when see the blue this is the outliers
# there are very few super low sales days
# there are plenty of days that are very high - which are good for business but hard to forecast
df_xg_all.sample(10)
XG_test_all.head()
XG_test_all['sales']
XG_test_all['XG prediction']
XG_test_all.head()
df_test_all_1_1 = XG_test_all[(XG_test_all.store==1)&(XG_test_all.item==1)]
_ = df_test_all_1_1[['sales','XG prediction']].plot(figsize=(15, 5))
df_test_all_2_1 = XG_test_all[(XG_test_all.store==2)&(XG_test_all.item==1)]
_ = df_test_all_2_1[['sales','XG prediction']].plot(figsize=(15, 5))
df_test_all_2_2 = XG_test_all[(XG_test_all.store==2)&(XG_test_all.item==2)]
_ = df_test_all_2_2[['sales','XG prediction']].plot(figsize=(15, 5))
XG_test_all.head()
# This calls the error calculating function
XG_Results= calculate_error(XG_test_all['sales'],XG_train_all['sales'],XG_test_all['XG prediction'],XG_train_all['XG prediction'])
print(XG_Results)
print(color.BOLD +"XG Boost Results ")
print ('\033[0m')

print("Mean Squared Error -MSE")
print("MSE_test",XG_Results['MSE_test'])
print("MSE_train",XG_Results['MSE_train'])
print(" ")
print("Mean Absolute Error - MAE")
print("MAE_test",XG_Results['MAE_test'])
print("MAE_train",XG_Results['MAE_train'])
print(" ")
print("Mean Absolute Percentage Error - MPE")
print("MAPE",XG_Results['MAPE'])
print(" ")
print("Root Mean Squared Error -RMSE")
print("RMSE",XG_Results['RMSE'])

XGaccuracy = accuracy_score(XG_test_all['sales'], XG_test_all['XG prediction'].round()) 
print("Accuracy: %.2f%%" % (XGaccuracy * 100.0))
# This accuracy score does not relfect the accuracy of the result. Instead I looked at the forecasts. I have therefore not used accuracy score further and instead used RMSE and others/
XG_test_all['error'] = XG_test_all['sales'] - XG_test_all['XG prediction']
XG_test_all['abs_error'] = XG_test_all['error'].apply(np.abs)
XG_test_all['abs_error_percent'] = (XG_test_all['abs_error'] / XG_test_all['sales'])*100
error_by_day = XG_test_all.groupby(['year','month','dayofmonth']).mean()[['sales','XG prediction','error','abs_error','store','item']]
error_by_day = XG_test_all.groupby(['year','month','dayofmonth']).mean()[['sales','XG prediction','error','abs_error','store','item']]
1
error_by_day.sort_values('error', ascending=True).head(5)
df_xg_all.head()
XG_test_all['error']

num_bins = 100
plt.title('XG by prod abs error percent')
plt.hist(XG_test_all['abs_error_percent'], bins =num_bins)
plt.xlim((0,50))
plt.show()
XG_test_all.head()
XG_test_all.abs_error_percent.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.995])
# used this information for presentation in pack to look at the accuracy of the model
XG_test_all.abs_error_percent.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.995])
# used this information for presentation in pack to look at the accuracy of the model
XG_test_predictions = XG_test_all.copy()
XG_test_predictions.head()
XG_test_predictions = XG_test_predictions.drop (['dayofweek', 'dayofmonth','quarter','month', 'year', 'dayofyear', 'weekofyear'],axis=1)
df_test_final.sum()
df_test_final = df_test_final.merge(XG_test_predictions, left_on=['date','store','item'], right_on=['date','store','item'] )
df_test_final.sample(10)
#CatBoostModel=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
CatBoostModel=CatBoostRegressor()
CatBoostModel.fit(X_train, y_train,eval_set=(X_test, y_test),plot=True)
catboostpred = CatBoostModel.predict(X_test)
print(catboostpred)
CAT_test_all =X_test.copy()
CAT_train_all =X_train.copy()
CAT_test_all['CAT prediction'] = CatBoostModel.predict(X_test)
CAT_train_all['CAT prediction'] =CatBoostModel.predict(X_train)
CAT_test_all['sales'] = y_test
CAT_train_all['sales'] = y_train
df_CAT_all = pd.concat([CAT_test_all, CAT_train_all], sort=False)

CAT_test_all.sum()
df_test_all_1_1 = CAT_test_all[(CAT_test_all.store==1)&(CAT_test_all.item==1)]
_=df_test_all_1_1[['sales','CAT prediction']].plot(figsize=(15, 5))
CAT_Results= calculate_error(CAT_test_all['sales'],CAT_train_all['sales'],CAT_test_all['CAT prediction'],CAT_train_all['CAT prediction'])
print(color.BOLD +"CAT Boost Results ")
print ('\033[0m')

print("Mean Squared Error -MSE")
print("MSE_test",CAT_Results['MSE_test'])
print("MSE_train",CAT_Results['MSE_train'])
print(" ")
print("Mean Absolute Error - MAE")
print("MAE_test",CAT_Results['MAE_test'])
print("MAE_train",CAT_Results['MAE_train'])
print(" ")
print("Mean Absolute Percentage Error - MPE")
print("MAPE",CAT_Results['MAPE'])
print(" ")
print("Root Mean Squared Error -RMSE")
print("RMSE",CAT_Results['RMSE'])
df_test_final = df_test_final.merge(CAT_test_all, left_on=['date','store','item'], right_on=['date','store','item'] )
df_test_final.head()
df_test_final.sum()
df_test_final.sample(5)
df_test_final_1_1= df_test_final[(df_test_final.store==1) & (df_test_final.item==1)]

#df_test_final_1_1= df_test_final_Auto[(df_test_final_Auto.store==1) & (df_test_final_Auto.item==1)]
df_test_final_1_1_Jan = df_test_final_1_1[(df_test_final.date<'2017-01-31')]
df_test_final_1_1.sample(3)
RMSE_1_1_XG  = np.mean(np.sqrt((df_test_final_1_1['XG prediction'] - df_test_final_1_1['sales']) ** 2)) 
print(RMSE_1_1_XG)
df_test_final_1_1_NovDec = df_test_final_1_1[(df_test_final.date>'2017-10-31')]
df_test_final_1_1_Jan.info()
df_test_final_1_1_Jan.head(2)
f, ax = plt.subplots(1)
f.set_figheight(5)
f.set_figwidth(10)
_ = df_test_final_1_1_Jan[['XG prediction', 'CAT prediction','sales_x']].plot(ax=ax, style=['-','-','.'])
ax.set_ylim(0, 50)
#ax.set_xbound(lower='12-12-2017', upper='31-12-2017')
plot = plt.suptitle('Jan 2017 sales and forecast for product 1 in store 1')
df_test_final_7days = df_test_final[(df_test_final.date>'2017-01-01')]
df_test_final_7days = df_test_final_7days[(df_test_final_7days.date<'2017-01-08')]
df_test_final_7days.head()
RMSE_7_days_Cat  = np.mean(np.sqrt((df_test_final_7days['CAT prediction'] - df_test_final_7days['sales_x']) ** 2)) 
print(RMSE_7_days_Cat)
df_test_final_new = df_test_final.copy()
df_test_final_new['date'] =  pd.to_datetime(df_test_final_new['date'])
df_test_final_new = df_test_final_new.set_index('date')
DailyFinal = df_test_final_new.resample('D').sum()
DailyFinal.head()
RMSE_daily_XG  = np.mean(np.sqrt((DailyFinal['XG prediction'] - DailyFinal['sales_x']) ** 2)) 
print(RMSE_daily_XG)
RMSE_daily_CAT  = np.mean(np.sqrt((DailyFinal['CAT prediction'] - DailyFinal['sales_x']) ** 2)) 
print(RMSE_daily_CAT)
MonthlyFinal = df_test_final_new.resample('M').sum()
MonthlyFinal.head()
MonthlyFinal.info()
RMSE_monthly_XG  = np.mean(np.sqrt((MonthlyFinal['XG prediction'] - MonthlyFinal['sales_x']) ** 2)) 
print(RMSE_monthly_XG)
RMSE_monthly_CAT  = np.mean(np.sqrt((MonthlyFinal['CAT prediction'] - MonthlyFinal['sales_x']) ** 2)) 
print(RMSE_monthly_CAT)
df_test_final.sample(10)
Store_Month_Test_Final = df_test_final_new.groupby(['store']).resample('M').sum()
Store_Month_Test_Final.tail(10)
RMSE_Store_Month_XG  = np.mean(np.sqrt((Store_Month_Test_Final['XG prediction'] - Store_Month_Test_Final['sales_x']) ** 2)) 
print(RMSE_Store_Month_XG)
RMSE_Store_Month_Cat  = np.mean(np.sqrt((Store_Month_Test_Final['CAT prediction'] - Store_Month_Test_Final['sales_x']) ** 2)) 
print(RMSE_Store_Month_Cat)
Store_Month_Test_Final.info()
Store_item_Month_Test_Final = df_test_final_new.groupby(['store','item']).resample('M').sum()
Store_item_Month_Test_Final.head()
RMSE_Store_item_XG  = np.mean(np.sqrt((Store_item_Month_Test_Final['XG prediction'] - Store_item_Month_Test_Final['sales_x']) ** 2)) 
print(RMSE_Store_item_XG)
RMSE_Store_item_Month_Cat  = np.mean(np.sqrt((Store_item_Month_Test_Final['CAT prediction'] - Store_item_Month_Test_Final['sales_x']) ** 2)) 
print(RMSE_Store_item_Month_Cat)
print(color.BOLD +"RMSE ")
print ('\033[0m')

print("Root Mean Squared Error -RMSE")
print("RMSE XG Boost",XG_Results['RMSE'])
print("RMSE",CAT_Results['RMSE'])
df_test_final.sum()
