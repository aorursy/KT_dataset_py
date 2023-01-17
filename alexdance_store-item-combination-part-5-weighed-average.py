import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error , mean_absolute_error
class color:  # Testing to make the heading look a liitle more impressive
   BOLD = '\033[1m'
df = pd.read_csv("../input/demand-forecasting-kernels-only/train.csv")
df.head()
split = "2017-01-01"
df['date'] =  pd.to_datetime(df['date'])
def calculate_errorb(test_sales,  test_prediction):
    MSE_test = mean_squared_error(y_true=test_sales,  y_pred=test_prediction) # Mean Square Error (MAE)
    MAE_test = mean_absolute_error(y_true=test_sales,  y_pred=test_prediction) # Mean Absolute Error (MAE)
    MAPE = np.mean(np.abs(test_prediction - test_sales  ) **2)  # Mean Absolute Percentage Error (MAPE)
    RMSE  = np.mean(np.sqrt((test_prediction - test_sales) ** 2))    
    return{'MSE_test': MSE_test ,'MAE_test': MAE_test,  'MAPE':MAPE, 'RMSE':RMSE}
split = "2016-12-31"
df['ItemStoreCombined'] = df['item'].map(str) + '-' + df['store'].map(str) 
df.head()
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear
df = df.set_index('date')

df.head()
df_train = df[ :split ] 
df_test = df[split : ] 
df_train.head()
df_test_final = df_test.copy()
df_test_final =df_test_final.drop (['dayofweek', 'quarter','month', 'year', 'dayofyear', 'weekofyear'],axis=1)
train_cols=list(df_train.columns)
print(train_cols)
df_train = df_train.loc[:,train_cols] 
test_cols=list(df_test.columns)
df_test = df_test.loc[:,test_cols] 
df_train['Calculated_year'] =  df_train.index.year - min(df_train.index.year) + 1
df_train.head()
df_train['Calculated_year'].value_counts()
month_weighting= (( df_train.groupby(['month']).agg([np.nanmean]).sales - np.nanmean(df_train.sales) ) / np.nanmean(df_train.sales)).rename(columns={'nanmean':'month_weighting'})
df_train=df_train.join(month_weighting,how='left',on='month')
df_train.tail()
month_weighting.head(13)
year_weighting= (( df_train.groupby(['year']).agg([np.nanmean]).sales - np.nanmean(df_train.sales) ) /  np.nanmean(df_train.sales)).rename(columns={'nanmean':'year_weighting'})
print(year_weighting)
CAGR = 0.096 #only for using on the train data - can be adjusted
year_weighting.loc[6,:] =  np.mean(CAGR)*3
df_train=df_train.join(year_weighting,how='left',on='year')
weekday_weighting= ( ( df_train.groupby(['dayofweek']).agg([np.nanmean]).sales - np.nanmean(df_train.sales) ) /  np.nanmean(df_train.sales)).rename(columns={'nanmean':'weekday_weighting'})
df_train=df_train.join(weekday_weighting,how='left',on='dayofweek')
store_item_weighting= ( ( df_train.groupby(['store','item']).agg([np.nanmean]).sales - np.nanmean(df_train.sales) ) / np.nanmean(df_train.sales)).rename(columns={'nanmean':'store_item_weighting'})
df_train=df_train.join(store_item_weighting,how='left',on=['store','item'])
df_train['product_combined_weighting']=np.product(df_train.loc[:,['month_weighting','year_weighting','weekday_weighting','store_item_weighting',]]+1,axis=1)
df_train.sample()
df_train.tail()
df_train.Calculated_year.nunique()
df_train.Calculated_year.value_counts()
print(weekday_weighting)
print(month_weighting)
print(store_item_weighting)
df_train.head()
df_train['sales_prediction']=np.round(df_train.product_combined_weighting*np.round(np.nanmean(df_train.sales),1))  
average_train_sales = np.nanmean(df_train.sales)
print(average_train_sales)
df_train.head()
df_test=df_test.join(month_weighting,how='left',on='month')
df_test['Calculated_year'] =  5
year_weighting_17 =0.22  # calculated seperately
df_test['year_weighting'] = year_weighting_17
df_test=df_test.join(weekday_weighting,how='left',on='dayofweek')
df_test=df_test.join(store_item_weighting,how='left',on=['store','item'])
df_test.head()
df_test['smry_product']=np.product(df_test.loc[:,['month_weighting','year_weighting','weekday_weighting','store_item_weighting',]]+1,axis=1)
df_test['weighted_sales_prediction']=df_test.smry_product*average_train_sales
average_train_sales
df_test.sum()
df_test.head()
RMSE_weighted  = np.mean(np.sqrt((df_test['weighted_sales_prediction'] - df_test['sales']) ** 2)) 
print(RMSE_weighted)