import pandas as pd

# Read data

train_data=pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')
# Get mean/median sales value from store/DayOfWeek,Promo

mean_value=train_data[train_data['Open']==1][['Sales','Store','DayOfWeek','Promo']].groupby(['Store','DayOfWeek','Promo']).mean()

median_value=train_data[train_data['Open']==1][['Sales','Store','DayOfWeek','Promo']].groupby(['Store','DayOfWeek','Promo']).median()
def set_testValue(value_df,isOpen,storeId,dayOfWeek,isPromo):

    if isOpen==0:

        return 0

    else:

        return value_df.ix[storeId,dayOfWeek,isPromo]

# Based on the mean value

test_data['Sales_Prediction_Mean']=test_data.apply(lambda row: set_testValue(

        mean_value,row['Open'],row['Store'],row['DayOfWeek'],row['Promo']),axis=1)

# Based on the median value

test_data['Sales_Prediction_Median']=test_data.apply(lambda row: set_testValue(

        median_value,row['Open'],row['Store'],row['DayOfWeek'],row['Promo']),axis=1)
#Output data to csv

mean_output=test_data.ix[:,['Id','Sales_Prediction_Mean']].rename(columns={'Sales_Prediction_Mean': 'Sales'})

mean_output.to_csv('rossmann_sales_mean.csv',index=False)



median_output=test_data.ix[:,['Id','Sales_Prediction_Median']].rename(columns={'Sales_Prediction_Median': 'Sales'})

median_output.to_csv('rossmann_sales_median.csv',index=False)
