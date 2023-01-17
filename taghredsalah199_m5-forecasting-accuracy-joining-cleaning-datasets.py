import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
forcast_data_calendar= pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

forcast_data_calendar
forcast_sales_eval= pd.read_csv('../input/m5-forecasting-accuracy/sales_train_evaluation.csv')

forcast_sales_eval

#We take this data frame about evaluation
forcast_sales_val= pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

forcast_sales_val

#But the validation under evaluation
forcast_sample_sumb= pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

forcast_sample_sumb

#The submission beside the eval and valid
forcast_sell_prices= pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

forcast_sell_prices

#join the sell price by item_id
result4 = pd.concat([forcast_sales_eval,forcast_sales_val],join='outer')

result4
result5 = pd.merge(result4,forcast_sample_sumb,how='outer',on='id')

result5

result5.to_csv('FEvalVal_concat.csv')

result4= pd.read_csv('FEvalVal_concat.csv')
forcast_sell_prices2=forcast_sell_prices.drop_duplicates(subset='item_id')

forcast_sell_prices2
result5 = pd.merge(result4,forcast_sell_prices2, how='outer',on='item_id')

result5
result5.to_csv('finall.csv')
