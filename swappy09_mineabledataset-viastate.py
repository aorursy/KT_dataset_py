# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
calender = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
sell_price = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
sales_train_validation = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
#Checking the shape of each data files

print("Calender: ", calender.shape)
print("Sell Price: ", sell_price.shape)
print("Sample Submission: ", sample_submission.shape)
print("Sales Train Validation: ", sales_train_validation.shape)
#Bringing days column into rows - Col -> Row Transformation

def TransposeDf(df):
    df_transpose = pd.melt(df, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])
    df_transpose = df_transpose.rename(columns={"variable":"d", "value":"qty"})
    #df_transpose = df_transpose.sort_values(by=['id','days'])
    return df_transpose
    
#CA

CA = TransposeDf(sales_train_validation[sales_train_validation.state_id=='CA'])

#Making the data type constant
CA['d']=CA['d'].astype(str)
calender['d']=calender['d'].astype(str)

#erging it with Calender and Sell Price
CA = CA.merge(calender, on=['d'], how='left')
CA = CA.merge(sell_price, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

#Checking the shape and saving the CSV
print("CA: ", CA.shape)
CA.to_csv("CA.csv", index=False)
print("CSV has been created for CA")

#Freeing up the memory
del [[CA]]
gc.collect()

#WI

WI = TransposeDf(sales_train_validation[sales_train_validation.state_id=='WI'])

#Making the data type constant
WI['d']=WI['d'].astype(str)
calender['d']=calender['d'].astype(str)

#erging it with Calender and Sell Price
WI = WI.merge(calender, on=['d'], how='left')
WI = WI.merge(sell_price, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

#Checking the shape and saving the CSV
print("WI: ", WI.shape)
WI.to_csv("WI.csv", index=False)
print("CSV has been created for WI")


#Freeing up the memory
del WI
gc.collect()
#TX

TX = TransposeDf(sales_train_validation[sales_train_validation.state_id=='TX'])

#Making the data type constant
TX['d']=TX['d'].astype(str)
calender['d']=calender['d'].astype(str)

#erging it with Calender and Sell Price
TX = TX.merge(calender, on=['d'], how='left')
TX = TX.merge(sell_price, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

#Checking the shape and saving the CSV
print("TX: ", TX.shape)
TX.to_csv("TX.csv", index=False)
print("CSV has been created for TX")


#Freeing up the memory
del TX
gc.collect()