import pandas as pd
import numpy as np
import csv
import datetime
from multiprocessing import Pool
# SLA Matrix
SLA = {"manila": {"manila" : 3, "luzon" : 5,"visayas" : 7,"mindanao" : 7}, 
      "luzon": {"manila" : 5, "luzon" : 5, "visayas" : 7 ,"mindanao" : 7},
      "visayas": {"manila" : 7, "luzon" : 7, "visayas" : 7, "mindanao" : 7},
      "mindanao": {"manila" : 7, "luzon" : 7, "visayas" : 7, "mindanao" : 7}}

# Key in public holidays
phs = [datetime.datetime(2020, 3, 25).date(),datetime.datetime(2020, 3, 30).date(),
       datetime.datetime(2020, 3, 31).date()]
phs
data_df = pd.read_csv('../input/logistics-shopee-code-league/delivery_orders_march.csv', 
                      dtype = {'orderedid': 'int64','buyeraddress': 'str','selleraddress': 'str'}) 
# Convert delivery time into dates
parse_dates = ['pick','1st_deliver_attempt','2nd_deliver_attempt']
for col in parse_dates:
    data_df[col] = pd.to_datetime(data_df[col],unit='s')
    data_df[col] = data_df[col].dt.date
    
data_df
# Helper function to extract SLA time
def get_sla(row):
    origin = row['buy']
    dest = row['sell']
    return SLA[origin][dest]

# Get buyer address
buy_df = data_df['buyeraddress'].str.split(' ').str[-1]
buy_df = buy_df.str.lower()
data_df['buy'] = buy_df

# Get seller address
sell_df = data_df['selleraddress'].str.split(' ').str[-1]
sell_df = sell_df.str.lower()
data_df['sell'] = sell_df

# Get SLA
data_df['sla'] = data_df.apply(get_sla, axis=1)
final_df = data_df.drop(['buyeraddress','selleraddress'], axis=1)
final_df
# Parallelize dataframe for faster processing
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

# Generate answers
def process(df):
    ans = []
    for index, row in df.iterrows():
        if index % 100000 == 0:
            print("Processing " + str(index))
        late = 0
        first = (row['1st_deliver_attempt'] - row['pick']).days
        for ph in phs: # Subtract PHs
            if row['pick'] <= ph <= row['1st_deliver_attempt']:
                first -= 1
                
        # Subtract sundays
        first -= np.busday_count(row['pick'], row['1st_deliver_attempt'], weekmask='Sun')

        # Check if 1st delivery is late
        if first > row['sla']:
            late = 1

        # Check if 2nd delivery is late
        if not pd.isnull(row['2nd_deliver_attempt']) and late == 0: # Check if got 2nd delivery and 1st delivery not late
            second = (row['2nd_deliver_attempt'] - row['1st_deliver_attempt']).days
            for ph in phs:
                if row['1st_deliver_attempt'] <= ph <= row['2nd_deliver_attempt']:
                    second -= 1

            second -= np.busday_count(row['1st_deliver_attempt'], row['2nd_deliver_attempt'], weekmask='Sun')
            if second > 3:
                late = 1
                
        ans.append((row['orderid'],late))
        
    sub = pd.DataFrame.from_records(ans, columns =['orderid', 'is_late']) 
    return sub
# Generate submission file
sub_df = parallelize_dataframe(final_df, process, n_cores=4)
sub_df.to_csv('submission.csv',index=False)
sub_df