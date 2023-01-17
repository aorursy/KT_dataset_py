import pandas as pd
import numpy as np
import datetime as dt
# Display the whole contents of the columns. Addresses are too long.
pd.set_option('display.max_colwidth', None)
order_df = pd.read_csv('/kaggle/input/open-shopee-code-league-logistic/delivery_orders_march.csv')
order_df
order_df.info()
order_df.isnull().sum()
# Convert epoch time to normal datetime format.

order_df['pickup'] = order_df.pick.apply(lambda x: dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
order_df['first_attempt'] = order_df['1st_deliver_attempt'].apply(lambda x: dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
order_df['second_attempt'] = order_df['2nd_deliver_attempt'].apply(lambda x: dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d') if (x > 0) else 0)
# drop all the columns with epoch time.
order_df.drop(['pick', '1st_deliver_attempt', '2nd_deliver_attempt'], axis = 1, inplace = True)
order_df.head()
#Convert addresses to lower case.

order_df.buyeraddress = order_df['buyeraddress'].apply(str.lower)
order_df.selleraddress = order_df['selleraddress'].apply(str.lower)
from tqdm import tqdm
# Create Origin column from selleraddress column.
# Use [-12:] because the location is at the end of the address. 'metro manila' is the longest among the 4 containing 12 char.
 
origin_list = []

for ind, row in tqdm(order_df.iterrows()):

    if 'metro manila' in row['selleraddress'][-12:]:
        place = 'metro manila'
    elif 'luzon' in row['selleraddress'][-12:]:
        place = 'luzon'        
    elif 'visayas' in row['selleraddress'][-12:]:
        place = 'visayas'
    elif 'mindanao' in row['selleraddress'][-12:]:
        place = 'mindanao'
 
    origin_list.append(place)
    
order_df['Origin'] = origin_list  
# Create Destination column from buyeraddress column.
# Use [-12:] because the location is at the end of the address. 'metro manila' is the longest among the 4 containing 12 char.

 
destination_list = []

for ind, row in tqdm(order_df.iterrows()):
  
    if 'metro manila' in row['buyeraddress'][-12:]:
        place = 'metro manila'
    elif 'luzon' in row['buyeraddress'][-12:]:
        place = 'luzon'        
    elif 'visayas' in row['buyeraddress'][-12:]:
        place = 'visayas'
    elif 'mindanao' in row['buyeraddress'][-12:]:
        place = 'mindanao'

        
    destination_list.append(place)
    
order_df['Destination'] = destination_list    
# Cheeck to see if everything is accounted for
print('Unique values in Origin:', order_df.Origin.unique())
print('Unique values in Destiantion:', order_df.Destination.unique())
print('No of rows in Origin:', len(order_df[order_df.Origin.str.contains('mindanao')]) + len(order_df[order_df.Origin.str.contains('visayas')]) + len(order_df[order_df.Origin.str.contains('luzon')]) + len(order_df[order_df.Origin.str.contains('metro manila')]))
print('No of rows in Destination:', len(order_df[order_df.Destination.str.contains('mindanao')]) + len(order_df[order_df.Destination.str.contains('visayas')]) + len(order_df[order_df.Destination.str.contains('luzon')]) + len(order_df[order_df.Destination.str.contains('metro manila')]))

# Comparing length of Origin and Destination, all the 4 locations are accounted for. 3,176,313 rows total.
order_df.drop(['buyeraddress', 'selleraddress'], axis = 1, inplace = True)
order_df
holiday = ['2020-03-08', '2020-03-25', '2020-03-30', '2020-03-31']
# Calculate days until first delivery attempt.
order_df['first_attempt_days'] = np.busday_count(order_df['pickup'], order_df['first_attempt'], weekmask = '1111110', holidays = holiday)
# Calculate days until second delivery attemmpt.
order_df['second_attempt_days'] = np.busday_count(order_df['first_attempt'], order_df['second_attempt'], weekmask = '1111110', holidays = holiday)
# set to 0 where there are no second attempt.
order_df.loc[order_df['second_attempt_days'] < 0, 'second_attempt_days'] = 0
order_df
# check order_df for late deliveries
islate_list = []

for ind, row in tqdm(order_df.iterrows()):
    
    late = 0
    
    if row['second_attempt_days'] > 3:
        late = 1
  
    elif row['Origin'] == 'metro manila':
        if row['Destination'] == 'metro manila':
            if row['first_attempt_days'] > 3:
                late = 1
        elif row['Destination'] == 'luzon':
            if row['first_attempt_days'] > 5:
                late = 1
        elif row['Destination'] == 'visayas':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'mindanao':
            if row['first_attempt_days'] > 7:
                late = 1
                
    elif row['Origin'] == 'luzon':
        if row['Destination'] == 'metro manila':
            if row['first_attempt_days'] > 5:
                late = 1
        elif row['Destination'] == 'luzon':
            if row['first_attempt_days'] > 5:
                late = 1
        elif row['Destination'] == 'visayas':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'mindanao':
            if row['first_attempt_days'] > 7:
                late = 1
                
    elif row['Origin'] == 'visayas':
        if row['Destination'] == 'metro manila':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'luzon':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'visayas':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'mindanao':
            if row['first_attempt_days'] > 7:
                late = 1
                
    elif row['Origin'] == 'mindanao':
        if row['Destination'] == 'metro manila':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'luzon':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'visayas':
            if row['first_attempt_days'] > 7:
                late = 1
        elif row['Destination'] == 'mindanao':
            if row['first_attempt_days'] > 7:
                late = 1
   
    islate_list.append(late)
    
order_df['islate'] = islate_list
print('late order:', len(order_df[order_df.islate == 1]))
print('on-time order:', len(order_df[order_df.islate == 0]))

print('Total:', len(order_df[order_df.islate == 1]) + len(order_df[order_df.islate == 0]))
results = order_df[['orderid', 'islate']]
results