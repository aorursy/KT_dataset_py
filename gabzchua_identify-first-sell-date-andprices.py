import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Get launch date



prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

prices['id'] = prices['store_id'] + "_" + prices['item_id']

cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

cal = cal[['wm_yr_wk', 'date', 'd']]

cal['d'] = cal['d'].str.replace("d_", "").astype(int)

prices = prices.merge(cal[['wm_yr_wk', 'd']].groupby(['wm_yr_wk'])[['d']].min().reset_index(), on = 'wm_yr_wk')

prices = prices.sort_values(['id', 'd'])

launch_date = prices.groupby(['id'])[['d']].min().reset_index()

launch_date.head()



launch_date.to_csv("m5_launch_date.csv", index = False)
import gc

gc.collect()
# Get prices



prices = prices.drop(['d', 'store_id', 'item_id'], axis= 1)

cal = cal.drop('date', axis = 1)

prices = prices.merge(cal, on = 'wm_yr_wk')

prices = prices.drop(['wm_yr_wk'], axis = 1)

prices = prices.pivot(index = 'id', columns = 'd', values = 'sell_price')



# Remove this line for eventual test set

prices = prices.iloc[:,:-28]



prices.reset_index().to_csv("m5_prices_wide.csv", index = False)
cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")



# Weekends



cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

weekends = cal[['wday']]

weekends[(weekends['wday'] == 1) | (weekends['wday'] == 2) | (weekends['wday'] == 7)] = 1

weekends[(weekends['wday'] != 1) & (weekends['wday'] != 2) & (weekends['wday'] != 7)] = 0



weekends.columns = ['weekend']



# Remove for eventual set

weekends = weekends.iloc[:-28]



weekends.to_csv("m5_weekends.csv", index = False)
# Snap

sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv", usecols = [0,1,2,3,4,5])

snap_CA = cal['snap_CA'].values

snap_WI = cal['snap_WI'].values

snap_TX = cal['snap_TX'].values

snap_CA_m = np.repeat(snap_CA.reshape(-1,1), repeats = np.sum(sales['state_id'] == "CA"), axis = 1).transpose()

snap_TX_m = np.repeat(snap_TX.reshape(-1,1), repeats = np.sum(sales['state_id'] == "TX"), axis = 1).transpose()

snap_WI_m = np.repeat(snap_WI.reshape(-1,1), repeats = np.sum(sales['state_id'] == "WI"), axis = 1).transpose()

snap = np.concatenate([snap_CA_m, snap_TX_m, snap_WI_m], axis = 0)



snap = pd.DataFrame(snap)



# remove for final set

snap.iloc[:, :-28].to_csv("snap.csv", index = False)
# holidays

list_of_hols = list(set(np.append(cal['event_name_1'].unique(), cal['event_name_2'].unique())))[1:]



holidays = cal.loc[(cal['event_name_1'] == list_of_hols[0]) | (cal['event_name_2'] == list_of_hols[0]), ['date', 'event_name_1', 'date','event_name_2']]



for i in list_of_hols[1:]:

    holidays = holidays.append(cal.loc[(cal['event_name_1'] == i) | (cal['event_name_2'] == i), ['date', 'event_name_1', 'date','event_name_2']])

    

holidays_m = holidays.iloc[:,:2].rename(columns = {'event_name_1':'holiday'}).append(holidays.iloc[:,2:].rename(columns = {'event_name_2':'holiday'}))



holidays_m = holidays_m.dropna()



holidays_m = holidays_m[['holiday', 'date']]

holidays_m.columns = ['holiday', 'ds']

holidays_m['lower_window'] = 0

holidays_m['upper_window'] = 0



holidays_m.loc[holidays_m['holiday'] == "Thanksgiving",'lower_window'] = -1

holidays_m.loc[holidays_m['holiday'] == "Thanksgiving",'upper_window'] = 1



holidays_m.loc[holidays_m['holiday'] == "Christmas",'lower_window'] = -1

holidays_m.loc[holidays_m['holiday'] == "Christmas",'upper_window'] = 1



holidays_m.loc[holidays_m['holiday'] == "Easter",'lower_window'] = -2



holidays_m.loc[holidays_m['holiday'] == "Eid al-Fitr",'lower_window'] = -1

holidays_m.loc[holidays_m['holiday'] == "Eid al-Fitr",'upper_window'] = 1





holidays_m['ds'] = pd.to_datetime(holidays_m['ds'])



holidays_m.to_csv("m5_holidays.csv", index = False)
sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

sale_hist = sales.iloc[:,6:]

cumulative_max = sale_hist.transpose().cummax().transpose()

extra = pd.DataFrame(np.repeat(cumulative_max.iloc[:,-1].values.reshape(-1,1), repeats = 28, axis = 1))

extra.columns = extra.columns + 1914

cumulative_max = pd.concat([cumulative_max, extra], axis = 1)

cumulative_max.to_csv("cumulative_max.csv", index = False)
sale_hist = sales.iloc[:,6:]

sale_hist[sale_hist != 0] = 2
sale_hist[sale_hist == 0] = 1

sale_hist[sale_hist == 2] = 0
cum_7_freq_zero = sale_hist.transpose().rolling(7).mean().transpose()

cum_14_freq_zero = sale_hist.transpose().rolling(14).mean().transpose()

cum_28_freq_zero = sale_hist.transpose().rolling(28).mean().transpose()

cum_56_freq_zero = sale_hist.transpose().rolling(56).mean().transpose()
extra = pd.DataFrame(np.repeat(cum_7_freq_zero.iloc[:,-1].values.reshape(-1,1), repeats = 28, axis = 1))

extra.columns = extra.columns + 1914

cum_7_freq_zero = pd.concat([cum_7_freq_zero, extra], axis = 1)



extra = pd.DataFrame(np.repeat(cum_14_freq_zero.iloc[:,-1].values.reshape(-1,1), repeats = 28, axis = 1))

extra.columns = extra.columns + 1914

cum_14_freq_zero = pd.concat([cum_14_freq_zero, extra], axis = 1)



extra = pd.DataFrame(np.repeat(cum_28_freq_zero.iloc[:,-1].values.reshape(-1,1), repeats = 28, axis = 1))

extra.columns = extra.columns + 1914

cum_28_freq_zero = pd.concat([cum_28_freq_zero, extra], axis = 1)



extra = pd.DataFrame(np.repeat(cum_56_freq_zero.iloc[:,-1].values.reshape(-1,1), repeats = 28, axis = 1))

extra.columns = extra.columns + 1914

cum_56_freq_zero = pd.concat([cum_56_freq_zero, extra], axis = 1)
cum_7_freq_zero.fillna(0).to_csv("cum_7_freq_zero.csv", index = False)

cum_14_freq_zero.fillna(0).to_csv("cum_14_freq_zero.csv", index = False)

cum_28_freq_zero.fillna(0).to_csv("cum_28_freq_zero.csv", index = False)

cum_56_freq_zero.fillna(0).to_csv("cum_56_freq_zero.csv", index = False)
cum_zero = sale_hist.transpose().cumsum().transpose()
history_zeros = (cum_zero.values - np.repeat(launch_date['d'].values.reshape(-1,1) - 1, repeats = 1913, axis = 1)) - 1

history_zeros[history_zeros < 0] = 0

cumulative_zero = pd.DataFrame(np.concatenate([history_zeros, np.repeat(history_zeros[:,-1].reshape(-1,1), 28, axis = 1)], axis = 1))

cumulative_zero.to_csv("cumulative_zero.csv", index = False)
df = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

df2 = pd.DataFrame(np.zeros([30490, 1913])).astype(int)



df3 = df.drop(['id'], axis = 1)

df3.columns = list(np.arange(1914, 1914+28))

df2.columns = df2.columns + 1

df_sample = pd.concat([df[['id']].head(30490), df2.astype(int), df3.head(30490)], axis = 1)

df_sample.to_csv("holder.csv", index = False)