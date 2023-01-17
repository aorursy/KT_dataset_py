import numpy as np # linear algebra
import glob
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs and figures
fields = ['TIME_M', 'EX', 'BID', 'BIDSIZ', 'ASK', 'ASKSIZ', 'SYM_ROOT', 'QU_COND']
path =r'../input/'
allFiles = glob.glob(path + "quotes*.csv")

list_ = []

for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0, usecols=fields)
    list_.append(df)
frame = pd.concat(list_, axis = 0, ignore_index = True)
frame['VAL'] = 1 # Create temporary variable for data testing

frame['TIME_M'] = pd.to_timedelta(frame['TIME_M']) # Set time format for a column  with time values

# Determine invalid and suspicious data. Apply checks for trading time, zero quotes, regular quotes, and outliers
frame.loc[(frame['TIME_M']<'9:30:00') | (frame['TIME_M']>'16:00:00') | (frame['ASK']==0) | (frame['BIDSIZ']==0) | 
          (frame['ASKSIZ']==0) | (frame['QU_COND']!='R') | (frame['ASK']>2.5*10**5), 'VAL'] = 0

# Adjust quote size
frame['BIDSIZ'] = 100*frame['BIDSIZ'] 
frame['ASKSIZ'] = 100*frame['ASKSIZ']
frame.drop(frame[frame.VAL!=1].index, inplace=True)
frame.drop('QU_COND', axis=1, inplace=True)
frame.drop('VAL', axis=1, inplace=True)
frame['SPREAD'] = frame['ASK'] - frame['BID']
spread_sum = frame.groupby('SYM_ROOT')['SPREAD'].sum()
bid_sum = frame.groupby('SYM_ROOT')['BID'].sum()
bidsiz_sum = frame.groupby('SYM_ROOT')['BIDSIZ'].sum()
ask_sum = frame.groupby('SYM_ROOT')['ASK'].sum()
asksiz_sum = frame.groupby('SYM_ROOT')['ASKSIZ'].sum()
spread_mean = frame.groupby('SYM_ROOT')['SPREAD'].mean()
bid_mean = frame.groupby('SYM_ROOT')['BID'].mean()
bidsiz_mean = frame.groupby('SYM_ROOT')['BIDSIZ'].mean()
ask_mean = frame.groupby('SYM_ROOT')['ASK'].mean()
asksiz_mean = frame.groupby('SYM_ROOT')['ASKSIZ'].mean()
spread_std = frame.groupby('SYM_ROOT')['SPREAD'].std()
bid_std = frame.groupby('SYM_ROOT')['BID'].std()
bidsiz_std = frame.groupby('SYM_ROOT')['BIDSIZ'].std()
ask_std = frame.groupby('SYM_ROOT')['ASK'].std()
asksiz_std = frame.groupby('SYM_ROOT')['ASKSIZ'].std()
stat_table = pd.concat([spread_sum, bid_sum, bidsiz_sum, ask_sum, asksiz_sum, spread_mean, bid_mean, bidsiz_mean, 
                    ask_mean, asksiz_mean,spread_std, bid_std, bidsiz_std, ask_std, asksiz_std], axis=1)
stat_table.columns = ['spread_sum', 'bid_sum', 'bidsiz_sum', 'ask_sum', 'asksiz_sum', 'spread_mean', 'bid_mean', 'bidsiz_mean', 
                  'ask_mean', 'asksiz_mean', 'spread_std', 'bid_std', 'bidsiz_std', 'ask_std', 'asksiz_std']
# Load data for 'A' tickers only 
frame_2 = pd.read_csv('../input/trades-20150226.csv', nrows=3033150)
# Rename columns for later merger
frame_2.rename(columns={'sym_root':'SYM_ROOT'}, inplace=True) 
frame_2.rename(columns={'exchange':'EX'}, inplace=True)

frame_2['size'] = 100*frame_2['size'] # Adjust trade size

# Create a variable for dollar volume of trades
frame_2['volume']=frame_2['size']*frame_2['price'] 
trade_size = frame_2.groupby(['SYM_ROOT'])['size'].sum()

table_1 = stat_table.join(trade_size, how='left', on=None)

table_1.fillna(0, inplace=True)

table_1.to_csv('table_1.csv')
exvolume_sum = frame_2.groupby(['EX', 'SYM_ROOT'])['volume'].sum()
exvolume_std = frame_2.groupby(['EX', 'SYM_ROOT'])['volume'].std()

exspread_sum = frame.groupby(['EX','SYM_ROOT'])['SPREAD'].sum()
exbid_sum = frame.groupby(['EX','SYM_ROOT'])['BID'].sum()
exbidsiz_sum = frame.groupby(['EX','SYM_ROOT'])['BIDSIZ'].sum()
exask_sum = frame.groupby(['EX','SYM_ROOT'])['ASK'].sum()
exasksiz_sum = frame.groupby(['EX','SYM_ROOT'])['ASKSIZ'].sum()
exspread_mean = frame.groupby(['EX','SYM_ROOT'])['SPREAD'].mean()
exbid_mean = frame.groupby(['EX','SYM_ROOT'])['BID'].mean()
exbidsiz_mean = frame.groupby(['EX','SYM_ROOT'])['BIDSIZ'].mean()
exask_mean = frame.groupby(['EX','SYM_ROOT'])['ASK'].mean()
exasksiz_mean = frame.groupby(['EX','SYM_ROOT'])['ASKSIZ'].mean()
exspread_std = frame.groupby(['EX','SYM_ROOT'])['SPREAD'].std()
exbid_std = frame.groupby(['EX','SYM_ROOT'])['BID'].std()
exbidsiz_std = frame.groupby(['EX','SYM_ROOT'])['BIDSIZ'].std()
exask_std = frame.groupby(['EX','SYM_ROOT'])['ASK'].std()
exasksiz_std = frame.groupby(['EX','SYM_ROOT'])['ASKSIZ'].std()
trade_volume = pd.concat([exvolume_sum, exvolume_std], axis=1)
trade_volume.columns = ['volume','volume_volatility']

stat_table_2 = pd.concat([exspread_sum, exbid_sum, exbidsiz_sum, exask_sum, exasksiz_sum, exspread_mean, exbid_mean, exbidsiz_mean, 
                    exask_mean, exasksiz_mean, exspread_std, exbid_std, exbidsiz_std, exask_std, exasksiz_std], axis=1)
stat_table_2.columns = ['spread_sum', 'bid_sum', 'bidsiz_sum', 'ask_sum', 'asksiz_sum', 'spread_mean', 'bid_mean', 'bidsiz_mean', 
                  'ask_mean', 'asksiz_mean', 'spread_std', 'bid_std', 'bidsiz_std', 'ask_std', 'asksiz_std']

table_2 = stat_table_2.join(trade_volume, on=['EX','SYM_ROOT'])
table_2.fillna(0, inplace=True)

table_2.to_csv('table_2.csv')
table_1
table_2
#frame.VAL.value_counts()

#plt.hist(frame.BID, bins=[0,5,10,15,20,25,30,35,40,45,50,55,60,65], color = 'green', edgecolor = 'black')

#frame.ASK.hist()

#type(frame['ASK'].quantile())

#frame.isnull().values.any()

#frame.loc[frame['EX'] == 'Z','SYM_ROOT'].unique()