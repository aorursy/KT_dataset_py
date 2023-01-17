import numpy as np
import pandas as pd
from kaggle.competitions import twosigmanews
#   You  can  only    call    make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
mt_df, news_df = env.get_training_data()
import matplotlib.pyplot as plt
import seaborn  as sns
# plotAsset plots assetCode1 from date1 to date2
def plotAsset(assetCode1, date1, date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = asset_df['close'].values

#    plt.figure(figsize=(10,6))
    plt.figure(figsize=(8,5))
    plt.title(assetCode1+": Opening and closing price")
    plt.plot(asset_df.time, asset_df.open, label='Open price')
    plt.plot(asset_df.time, asset_df.close, label='Close price')
    plt.legend()
    plt.show()

import matplotlib.dates as mdates

# plotAssetGrid plots assetCode1 from date1 to date2 as a subplot
def plotAssetGrid(assetCode1, date1, date2, ax):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    center = int(asset_df.shape[0]/2)
    ts = asset_df.iloc[center]["time"]
    ts_str = pd.to_datetime(str(ts)) 
    d_str = ts_str.strftime('%Y.%b.%d')
    #date_str = str(ts.year)+"/"+str(ts.month)+"/"+str(ts.day)
    #y_str = str(asset_df.iloc[center]["time"].year)
    #m1_str = str(asset_df.iloc[center]["time"].month)
    #d1_str = str(asset_df.iloc[center]["time"].day)
    ax.set_title(assetCode1+": "+d_str)
    ax.plot(asset_df.time, asset_df.open, label='Open price')
    ax.plot(asset_df.time, asset_df.close, label='Close price')
    myFmt = mdates.DateFormatter('%d/%b')
    ax.xaxis.set_major_formatter(myFmt)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    #ax.tick_params(labelrotation=45)
    #ax.get_xticklabels().set_rotation(45)
from IPython.display import display

def printAssetDf(assetCode1, date1, date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]

    display(asset_df)
def graph_outliers(show_data=False, show_plots=True):
    offset = pd.Timedelta(7, unit='d')
    for index, row in outliers.iterrows():
        print (row["time"], row['assetCode'])
        start = row["time"]-offset
        end = row["time"]+offset
        print(str(start) + " -> " + str(end))

        if show_data:
            printAssetDf(row['assetCode'], row["time"]-offset, row["time"]+offset)
        if show_plots:
            plotAsset(row['assetCode'], row["time"]-offset, row["time"]+offset)
import math

def graph_outliers_grid(show_data=True, show_plots=True, print_assets=5):
    count = len(outliers)
    cols = 5
    rows = int(math.ceil(count/cols))
    
    if show_plots:
        offset = pd.Timedelta(7, unit='d')
        fig, axes = plt.subplots(ncols=cols, nrows=rows,figsize=(15,15))
        #fig.autofmt_xdate(rotation=25)
        axes = [col for row in axes for col in row]

        idx = 0
        for index, row in outliers.iterrows():
            plotAssetGrid(row['assetCode'], row["time"]-offset, row["time"]+offset, axes[idx])
            idx += 1
        fig.tight_layout()
        plt.show()

    if show_data:
        offset = pd.Timedelta(5, unit='d')
        count = 0
        for index, row in outliers.iterrows():
            print (row["time"], row['assetCode'])
            start = row["time"]-offset
            end = row["time"]+offset
            print(str(start) + " -> " + str(end))
            printAssetDf(row['assetCode'], row["time"]-offset, row["time"]+offset)
            count += 1
            if count >= print_assets:
                break
        
#mask = np.abs(market_df['returnsClosePrevRaw1']-market_df['returnsClosePrevRaw1'].mean()) > (20*market_df['returnsClosePrevRaw1'].std())
mask = np.abs(mt_df['returnsClosePrevRaw1']-mt_df['returnsClosePrevRaw1'].mean()) > 1
outliers = mt_df.loc[mask]
outliers
graph_outliers_grid(True, True)
outlier_collection = mt_df.head(0)
outlier_collection = outlier_collection.append(mt_df.iloc[3845015])
outlier_collection = outlier_collection.append(mt_df.iloc[3845309])
outlier_collection = outlier_collection.append(mt_df.iloc[3845467])
outlier_collection = outlier_collection.append(mt_df.iloc[3845835])
outlier_collection = outlier_collection.append(mt_df.iloc[3846067])
outlier_collection = outlier_collection.append(mt_df.iloc[3846276])
outlier_collection = outlier_collection.append(mt_df.iloc[3846636])
outlier_collection
mt_df['price_diff'] = mt_df['close'] - mt_df['open']
outliers = mt_df.sort_values('price_diff', ascending=False)[:30]
graph_outliers_grid(True, True)
outlier_collection = outlier_collection.append(mt_df.iloc[50031])
outlier_collection = outlier_collection.append(mt_df.iloc[92477])
outlier_collection = outlier_collection.append(mt_df.iloc[206676])
outlier_collection = outlier_collection.append(mt_df.iloc[459234])
outlier_collection = outlier_collection.append(mt_df.iloc[132779])
outlier_collection = outlier_collection.append(mt_df.iloc[50374])
outlier_collection = outlier_collection.append(mt_df.iloc[276388])
outlier_collection = outlier_collection.append(mt_df.iloc[3845946])
outlier_collection = outlier_collection.append(mt_df.iloc[616236])
outlier_collection = outlier_collection.append(mt_df.iloc[3846151])
outlier_collection = outlier_collection.append(mt_df.iloc[49062])

#outlier_collection
outliers = mt_df.sort_values('returnsOpenPrevRaw1', ascending=False)[:30]
graph_outliers_grid(True, True)
outlier_collection = outlier_collection.append(mt_df.iloc[588960])
outlier_collection = outlier_collection.append(mt_df.iloc[165718])
outlier_collection = outlier_collection.append(mt_df.iloc[25574])
outlier_collection = outlier_collection.append(mt_df.iloc[555738])
outlier_collection = outlier_collection.append(mt_df.iloc[56387])
#TW.N seems to have started trading on this day since there is no previous day
#so maybe the beginning spike is a high initial listing price?
#https://www.sec.gov/Archives/edgar/data/1470215/000119312510212218/d424b1.htm:
#"Towers Watson was formed on January 1, 2010, from the merger of Towers Perrin and Watson Wyatt"
outlier_collection = outlier_collection.append(mt_df.iloc[1127598])

#outlier_collection
#outliers = mt_df.sort_values('returnsClosePrevRaw1', ascending=False)[:30]
#graph_outliers_grid(True, True)
#outlier_collection
outliers = mt_df.sort_values('returnsClosePrevMktres1', ascending=False)[:30]
graph_outliers_grid(True, True)
#outlier_collection
outliers = mt_df.sort_values('returnsOpenPrevMktres1', ascending=False)[:30]
graph_outliers_grid(True, True, 5)
#graph_outliers_grid(True, True, 30)
#ABV strange values:
abv_collection = outlier_collection.head(0)
abv_collection = abv_collection.append(mt_df.iloc[611442])
abv_collection = abv_collection.append(mt_df.iloc[613039])
abv_collection = abv_collection.append(mt_df.iloc[614639])
abv_collection = abv_collection.append(mt_df.iloc[616236])
abv_collection = abv_collection.append(mt_df.iloc[617832])
abv_collection = abv_collection.append(mt_df.iloc[619423])
abv_collection = abv_collection.append(mt_df.iloc[621008])
abv_collection = abv_collection.append(mt_df.iloc[622591])
abv_collection = abv_collection.append(mt_df.iloc[624178])
abv_collection = abv_collection.append(mt_df.iloc[625765])
abv_collection = abv_collection.append(mt_df.iloc[627355])
abv_collection = abv_collection.append(mt_df.iloc[628947])
abv_collection = abv_collection.append(mt_df.iloc[630544])
abv_collection = abv_collection.append(mt_df.iloc[632142])
abv_collection = abv_collection.append(mt_df.iloc[633738])
abv_collection = abv_collection.append(mt_df.iloc[635332])
abv_collection = abv_collection.append(mt_df.iloc[636927])
abv_collection = abv_collection.append(mt_df.iloc[638521])
abv_collection = abv_collection.append(mt_df.iloc[640115])
abv_collection = abv_collection.append(mt_df.iloc[641710])
abv_collection = abv_collection.append(mt_df.iloc[643305])
abv_collection = abv_collection.append(mt_df.iloc[644902])
abv_collection = abv_collection.append(mt_df.iloc[646503])
abv_collection = abv_collection.append(mt_df.iloc[648105])
abv_collection = abv_collection.append(mt_df.iloc[649712])
abv_collection = abv_collection.append(mt_df.iloc[651319])
abv_collection = abv_collection.append(mt_df.iloc[652923])
abv_collection = abv_collection.append(mt_df.iloc[654527])
abv_collection[['time', 'assetCode', 'open', 'close', 'returnsOpenPrevRaw1', 'returnsOpenPrevMktres1', 'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']]
#outlier_collection
import matplotlib.dates as mdates

asset_df = abv_collection
fig, ax = plt.subplots(ncols=2, nrows=1,figsize=(10,4))

ts = asset_df.iloc[0]["time"]
ts_str = pd.to_datetime(str(ts)) 
d_str = ts_str.strftime('%Y.%b.%d')

ax[0].set_title("ABV open/close: "+d_str)
ax[0].plot(asset_df.time, asset_df.open, label='Open price')
ax[0].plot(asset_df.time, asset_df.close, label='Close price')
ax[0].legend(loc="lower right")
myFmt = mdates.DateFormatter('%d/%b')
ax[0].xaxis.set_major_formatter(myFmt)
for tick in ax[0].get_xticklabels():
    tick.set_rotation(45)
    
ax[1].set_title("ABV Mktres1/10: "+d_str)
ax[1].plot(asset_df.time, asset_df.returnsOpenPrevMktres1, label='returnsOpenPrevMktres1')
ax[1].plot(asset_df.time, asset_df.returnsOpenPrevMktres10, label='returnsOpenPrevMktres10')
ax[1].legend(loc="upper right")
myFmt = mdates.DateFormatter('%d/%b')
ax[1].xaxis.set_major_formatter(myFmt)
for tick in ax[1].get_xticklabels():
    tick.set_rotation(45)

fig.tight_layout()
plt.show()

#outliers = mt_df.sort_values('returnsClosePrevRaw1', ascending=True)[:30]
#graph_outliers_grid(True, True)
#outlier_collection
outlier_collection.sort_values(by='time')


