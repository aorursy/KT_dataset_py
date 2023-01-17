import pandas as pd # for reading in the dataset
google_trends_data = pd.read_csv('/kaggle/input/google-trends-volumes-for-8000-stocks/google_search_history.csv')
google_trends_data['date'] = pd.DatetimeIndex(google_trends_data['date'])
google_trends_data[google_trends_data.columns[1:]] = google_trends_data[google_trends_data.columns[1:]].astype(float)
google_trends_data.set_index('date', inplace = True) # set date to index
import os 
cols = []
for file in os.listdir('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks'):
    cols.append(file[:file.find('.us')])
from tqdm import tqdm # view progress
lowered_gt_cols = [stock.lower() for stock in google_trends_data.columns]
stock_data = pd.DataFrame(columns = cols)
filter = 100 # the data download is cut to 100 to keep the download short. Set it to -1 to download all stock datas.
for file in tqdm(os.listdir('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks')[:filter]):
    stock_name = file[:file.find('.us')]
    if stock_name in lowered_gt_cols:
        data_for_stock = pd.read_csv('/kaggle/input/price-volume-data-for-all-us-stocks-etfs/Stocks/{}'.format(file))
        stock_data[stock_name] = data_for_stock.set_index('Date')['Close']
stock_data = stock_data.dropna(axis = 1)
stock_data.index = pd.DatetimeIndex(stock_data.index)
import matplotlib.pyplot as plt
import sklearn.preprocessing
stock = 'VIV'
stock_prices = stock_data[stock.lower()][1200:] 
google_trends = google_trends_data[stock][google_trends_data.index >= stock_prices.index.min()]
minmax_scaler = sklearn.preprocessing.MinMaxScaler([stock_prices.min(), stock_prices.max()])
plt.plot(google_trends.index, minmax_scaler.fit_transform(google_trends.values.reshape(-1, 1)))
plt.plot(stock_prices)

