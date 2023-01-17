import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

%matplotlib inline



oil_data=pd.read_csv('../input/oil-data/DCOILWTICO.csv',header=0)

nee_data = pd.read_csv('../input/nee-stock-data/NEE.csv',header=0)

nee_data = nee_data[['Date', 'Adj Close']]

oil_data = oil_data.loc[oil_data['DCOILWTICO'] != '.']

oil_data = oil_data['DCOILWTICO']

print(oil_data.head())

print(nee_data.head())

start_date = '1986-01-01'

end_date = '2019-07-23'

adjusted_nee_data = nee_data.loc[(nee_data.Date > start_date) & (nee_data.Date < end_date ) ].reset_index(drop=True)

# convert date column to appropriate date obj

adjusted_nee_data['DCOILWTICO'] = oil_data

plot_data = adjusted_nee_data

plot_data['DCOILWTICO'] = pd.to_numeric(plot_data['DCOILWTICO'],errors='coerce')

plot_data['Date'] = pd.to_datetime(plot_data['Date'])

plot_data['pct_change_oil'] = plot_data['DCOILWTICO'].pct_change()

plot_data['pct_change_nee'] = plot_data['Adj Close'].pct_change()

plot_data = plot_data[['Date', 'pct_change_oil','pct_change_nee']][1:]

print(plot_data)
plt.figure(figsize=(20,6))

plt.plot(plot_data['Date'],plot_data['pct_change_oil'].cumsum(),color='black',label="Oil")

plt.plot(plot_data['Date'],plot_data['pct_change_nee'].cumsum(),color='red', label="nee")

plt.legend()

plt.grid()

plt.title("Comparison between NEE stock value and Oil")