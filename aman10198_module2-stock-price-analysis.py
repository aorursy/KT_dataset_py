import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/module2-dataset/week2.csv")
data.head()
data.drop('Date.1', axis = 1, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', drop = False, inplace = True)
data.head()
plt.figure(figsize = (20,8), clear = True, dpi = 100)

sns.set(style = 'dark')

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.ylim(1000,4000)

sns.lineplot(x = 'Date', y = 'Close Price', data = data)

plt.xlabel('Date', size = 17)

plt.ylabel('Close Price', size = 17)

plt.title('Close Price per Day', size = 20)

plt.show()
# I case

data['Day_Perc_Change'].max()
# II case

data['Day_Perc_Change'].min()
day_the_ClosePrice_dropped = data['Day_Perc_Change'].idxmin()
day_the_ClosePrice_dropped
plt.figure(figsize = (15,6), dpi = 100)

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.stem( data['Date'],data['Day_Perc_Change']*100, linefmt = 'C0-', markerfmt = 'C5o')

plt.ylim(-60,10)

plt.xlabel('Date')

plt.ylabel('Day Per Change')

plt.xlabel('Date', size = 15)

plt.ylabel('Day Per Change Price (%)', size = 15)

plt.title('Price Change per Day', size = 20)

plt.show()
max_old = data['Total Traded Quantity'].max()

min_old = data['Total Traded Quantity'].min()

tota_traded_quantity_normalized = ((data['Total Traded Quantity'] - min_old)/(max_old - min_old)) * (1 - 0) + 0
plt.figure(figsize = (15,6), dpi = 100)

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.plot(tota_traded_quantity_normalized)

plt.xlabel('Date')

plt.ylabel('Day Per Change')

plt.xlabel('Date', size = 15)

plt.ylabel('Total Traded Quantity normalized', size = 15)

plt.title('Normalized Total Traded Quantity Per Day', size = 20)

plt.show()
plt.figure(figsize = (15,6), dpi = 100)

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.plot(tota_traded_quantity_normalized*100)

plt.stem( data['Date'],data['Day_Perc_Change']*100, linefmt = 'C3--', markerfmt = ' ')

plt.legend(['Total Traded Quantity (normalized)', '% Change in Price in per day'])

plt.xlabel('Date', size = 15)

plt.ylim(-100,103)

plt.show()
total_traded_quantity_in_Trends = dict(data.groupby('Trend')['Total Traded Quantity'].sum())
total_traded_quantity_in_Trends.values()
count_total_traded_quantity_in_Trends = data.groupby('Trend')['Symbol'].count()
key = list(total_traded_quantity_in_Trends.keys())

key
values = []

for i in range(len(key)):

    

    aux_data = data[data['Trend'] == key[i]]['Total Traded Quantity']

    values.append([key[i], aux_data.mean(), aux_data.median()])

    print("For key >>", values[i])
total_traded_quantity_in_Trends.keys()
colors = ['yellowgreen', 'lightblue', 'mistyrose', 'lightcoral', 'lightgrey', 'darkcyan', 'cornsilk']



plt.figure(figsize = (16,8), dpi = 100)



grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.0)



plt.subplot(grid[0:, 0])

plt.subplot(1,2,1)

colors = ['yellowgreen', 'coral', 'hotpink', 'peachpuff', 'lightpink', 'gold','darkcyan']

pie_plot = plt.pie(total_traded_quantity_in_Trends.values(), labels = total_traded_quantity_in_Trends.keys(),

                  autopct = '%1.1f%%', startangle=140, colors = colors)



ax1 = plt.subplot(grid[0,1])

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.bar([val_key[0] for val_key in values], [val_mean[1] for val_mean in values], color = 'mediumslateblue')

plt.xticks(rotation=45, size = 13)

plt.legend(['mean'])



ax2 = plt.subplot(grid[1,1], sharex = ax1)

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.bar([val_key[0] for val_key in values], [val_median[2] for val_median in values], color = 'hotpink')

plt.legend(['median'])



plt.xticks(rotation=45, size = 13)



plt.show()
plt.figure(figsize = (15,6))

sns.set(style = 'dark')

plt.grid(True, color = 'darkBlue', alpha = 0.2)

sns.kdeplot(data['Day_Perc_Change'], shade = True)

plt.show()
data1 = pd.read_csv("../input/datalarge-cap/MARUTI.csv")

data1 = data1[data1['Series'] == 'EQ'][['Close Price', 'Date']][1:]

data1.set_index('Date', drop = True, inplace = True)



data2 = pd.read_csv("../input/datalarge-cap/INFY.csv")

data2 = data2[data2['Series'] == 'EQ'][['Close Price', 'Date']][1:]

data2.set_index('Date', drop = True, inplace = True)



data3 = pd.read_csv("../input/datalarge-cap/AXISBANK.csv")

data3 = data3[data3['Series'] == 'EQ'][['Close Price', 'Date']][1:]

data3.set_index('Date', drop = True, inplace = True)



data4 = pd.read_csv("../input/datalarge-cap/TITAN.csv")

data4 = data4[data4['Series'] == 'EQ'][['Close Price', 'Date']][1:]

data4.set_index('Date', drop = True, inplace = True)



data5 = pd.read_csv("../input/datalarge-cap/HDFCBANK.csv")

data5 = data5[data5['Series'] == 'EQ'][['Close Price', 'Date']][1:]

data5.set_index('Date', drop = True, inplace = True)
data_of_company = pd.DataFrame({'Maruti':data1['Close Price'],

                                    'INFY': data2['Close Price'],

                                    'AXISBANK':data3['Close Price'],

                                    'Titan': data4['Close Price'],

                                    'HDFCBANK': data5['Close Price']})
print(data_of_company.shape)

data_of_company.head()
data_per_change_company = pd.DataFrame({

    'Maruti': data_of_company['Maruti'].pct_change(),

    'INFY'  : data_of_company['INFY'].pct_change(),

    'AXISBANK'  : data_of_company['AXISBANK'].pct_change(),

    'Titan'  : data_of_company['Titan'].pct_change(),

    'HDFCBANK'  : data_of_company['HDFCBANK'].pct_change(),

})
data_per_change_company.head(3)
data_per_change_company.dropna(inplace = True)

data_per_change_company.head(3)
g = sns.pairplot(data_per_change_company)

plt.show()
rolling_average_7_day = data_per_change_company['Maruti'].rolling(7,axis = 0).mean()

rolling_average_7_day.dropna(inplace = True)
std_deviation = rolling_average_7_day.std()
std_deviation # it represent the variation in the data we are going to observe 
rolling_average_7_day.mean()
plt.figure(figsize = (15,6), dpi = 100)

plt.grid(color = 'darkBlue', alpha = 0.3)

sns.distplot(rolling_average_7_day, kde = False)

plt.show()
plt.figure(figsize = (15,6), dpi = 100)

rolling_average_7_day.plot()

plt.ylim(-0.050,0.050)

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.show()
nifty50 = pd.read_csv("../input/nifty50/Nifty50.csv")
print(nifty50.shape)

nifty50.head()
nifty50.tail()
nifty_per_change = nifty50['Close'].pct_change()

nifty_per_change.dropna(inplace = True)
nifty_rolling_average_7_day = nifty_per_change.rolling(7,axis = 0).mean()

nifty_rolling_average_7_day.dropna(inplace = True)
plt.figure(figsize = (15,6), dpi = 100)

sns.set(style = 'white')

rolling_average_7_day.plot(color = 'red')

nifty_rolling_average_7_day.plot(color = 'darkBlue')

plt.ylim(-0.02,0.1)

plt.legend(['Maruti', 'nifty'])

plt.grid(color = 'darkBlue', alpha = 0.3)

plt.show()
moving_average_21 = data['Average Price'].rolling(21).mean()

moving_average_34 = data['Average Price'].rolling(34).mean()
plt.figure(figsize = (15,6))

plt.grid(True)

moving_average_21.plot( color = 'red', alpha = 0.5, label = 'Moving Average 21')

moving_average_34.plot( color = 'blue', alpha = 0.5, label = 'Moving Average 34')

data['Average Price'].plot(color = 'green', label = 'Average Price', linewidth = 1.5)

plt.legend()

plt.title('TCS Average Price')

plt.show()
new_data = pd.DataFrame({'14_day_Avg': data['Close Price'].rolling(14).mean(),

                        '14_day_std':data['Close Price'].rolling(14).std(),

                        'Close Price':data['Close Price']})
new_data['Upper_Band'] = new_data['14_day_Avg'] + new_data['14_day_std']*2

new_data['Lower_Band'] = new_data['14_day_Avg'] - new_data['14_day_std']*2
plt.figure(figsize = (15,6))

new_data['Close Price'].plot(color = 'black', linewidth = 2)

new_data['14_day_Avg'].plot(color = 'k', alpha = 0.4)

new_data['Upper_Band'].plot(color = 'darkBlue', alpha = 0.6)

new_data['Lower_Band'].plot(color = 'red', alpha = 0.6)

plt.legend()

plt.grid(True)

plt.title('30 Day Bollinger Band for TCS')

plt.ylabel('Price (USD)')

plt.show();
new_data.to_csv('week3.csv')