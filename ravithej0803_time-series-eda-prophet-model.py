# Import the necessary libraries
# preprocessing libraries
import pandas as pd
import numpy as np

## visualization libraries
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


## modelling libraries
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

print("all libraries imported")
# reading the dataset
# Import the necessary libraries
# preprocessing libraries
import pandas as pd
import numpy as np

## visualization libraries
import seaborn as sns
color = sns.color_palette()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


## modelling libraries
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

print("all libraries imported")
data.head(10)


data = pd.read_csv('../input/crypto-markets.csv')
data.head(10)
# Info about data
data.info()

# convert the datatype of date to datetime from object
data['date'] = pd.to_datetime(data['date'])
data.info()
# To normalize the market cap and volume values
data['market in billions'] = data['market'] / 1000000000
data['volume in millions'] = data['volume'] / 1000000000
data['volume in billons'] = data['volume']
data
data_bitcoin = data[data.name == 'Bitcoin']
data_bitcoin.head(10)

# PLotting the Bitcoin closing prices distribution
data_bitcoin['date'] = pd.to_datetime(data_bitcoin['date'])
data_bitcoin['Date_mpl'] = data_bitcoin['date'].apply(lambda x: mdates.date2num(x))

fig, ax = plt.subplots(figsize=(12,8))
sns.tsplot(data_bitcoin.close.values, time=data_bitcoin.Date_mpl.values, alpha=0.8, color=color[4], ax=ax)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
fig.autofmt_xdate()
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title("Closing price distribution of bitcoin", fontsize=15)
plt.show()
data.info()
ax = data.groupby(['name'])['market in billions'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Market cap (in billion USD)");
ax.set_ylabel("Cryptocurrency Name");
plt.title("Top 10 Currencies by Market Cap");
ax = data.groupby(['name'])['volume in millions'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Transaction Volume (in million)");
plt.title("Top 10 Currencies by Transaction Volume");
top_5_currency_names = data.groupby(['name'])['market'].last().sort_values(ascending=False).head(5).index
data_top_5_currencies = data[data['name'].isin(top_5_currency_names)]
data_top_5_currencies.head(5)
ax = data_top_5_currencies.groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency");
ax = data_top_5_currencies.groupby(['date', 'name'])['market'].mean().unstack().plot();
ax.set_ylabel("Market Cap");
plt.title("Market Cap for different currencies");
ax = data_top_5_currencies.groupby(['date', 'name'])['volume'].mean().unstack().plot();
ax.set_ylabel("Volume of transactions");
plt.title("Transactional volume per currency");
# To predict the future price of bitcoin crypto currency using Prophet library

from fbprophet import Prophet
#data_bitcoin[["date","close"]].head(10)
data_bitcoin_analysis = data_bitcoin[["date","close"]]
#data_bitcoin_analysis.head(5)

data_bitcoin_analysis.columns = ["ds", "y"]

m = Prophet()
m.fit(data_bitcoin_analysis);
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast)

# Plot individulal components of forecast: trend, weekly/yearly seasonality
m.plot_components(forecast)
