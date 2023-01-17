import pandas as pd
import numpy as np
import seaborn as sns

from fbprophet import Prophet

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
def load_data(filename):
    """
    Loading data and applying types casting.
    
    This way it is easier to have the same data both for the training and testing data
    """
    data_dtypes = {
    'Open': 'category', 
    'Promo': 'category', 
    'StateHoliday': 'category',
    'SchoolHoliday': 'category'
    }
    res = pd.read_csv(
        filename, parse_dates=['Date'], dtype=data_dtypes)
    res.sort_values(by='Date', inplace=True)
    return res
train = load_data('../input/train.csv')

# The test data set won't be used in this notebook
# test = load_data('../data/raw/test.csv')

# Add dates related features
train['Month'] = train.Date.dt.month
train['WeekOfYear'] = train.Date.dt.weekofyear

# Daily sales per customer
train['SalesPerCustomer'] = train.Sales / train.Customers

print("Loaded {} entries".format(train.shape[0]))

train.head(10)
train[(train.Open=='0') & (train.Sales > 0)].shape[0]
train[(train.Open=='1') & (train.Sales == 0)].shape[0]
train = train[(train.Open=='1') & (train.Sales > 0)]
ax = train.groupby('Month')[['Sales', 'Customers']].mean().plot(secondary_y=['Customers'], figsize=(15,5), marker='o')
ax.set_ylabel('€')
ax.right_ax.set_ylabel('Count')
ax.set_xticks(range(1,13))
plt.title('Average sales and number of customers per month');
ax = train.groupby('Month')[['SalesPerCustomer']].mean().plot(figsize=(15,5), marker='o')
ax.set_ylabel('€')
ax.set_xticks(range(1,13))
plt.title('Average sales per customer per month');
train.groupby(['DayOfWeek'])['Sales'].mean().plot(figsize=(13,5))
plt.title('Average sales per day of the week')
plt.ylabel('€');
ax = train.groupby('DayOfWeek')[['SalesPerCustomer']].mean().plot(figsize=(15,5), marker='o')
ax.set_ylabel('€')
plt.title('Average spend per customer per day of the week');
tmp = train.groupby(['Month', 'DayOfWeek'])['SalesPerCustomer'].mean().reset_index()
sns.factorplot(data = tmp, x = 'Month', y = "SalesPerCustomer", row = 'DayOfWeek',
              size=2, aspect=4.5);
train.groupby('WeekOfYear')['Sales'].median().plot(figsize=(15,5), xticks=np.arange(1,53), rot=90)
plt.title('Median of sales per week of the year')
plt.ylabel('€');
tmp = train.groupby(['Store'])['Sales'].agg(['mean', 'std']).sort_values('mean')
tmp.plot.scatter('mean', 'std', figsize=(8,8))
plt.title('Daily average sale vs. STD');
sns.distplot(train.groupby(['Store'])['Customers'].mean())
def train_data_store(storeID, df=train):
    """
    For a given store, return the daily sales 
    formated for prophet
    """
    condition = df.Store==storeID
    data = df[condition][['Date', 'Sales']].rename(
        columns={ # Rename columns to meet needs of the prophet
            'Date': 'ds',
            'Sales': 'y'
        }
    ).sort_values('ds')
    return data
def RMSPE(y, yhat):
    """
    Compute the score as per the definition on 
    https://www.kaggle.com/c/rossmann-store-sales#evaluation
    """
    return np.sqrt(((y - yhat).div(y) ** 2).sum() / len(y))
def fit_prophet_for_store(storeID=1, train_ratio=0.8, holidays=None):
    """
    Given a store ID, take ~`train_traio` of the data
    for training of the prophet
    """
    data = train_data_store(storeID)
    
    N = int(np.floor(train_ratio * data.shape[0]))
    print("Training on {}% of the data = {} entries".format(
        train_ratio, N
    ))
    
    prophet = Prophet(interval_width = 0.95, holidays = holidays)
    prophet.fit(data[:N])
    
    return prophet, data, N
storeID=1
prophet, data, N = fit_prophet_for_store(storeID=storeID, train_ratio=0.9)
df_sub = pd.DataFrame(prophet.predict(data))
prophet.plot(prophet.predict(data));
RMSPE(data[N:]['y'].values, prophet.predict(data[N:])['yhat'])
stateHolidaysDates = train[train.StateHoliday.isin(['a', 'b', 'c'])]['Date'].unique()
schoolHolidaysDates = train[train.SchoolHoliday=='1']['Date'].unique()
holidays = pd.concat(
    [
        pd.DataFrame({
            'holiday': 'state',
            'ds': stateHolidaysDates
        }),
        pd.DataFrame({
            'holiday': 'school',
            'ds': schoolHolidaysDates
        })
    ]
)
holidays.sample(5)
prophet, data, N = fit_prophet_for_store(storeID=storeID, train_ratio=0.9, holidays=holidays)
prophet.plot(prophet.predict(data));
prophet.predict(data)
Prediction_accuracy = 1 - RMSPE(data[N:]['y'].values, prophet.predict(data[N:])['yhat'])
print(Prediction_accuracy*100)