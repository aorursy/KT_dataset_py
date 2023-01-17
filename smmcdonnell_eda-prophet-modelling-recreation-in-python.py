import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
plt.style.use('ggplot')
path = "../input/train.csv"
train = pd.read_csv(path)
train.head()
train.shape
train["year"] = pd.to_datetime(train["date"]).dt.year
train["month"] = pd.to_datetime(train["date"]).dt.month
train["month_year"] = pd.to_datetime(train["date"]).dt.to_period('M')
train.head()
# train.count() will give a value only if not nan
count_nan = len(train) - train.count()
count_nan
plt.hist(train["sales"])
plt.show()
# R script: MSP <- aggregate(sales ~date, train, mean)
mean_sales = train.groupby(["date"], as_index=False)
mean_sales = mean_sales[["sales"]].mean()
mean_sales["idx"] = mean_sales.index
# Could use the follow:
# plt.scatter(x=mean_sales["date"], y=mean_sales["sales"])
# plt.show()
# Seaborn gives us a closer analogue to the work done in R.
g = sns.relplot(x="idx", y="sales", data=mean_sales, kind="line")
# Change in rate of sales
# R script: MSP$rate = c(0, 100*diff(MSP$sales)/MSP[-nrow(MSP),]$sales)
# plt.scatter(x=mean_sales.index, y=rt)
# plt.show()
# rt is short form for "Rate"
rt = pd.Series(mean_sales["sales"]).pct_change()
rt = pd.DataFrame(rt)
rt["idx"] = rt.index
rt.fillna(0, inplace=True)
g = sns.relplot(y="sales", x="idx", data=rt, kind="line")
# R script: MSP <- aggregate(sales ~<Month, train, mean)
# plt.scatter(x=mean_sales_monthly.index, y=mean_sales_monthly["sales"])
# plt.show()
# Used index instead of month-year values 
# because matplotlib complains otherwise
mean_sales_monthly = train.groupby(["month_year"], as_index=False)
mean_sales_monthly = mean_sales_monthly[["sales"]].mean()
mean_sales_monthly["idx"] = mean_sales_monthly.index
g = sns.relplot(y="sales", x="idx", data=mean_sales_monthly, kind="line")
# Change in rate of sales
# R script: MSP$rate = c(0, 100*diff(MSP$sales)/MSP[-nrow(MSP),]$sales)
# rt = pd.Series(mean_sales_monthly["sales"]).pct_change()
rt = pd.Series(mean_sales_monthly["sales"]).pct_change()
rt = pd.DataFrame(rt)
rt["idx"] = rt.index
rt.fillna(0, inplace=True)
g = sns.relplot(y="sales", x="idx", data=rt, kind="line")
# R script: MSP <- aggregate(sales ~Year, train, mean)
#plt.scatter(x=mean_sales_yearly.year, y=mean_sales_yearly["sales"])
#plt.show()
mean_sales_yearly = train.groupby(["year"], as_index=False)
mean_sales_yearly = mean_sales_yearly[["sales"]].mean()
mean_sales_yearly["idx"] = mean_sales_yearly.index
g = sns.relplot(y="sales", x="idx", data=mean_sales_yearly, kind="line")
# Change in rate of sales
# R script: MSP$rate = c(0, 100*diff(MSP$sales)/MSP[-nrow(MSP),]$sales)
#plt.scatter(x=mean_sales_yearly.year, y=rt)
#plt.show()
rt = pd.Series(mean_sales_yearly["sales"]).pct_change()
rt = pd.DataFrame(rt)
rt["idx"] = rt.index
rt.fillna(0, inplace=True)
g = sns.relplot(y="sales", x="idx", data=rt, kind="line")
#unique(train$store)
#Year_state<-aggregate(sales ~store+Year, train,mean)
#pal<-rep(brewer.pal(10, "BrBG"),5)
#stores = pd.unique(train["store"])
#years_stores = train.groupby(["year", "store"], as_index=False)
#years_stores = years_stores[["sales"]].mean()
#plt.scatter(x=years_stores["sales"],y=years_stores["store"])
data = train.groupby(['store',"year"])
mean = data[["sales"]].mean()
mean = mean.add_suffix('').reset_index()
g = sns.relplot(y="sales", x="year", data=mean, kind="line", hue="store")
# unique(train$item)
# Year_state<-aggregate(sales ~item+Year, train,mean)
data = train.groupby(['item',"year"])
mean = data[["sales"]].mean()
mean = mean.add_suffix('').reset_index()
g = sns.relplot(y="sales", x="year", data=mean, kind="line", hue="item")
import warnings
warnings.filterwarnings('ignore')
s1i1 = train[(train["store"]==1) & (train["item"])==1]
s1i1["sales"] = np.log1p(s1i1["sales"])
s1i1.head()
# R script: stats=aggregate(stats$y,by=list(stats$ds),FUN=sum)
# R script: MSP <- aggregate(sales ~Year, train, mean)
stats = s1i1[["date", "sales"]]
stats.columns = ["ds", "y"]
stats.head()
m = Prophet()
m.fit(stats)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
playoffs = ['2013-07-12', '2014-07-12', '2014-07-19',
                 '2014-07-02', '2015-07-11', '2016-07-17',
                 '2016-07-24', '2016-07-07','2016-07-24']
superbowl = ['2013-01-01', '2013-12-25', '2014-01-01', '2014-12-25','2015-01-01', '2015-12-25','2016-01-01', '2016-12-25',
                '2017-01-01', '2017-12-25']

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(playoffs),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(superbowl),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
s1i1["dow"] = pd.to_datetime(s1i1["date"]).dt.day_name() # day of week
s1i1.head()
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0
stats = s1i1[["date", "sales"]]
stats.columns = ["ds", "y"]
stats.head()
stats["nfl_sunday"] = stats['ds'].apply(nfl_sunday)
stats.head()
# R script below:
#model_prophet <- prophet()
#model_prophet <- add_regressor(model_prophet, 'nfl_sunday')
#model_prophet <- add_seasonality(model_prophet, name='daily', period=60, fourier.order=5)
#model_prophet <- prophet(stats, holidays = holidays,holidays.prior.scale = 0.5, yearly.seasonality = 4,
#                         interval.width = 0.95,changepoint.prior.scale = 0.006,daily.seasonality = T)
#future = make_future_dataframe(model_prophet, periods = 90, freq = 'days')
#forecast = predict(model_prophet, future)
m = Prophet(holidays=holidays, holidays_prior_scale=0.5,
            yearly_seasonality=4,  interval_width=0.95,
            changepoint_prior_scale=0.006, daily_seasonality=True)
m.add_regressor('nfl_sunday')
m.add_seasonality(name='daily', period=60, fourier_order=5)
m.fit(stats)
future = m.make_future_dataframe(periods=90, freq="D") # Daily frequency
future['nfl_sunday'] = future['ds'].apply(nfl_sunday)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
# R script: predict_store1_item1=data.frame(date=forecast$ds,forecast=expm1(forecast$yhat))
# predict_store1_item1$yearmonth=as.yearmon(predict_store1_item1$date)
# colnames(predict_store1_item1)<-c("ds","forecast","yearmonth")
ps1i1 = forecast[["ds"]]
ps1i1["forecast"] = np.expm1(forecast["yhat"])
ps1i1["yearmonth"] = pd.to_datetime(ps1i1["ds"]).dt.to_period("M")
ps1i1.head()
def smape(outsample, forecast):
    num = np.abs(outsample-forecast)
    denom = np.abs(outsample) + np.abs(forecast)
    return (num/denom)/2

stats["ds"] = pd.to_datetime(stats["ds"])
ps1i1["ds"] = pd.to_datetime(ps1i1["ds"])
# R script: train_predict = merge(stats, ps1i1, by = "ds", all.x=T)
train_predict = stats.merge(ps1i1)
smape_err = smape(train_predict["y"], train_predict["forecast"])
smape_err = smape_err[~np.isnan(smape_err)]
np.mean(smape_err)
# Training data from the very beginning
# Note that I've added some columns
train["sales"] = np.log1p(train["sales"]) 
train.columns = ["ds", "store", "item", "sales", "y", "m", "my"]
train.head()
def make_prediction(df):
    
    playoffs = ['2013-07-12', '2014-07-12', '2014-07-19',
                 '2014-07-02', '2015-07-11', '2016-07-17',
                 '2016-07-24', '2016-07-07','2016-07-24']
    superbowl = ['2013-01-01', '2013-12-25', '2014-01-01', '2014-12-25','2015-01-01', '2015-12-25','2016-01-01', '2016-12-25',
                    '2017-01-01', '2017-12-25']

    playoffs = pd.DataFrame({
      'holiday': 'playoff',
      'ds': pd.to_datetime(playoffs),
      'lower_window': 0,
      'upper_window': 1,
    })
    superbowls = pd.DataFrame({
      'holiday': 'superbowl',
      'ds': pd.to_datetime(superbowl),
      'lower_window': 0,
      'upper_window': 1,
    })
    holidays = pd.concat((playoffs, superbowls))
    
    m = Prophet(holidays=holidays, holidays_prior_scale=0.5,
            yearly_seasonality=4,  interval_width=0.95,
            changepoint_prior_scale=0.006, daily_seasonality=True)
    m.add_seasonality(name='daily', period=60, fourier_order=5)
    m.fit(df)
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)
    return forecast
df = train[(train["store"]==1) & (train["item"] ==2)]
df = df[["ds", "sales"]]
df.columns = ["ds", "y"]
df.head()
prediction = make_prediction(df)
prediction[["ds", "yhat"]].tail()
