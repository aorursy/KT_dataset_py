# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

  
sunspots = pd.read_csv('/kaggle/input/sunspots/Sunspots.csv',index_col='Date',parse_dates = True) 
sunspots = sunspots.rename(columns={'Monthly Mean Total Sunspot Number':'MeanSunspotNum'})
sunspots = sunspots.drop(['Unnamed: 0'], axis=1)
sunspots = sunspots[sunspots.index > '2000-01-31']
sunspots.head()
#### plot the data
def plot_df(df, x, y, title="", xlabel='Date', ylabel='MeanSunspotNum', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(sunspots, x=sunspots.index, y=sunspots.MeanSunspotNum, title='Series')    
#### Seasonal plot
# Prepare data
df = sunspots.reset_index()
df['year'] = [d.year for d in df.Date]
df['month'] = [d.strftime('%b') for d in df.Date]
years = df['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
plt.show()
# type(df.loc[df.year==y, :])
# plt.plot(df.month, df.MeanSunspotNum, color='tab:red')
for i, y in enumerate(years[::-1][:6]):
    if i > 0:   
        plt.plot('month', 'MeanSunspotNum', data=df.loc[df.year==y, :], color=mycolors[0], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'MeanSunspotNum'][-1:].values[0], y, fontsize=12, color=mycolors[i])
        
# Decoration
plt.gca().set(ylabel='MeanSunspotNum', xlabel='Month')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of SunspotsMeanNum", fontsize=16)
plt.show()
df['year'] = [d.year for d in df.Date]
df['month'] = [d.strftime('%b') for d in df.Date]
years = df['year'].unique()

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='MeanSunspotNum', data=df.loc[df.year.isin(list(range(2000, 2019))), :], ax=axes[0])
sns.boxplot(x='month', y='MeanSunspotNum', data=df.loc[df.year.isin(list(range(2000, 2019))), :])

# # Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()
from statsmodels.tsa.stattools import adfuller, kpss

# ADF Test
result = adfuller(df.MeanSunspotNum.values[:], autolag='AIC')
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')

# KPSS Test
result = kpss(df.MeanSunspotNum.values[:], regression='c')
print('\nKPSS Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[3].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


result_add = seasonal_decompose(sunspots['MeanSunspotNum'],  
                            model ='additive') 
result_add.plot()

!pip install pmdarima
from pmdarima import auto_arima 

stepwise_fit = auto_arima(sunspots['MeanSunspotNum'], start_p = 1, start_q = 1, 
                          max_p = 4, max_q = 4, m = 43, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise 
  
# To print the summary 
stepwise_fit.summary() 
# Split data into train / test sets 
train = sunspots[sunspots.index < '2018-01-31'] 
test = sunspots[sunspots.index >= '2018-01-31']  # set for testing 
  
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['MeanSunspotNum'],  
                order = (3, 1, 1),  
                seasonal_order =(0, 1, 1, 43)) 
  
result = model.fit() 
result.summary() 
start = len(train) 
end = len(train) + len(test) - 1
  
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
  
# plot predictions and actual values 
predictions.plot(legend = True) 
test['MeanSunspotNum'].plot(legend = True) 
# Load specific evaluation tools 
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
  
# Calculate root mean squared error 
print (rmse(test["MeanSunspotNum"], predictions)) 
  
# Calculate mean squared error 
print (mean_squared_error(test["MeanSunspotNum"], predictions)) 
model = SARIMAX(sunspots['MeanSunspotNum'],  
                        order = (3, 1, 1),  
                seasonal_order =(0, 1, 1, 43)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(sunspots),  
                          end = (len(sunspots)-1) + 3 * 12,  
                          typ = 'levels').rename('Forecast') 
  
# Plot the forecast values 
sunspots['MeanSunspotNum'].plot(figsize = (12, 5), legend = True) 
forecast.plot(legend = True) 
