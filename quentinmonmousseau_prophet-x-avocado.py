import pandas as pd # data structures and data analysis

import matplotlib.pyplot as plt # data visualisation

%matplotlib inline

import seaborn as sns # data visualisation

from fbprophet import Prophet # time series forecasting



# Store the data in a dataframe

df = pd.read_csv("../input/avocado-prices/avocado.csv")

# Visualize the dataframe

df.head()
#Prophet only takes 2 columns, one for the time (ds standing for datestamp), one for the value y you want to predict (i.e. the target)

df = df[['Date', 'AveragePrice']]

df.columns = ['ds', 'y']



#Remove NaN values

df.dropna()
# Reformat date column

df['ds'] = pd.to_datetime(df['ds'])

df = df.set_index('ds')



daily_df = df.resample('D').mean()

d_df = daily_df.reset_index().dropna()



d_df.sort_values(by=['ds'])

df.tail()
pd.plotting.register_matplotlib_converters() #added to avoid an error produced by a recent update



# Plot a line chart

plt.figure(figsize = (25, 10)) # figure size

sns.set_style('whitegrid') # background style

sns.lineplot(x='ds', y='y', data=d_df, color='#76b900') # chart

plt.title('Avocado prices', fontsize=16) # title

plt.xlabel('date', fontsize=16) # x label

plt.ylabel('average price', fontsize=16) # y label
# Initialize a Prophet model

m = Prophet()

# Fit the model on our data

m.fit(d_df)
# Specify the number of days to forecast using the periods parameter

future = m.make_future_dataframe(periods=90)

future.tail()
# Let's forecast

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() # yhat being the prediction
# The black dots represent outliers and the light-blue shaded regions are the uncertainty intervals

pd.plotting.register_matplotlib_converters()

fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
from fbprophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(m, horizon='90 days')

df_cv.head()
df_p = performance_metrics(df_cv)

df_p.head(5)
from fbprophet.plot import plot_cross_validation_metric

fig3 = plot_cross_validation_metric(df_cv, metric='mape')