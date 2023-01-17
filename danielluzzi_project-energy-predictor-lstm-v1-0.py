import os;

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename));

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



import pylab 

import scipy.stats as stats

import statsmodels.api as sm



from numpy import mean

from numpy import median

from numpy import percentile



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.preprocessing import StandardScaler



from pandas import read_csv

from pandas import datetime

from pandas import DataFrame



from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import autocorrelation_plot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
df = pd.read_csv('/kaggle/input/daily_values_ee.csv')
df.info()
df.head(3)
print('Size of df data', df.shape)
plt.figure(figsize=(16,4))

sns.heatmap(df.isnull()) # plot the missing data

plt.title('Missing Data')



total = df.isnull().sum().sort_values(ascending = False) # count list of missing data per column

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False) # percentage list of missing data per column

missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) # create a df of the missing data

missing_data.head(6)
df=df.tail(593) # this way we will exclude the first 7 rows

df.head()
## Function to reduce the DF size



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



df = reduce_mem_usage(df);

fig, saxis = plt.subplots(3, 2,figsize=(16,14))



sns.distplot(df['53_kW_mean'].dropna(), bins=20, ax = saxis[0,0])

sns.distplot(df['71_kW_mean'].dropna(), bins=20, ax = saxis[1,0])

sns.distplot(df['71A_kW_mean'].dropna(), bins=20, ax = saxis[0,1])

sns.distplot(df['83_kW_mean'].dropna(), bins=20, ax = saxis[1,1])

sns.distplot(df['71_71AkW_mean'].dropna(),bins=20, ax= saxis[2,0])

sns.distplot(df['totalkW_mean'].dropna(), bins=20, ax= saxis[2,1]);
plt.figure(figsize=[16,12])



plt.subplot(321)

plt.plot(df['53_kW_mean'])

plt.ylabel('kW ')

plt.title('53')



plt.subplot(322)

plt.plot(df['71_kW_mean'])

plt.ylabel('kW')

plt.title('71')



plt.subplot(323)

plt.plot(df['71A_kW_mean'])

plt.ylabel('kW')

plt.title('71A')



plt.subplot(324)

plt.plot(df['83_kW_mean'])

plt.ylabel('kW')

plt.title('83')



plt.subplot(325)

plt.plot(df['71_71AkW_mean'])

plt.ylabel('kW')

plt.title('71 + 71A')



plt.subplot(326)

plt.plot(df['totalkW_mean'])

plt.ylabel('kW')

plt.title('Total');



# Since we will work initially with the total values, let's filter it

total_cols = ['Date','totalkW_mean','totalkW_max','totalkW_time_max','totalkW_d-1','totalkW_w-1','totalkW_d/1','totalkW_w/1','Month','Year','Day','WD','CDD_15','HDD_25','NWD','temp_max','insolation','temp_mean','RH']

total_df = df[total_cols]

total_df.info() # review the variables
# let's analyze the correlation of the dataset

plt.figure(figsize=(16,10))

sns.heatmap(total_df.corr(), mask=False, annot=True, cmap='viridis')

plt.title('Correlations');
# focusing on the dependent variable

plt.figure(figsize=(16,1))

sns.heatmap(total_df.corr()[-17:1], mask=False, annot=True, cmap='viridis')

plt.title('Correlations - Total kW');
plt.figure(figsize=(14,6))

plt.plot(total_df['HDD_25'], color='red', label='HDD')

plt.plot(total_df['CDD_15'], color='blue', label='CDD')

plt.plot(total_df['temp_mean'], color='black', label='Mean Temperature')

plt.legend()

plt.xlabel('day')

plt.title('Weather Data - HDD, CDD and Mean Temperature (ºC)')



plt.figure(figsize=(14,6))

plt.plot(total_df['RH'], color='purple',label='Relative Hudimity')

plt.xlabel('day')

plt.title('Weather Data - Relative Humidity (%)');
# here we will plot the rolling mean with a 7 window day period to analyze the weekly frequency

plt.figure(figsize=(16,6))

plt.plot(total_df['totalkW_mean'].rolling(window=7,center=False).mean(),label='Rolling Mean') # analyze the rolling mean considering a 7 days window

plt.plot(total_df['totalkW_mean'].rolling(window=7,center=False).std(),label='Rolling Std Dev')# analyze the rolling std dev considering a 7 days window

plt.legend()

plt.xlabel('day')

plt.title('7 day window - Mean & Std. Deviation (kW)');

autocorr=plot_acf(total_df['totalkW_mean'].dropna(), lags=30); # autocorrelation with a 30 days window
Consumo_WD_M = total_df[['WD','Month','totalkW_mean']].groupby(by=['WD','Month']).mean()['totalkW_mean'].unstack()/1000

plt.figure(figsize=(14,6))

sns.heatmap(Consumo_WD_M,cmap='viridis', annot=True)

plt.title("Mean (MW)");
plt.figure(figsize=(10,5))

sns.boxplot(x='WD', y='totalkW_mean', data=total_df)

plt.title("Day of the Week Mean (kW)");



plt.figure(figsize=(10,5))

sns.boxplot(x='Month', y='totalkW_mean', data=total_df)

plt.title("Monthly Mean (kW)");
 # this plot shows the time during the day where it has the greatest power

count=total_df[['WD','totalkW_time_max','totalkW_max']].groupby(by=['WD','totalkW_time_max']).count()['totalkW_max'].unstack()

count[np.isnan(count)] = 0

plt.figure(figsize=(13,5))

sns.heatmap(count.dropna())

plt.title("Count of maximum power per hour");

total = total_df['totalkW_time_max'].value_counts()

count  = pd.concat([total], axis=1, keys=['Count'])

count.reset_index(drop=False, inplace=True)

count.head(8)

count.sort_values(by=['index'], inplace=True, ascending=True)

plt.figure(figsize=(14,4))

sns.barplot(x='index', y='Count', data=count, color='black')

plt.title("Count of maximum power per hour");





count.sort_values(by=['Count'], inplace=True, ascending=False)

plt.figure(figsize=(14,4))

sns.barplot(x='index', y='Count', data=count, color='black');

plt.title("Count of maximum power per hour - descendant");

# X is the independent variable - aka INPUT

# y is the dependent variable - aka OUTPUT

X = total_df.drop(['Date','Year','Day','totalkW_mean','totalkW_max','totalkW_time_max','insolation'], axis=1).values

y = total_df['totalkW_mean'].values

# the model only works with arrays, not DF
# splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=102)

# since there are not too many data, I thought it was better to have a bigger training set



from sklearn.preprocessing import MinMaxScaler



# normalizing it

scaler = MinMaxScaler()

scaler.fit(X_train) # scale the X_train

X_train = scaler.transform(X_train)  # apply for the train data

X_test = scaler.transform(X_test) # apply for the test data

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])) # need to add a column for using LSTM Models

X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1])) # need to add a column for using LSTM Models



print('X_train shape = ',X_train.shape)

print('X_test shape = ' ,X_test.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import LSTM

from tensorflow.keras.optimizers import Adam





# Here I create a Long Short-Term Memory artificial recurrent neural network as a model

model = Sequential()

model.add(LSTM( units=4, input_shape=(1, 12), activation='sigmoid', return_sequences=True))

# first layer receiving the input. Sigmoid was choosing as an activation funcion in order to work properly for LSTM

model.add(LSTM( units=2, input_shape=(1, 12), activation='linear',recurrent_activation='linear', return_sequences=False))

# the second layer work with a linear activation function to behave more in tune with our input data

model.add(Dense(units=1,activation='linear'))

# finally a last layer with only one output, which is the respective mean power of the day

model.compile(loss='mse', optimizer='adam')

# for the optimization model, the Minimum Squared Error is minimized as the objective function
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)

# added an early stop that is trigger when the difference of the validation loss at each epoch is minimum.
model.fit(x = X_train, 

          y = y_train, 

          epochs = 5000, 

          validation_data = (X_test, y_test), 

          batch_size = 128,

          callbacks=[early_stop],

          verbose=0

         );



# 5000 epochs to train the model

# 128 cases for updating the model parameters, which is the batch size
model_loss = pd.DataFrame(model.history.history)

print(model_loss.tail(1))

model_loss[['loss','val_loss']].plot()

plt.ylabel("loss")

plt.xlabel("epoch")

plt.title("Losses");
predictions = model.predict(X_test)



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % total_df['totalkW_mean'].astype(float).mean())

print("y_test Mean = %.2f" % y_test.astype(float).mean())

print("y_test Std Dev = %.2f" % y_test.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(y_test, lower_p)) # retrieve observation at lower percentile

print('%.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(y_test, upper_p))

print('%.1fth percentile = %.2f' % (upper_p, upper))

print("predictions Mean = %.2f" % predictions.astype(float).mean())

print("predictions Std Dev = %.2f" % predictions.astype(float).std())



print("MAE = %.2f" % mean_absolute_error(y_test,predictions))

print("MSE = %.2f" % np.sqrt(mean_squared_error(y_test,predictions)))

print("EVS = %.2f" % explained_variance_score(y_test,predictions));
plt.figure(figsize=(14,5))

plt.scatter(y_test,predictions, color='blue') # Our predictions



plt.plot(y_test,y_test,'red') # y=x line

plt.plot(y_test,y_test+y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.plot(y_test,y_test-y_test_std_dev,'grey', linewidth=.1) # y=x line





plt.xlabel("kW")

plt.title("Scatter - y_test vs. predicted values (kW)");



plt.figure(figsize=(14,5))

plt.plot(y_test, 'red', linewidth=1)

plt.plot(predictions, 'blue',linewidth=.5)

plt.title("y_test vs. predicted values (kW)")

plt.xlabel("y_test");
y_test_pd = DataFrame(y_test)

predictions_pd = DataFrame(predictions)

comparisson  = pd.concat([y_test_pd, predictions_pd], axis=1, keys=['y_test', 'predictions'])

comparisson['resid'] = comparisson['y_test'] -  comparisson['predictions']

comparisson['resid_ratio'] = (comparisson['predictions'] / comparisson['y_test'])

comparisson['residsqr'] = comparisson['resid']*comparisson['resid']



plt.figure(figsize=[12,4])



comp_resid = plt.plot(comparisson['resid'])

plt.title('Residuals (kW)')

plt.figure(figsize=[12,4])

comp_resid_ratio = plt.plot(comparisson['resid_ratio'])

plt.title('Residuals Ratio')

plt.figure(figsize=(12,4))

sns.distplot((comparisson['resid']), bins=30)

plt.title('Residuals Distribution (kW)')

plt.xlabel("");
autocorr_resid = plot_acf(comparisson['resid']);

plt.title('Autocorrelation - Residuals');



autocorr_resid_sqr = plot_acf(comparisson['residsqr']);

plt.title('Autocorrelation - Residuals Squared');
X_df = total_df.drop(['Date','Day','Year','totalkW_max','totalkW_mean','insolation','totalkW_time_max'], axis=1)

X_df = X_df.reset_index(drop=True)



y_df = total_df['totalkW_mean']

y_df = pd.concat([y_df], axis=1, keys="total")

y_df = y_df.reset_index(drop=True)



initial_values = 450



nl1 = y_df.head(initial_values)

nl2 = X_df.head(initial_values)
i=len(nl1)-1

j=1

pred=0

steps=144

for j in range(1,steps):

    nl2 = nl2.append(pd.Series([nl1.loc[i][0]-nl1.loc[i-1][0]] + [nl1.loc[i][0]-nl1.loc[i-7][0]] + [nl1.loc[i][0]/nl1.loc[i-1][0]] + [nl1.loc[i][0]/nl1.loc[i-7][0]] + [nl2.loc[i-364][4]] + [nl2.loc[i-6][5]] + [X_df.loc[i][6]] + [X_df.loc[i][7]] + [X_df.loc[i][8]] + [X_df.loc[i][9]] + [X_df.loc[i][10]] + [X_df.loc[i][11]],  index=nl2.columns), ignore_index=True)

    #print(nl2)

    nly = nl2.tail(1).values

    nly=scaler.transform(nly)

    nly = np.reshape(nly, (nly.shape[0], 1, nly.shape[1]))

    pred = float(model.predict(nly))

    #print(pred)

    nl1 = nl1.append(pd.Series(pred, index=nl1.columns), ignore_index=True)

    j=j+1

    i=i+1



# Plotting the real and the predicted values

plt.figure(figsize=(10,4))

plt.plot(nl1.head(initial_values),label='Real Series')

plt.plot(nl1.tail(steps), label='Predicted Values')

plt.legend()

plt.title("Model 1 - Application (kW)")

plt.xlabel("day");
plt.figure(figsize=(10,4))

plt.plot(nl2['totalkW_d-1'], label='d-1')

plt.plot(nl2['totalkW_w-1'], label='w-1')

plt.legend()

plt.title("Model 1 - 'totalkW_mean' [d-1] & [w-1] (kW)")

plt.xlabel("day");
# comparing the distribution of the following values



plt.figure(figsize=[18,8])

plt.subplot(221)

sns.distplot(nl2['totalkW_d/1'].head(initial_values), label='Real Series')

sns.distplot(nl2['totalkW_d/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 1 - Total kW [d/1]")

plt.xlabel("");



plt.subplot(223)

sns.distplot(nl2['totalkW_w/1'].head(initial_values), label='Real Series')

sns.distplot(nl2['totalkW_w/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 1 - Total kW [w/1]")

plt.xlabel("");



plt.subplot(222)

sns.distplot(nl2['totalkW_d-1'].head(initial_values), label='Real Series')

sns.distplot(nl2['totalkW_d-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 1 - Total kW [d-1] (kW)")

plt.xlabel("");



plt.subplot(224)

sns.distplot(nl2['totalkW_w-1'].head(initial_values), label='Real Series')

sns.distplot(nl2['totalkW_w-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 1 - Total kW [w-1] (kW)")

plt.xlabel("");
df_real = y_df['t']

df_real = df_real.tail(steps)

df_pred = nl1.tail(steps)



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % df_real.astype(float).mean())

print("Real Std Dev = %.2f" % df_real.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(df_real, lower_p)) # retrieve observation at lower percentile

print('%.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(df_real, upper_p))

print('%.1fth percentile = %.2f' % (upper_p, upper))

print("predictions Mean = %.2f" % df_pred.astype(float).mean())

print("predictions Std Dev = %.2f" % df_pred.astype(float).std())



print("MAE = %.2f" % mean_absolute_error(df_real,df_pred))

print("MSE = %.2f" % np.sqrt(mean_squared_error(df_real,df_pred)))

print("EVS = %.2f" % explained_variance_score(df_real,df_pred))

print("Real Mean = %.2f" % df_real.mean())

print("Prediction Mean = %.2f" % df_pred['t'].mean())
plt.figure(figsize=(12,5)) 

plt.scatter(df_real,df_pred) # Model 1 predictions

plt.plot(df_real,df_real,'red')  # y=x line

plt.plot(y_test,y_test+y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.plot(y_test,y_test-y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.title("Scatter Model 1 - Real Series vs. predicted values (kW)")

plt.xlabel("kW");



plt.figure(figsize=(12,5)) # Plot predictions and the real values together

plt.plot(df_real, 'blue', label= 'Real Series')

plt.plot(df_pred, 'red', label= 'Predictions')

plt.title("Model 1 - Real Series vs. Predicted Values (kW)")

plt.xlabel("days")

plt.legend();
X_df = total_df.drop(['Date','Day','Year','totalkW_max','totalkW_mean','insolation','totalkW_time_max'], axis=1)

X_df = X_df.reset_index(drop=True)



y_df = total_df['totalkW_mean']

y_df = pd.concat([y_df], axis=1, keys="total")

y_df = y_df.reset_index(drop=True)



initial_values = 30



nl3 = y_df.head(initial_values)

nl4 = X_df.head(initial_values)
i=len(nl3)-1

j=1

pred=0

steps=564

for j in range(1,steps):

    nl4 = nl4.append(pd.Series([nl3.loc[i][0]-nl3.loc[i-1][0]] + [nl3.loc[i][0]-nl3.loc[i-7][0]] + [nl3.loc[i][0]/nl3.loc[i-1][0]] + [nl3.loc[i][0]/nl3.loc[i-7][0]] + [X_df.loc[i][4]] + [nl4.loc[i-6][5]] + [X_df.loc[i][6]] + [X_df.loc[i][7]] + [X_df.loc[i][8]] + [X_df.loc[i][9]] + [X_df.loc[i][10]] + [X_df.loc[i][11]],  index=nl4.columns), ignore_index=True)

    #print(nl4)

    nly = nl4.tail(1).values

    nly=scaler.transform(nly)

    nly = np.reshape(nly, (nly.shape[0], 1, nly.shape[1]))

    pred = float(model.predict(nly))

    #print(pred)

    nl3 = nl3.append(pd.Series(pred, index=nl3.columns), ignore_index=True)

    j=j+1

    i=i+1

plt.figure(figsize=(10,4))

plt.plot(nl3.head(initial_values),label='Real Series')

plt.plot(nl3.tail(steps), label='Predicted Values')

plt.legend()

plt.title("Model 2 - Application (kW)")

plt.xlabel("day");
# comparing the distribution of the following values



plt.figure(figsize=[18,8])

plt.subplot(221)

sns.distplot(nl4['totalkW_d/1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_d/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 2 - Total kW [d/1]")

plt.xlabel("");



plt.subplot(223)

sns.distplot(nl4['totalkW_w/1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_w/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 2 - Total kW [w/1]")

plt.xlabel("");



plt.subplot(222)

sns.distplot(nl4['totalkW_d-1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_d-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 2 - Total kW [d-1] (kW)")

plt.xlabel("");



plt.subplot(224)

sns.distplot(nl4['totalkW_w-1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_w-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Model 2 - Total kW [w-1] (kW)")

plt.xlabel("");
df_real = y_df['t']

df_real = df_real.tail(steps)

df_pred = nl3.tail(steps)



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % df_real.astype(float).mean())

print("Real Std Dev = %.2f" % df_real.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(df_real, lower_p)) # retrieve observation at lower percentile

print('%.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(df_real, upper_p))

print('%.1fth percentile = %.2f' % (upper_p, upper))

print("predictions Mean = %.2f" % df_pred.astype(float).mean())

print("predictions Std Dev = %.2f" % df_pred.astype(float).std())



print("MAE = %.2f" % mean_absolute_error(df_real,df_pred))

print("MSE = %.2f" % np.sqrt(mean_squared_error(df_real,df_pred)))

print("EVS = %.2f" % explained_variance_score(df_real,df_pred))

print("Real Mean = %.2f" % df_real.astype(float).mean())

print("Prediction Mean = %.2f" % df_pred['t'].mean())
df_real = y_df['t']

df_real = df_real.tail(steps)

df_pred = nl3.tail(steps)

model_2_df = nl4

model_2_df['predictions'] = df_pred

model_2_df['real'] = df_real



plt.figure(figsize=(12,5)) 

plt.scatter(df_real,df_pred) # Model 1 predictions

plt.plot(df_real,df_real,'red')  # y=x line

plt.plot(y_test,y_test+y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.plot(y_test,y_test-y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.title("Scatter Model 2 - Real Series vs. predicted values (kW)")

plt.xlabel("kW");



plt.figure(figsize=(18,6)) # Plot predictions and the real values together

plt.plot(df_real, 'blue', label= 'Real Series')

plt.plot(df_pred, 'red', label= 'Predictions')

plt.title("Model 2 - Real Series vs. Predicted Values (kW)")

plt.xlabel("days")

plt.legend();
df_model_2 = nl4

df['real'] = df_real

df['predictions'] = df_pred

# join the predictions and the real values to the used DF, so we can correlate all the dependent and independent variables
plt.figure(figsize=(12,4))

sns.lineplot(x='WD', y='real', data=df_model_2, label='real')

sns.lineplot(x='WD', y='predictions', data=df_model_2, label='predictions');

plt.title("Model 2 - Real Series vs. Predicted Values (kW)")

plt.ylabel("")

plt.legend();

g = sns.FacetGrid(df_model_2, col='WD')

g = g.map(plt.hist, 'real')

h = sns.FacetGrid(df_model_2, col='WD')

h = h.map(plt.hist, 'predictions');
plt.figure(figsize=(14,1))

rwd = df_model_2.pivot_table(values='real',columns='WD')

sns.heatmap(rwd/1000, annot=True,linecolor='white',linewidths=1, vmin=0.0, vmax=0.4)

plt.title("Model 2 - Real Series (MW)")



plt.figure(figsize=(14,1));

pwd = df_model_2.pivot_table(values='predictions',columns='WD')

sns.heatmap(pwd/1000, annot=True,linecolor='white',linewidths=1, vmin=0.0, vmax=0.4)

plt.title("Model 2 - Predictions (MW)")



plt.figure(figsize=(14,1));

ratio_wd = pwd.values/rwd.values

sns.heatmap(ratio_wd, annot=True,linecolor='white',linewidths=1, vmin=0.8, vmax=1.4)

plt.title("Model 2 - Predictions / Real Series");
plt.figure(figsize=(14,2))

sns.heatmap(df_model_2.corr()[-2:17], annot=True,linecolor='white',linewidths=1);