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
plt.figure(figsize=(16,6))

sns.distplot(df['53_kW_mean'].dropna(), bins=40, label='53')

sns.distplot(df[df['71_kW_mean']!=0]['71_kW_mean'].dropna(), bins=20, label='71')

sns.distplot(df['71A_kW_mean'].dropna(), bins=40, label='71A')

sns.distplot(df['83_kW_mean'].dropna(), bins=40, label='83')

plt.legend()

plt.xlim(0,300)

plt.xlabel("kW")

plt.title('Distribution + kde - Substations (kW)', fontsize=20);

plt.ylabel('p.u');



plt.figure(figsize=(16,6))

sns.distplot(df['totalkW_mean'].dropna(), bins=40, color='orange');

plt.title('Distribution + kde - Total SEs (kW)', fontsize=20);

plt.xlabel("kW")

plt.ylabel('p.u');
plt.figure(figsize=[16,12])



plt.subplot(321)

plt.plot(df['53_kW_mean'])

plt.ylabel('kW ')

plt.title('53', fontsize=20)



plt.subplot(322)

plt.plot(df['71_kW_mean'], color='orange')

plt.ylabel('kW')

plt.title('71', fontsize=20)



plt.subplot(323)

plt.plot(df['71A_kW_mean'], color='orange')

plt.ylabel('kW')

plt.title('71A', fontsize=20)



plt.subplot(324)

plt.plot(df['83_kW_mean'])

plt.ylabel('kW')

plt.title('83', fontsize=20)



plt.subplot(325)

plt.plot(df['71_71AkW_mean'])

plt.ylabel('kW')

plt.title('71 + 71A', fontsize=20)



plt.subplot(326)

plt.plot(df['totalkW_mean'], color='orange')

plt.ylabel('kW')

plt.title('Total', fontsize=20);



# Since we will work initially with the total values, let's filter it

total_cols = ['Date','totalkW_mean','totalkW_max','totalkW_time_max','totalkW_d-1','totalkW_w-1','totalkW_d/1','totalkW_w/1','Month','Year','Day','WD','CDD_15','HDD_25','NWD','temp_max','insolation','temp_mean','RH']

total_df = df[total_cols]

total_df.info() # review the variables
corr = total_df.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='viridis', vmax=1, center=0,

            square=False, linewidths=1, cbar_kws={"shrink": .5}, annot=True)



plt.title('Correlations', fontsize = 20)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 12);
# focusing on the dependent variable

plt.figure(figsize=(18,1))

sns.heatmap(total_df.corr()[-18:1], mask=False, annot=True, cmap='viridis')

plt.title('Correlations - Total kW');
plt.figure(figsize=(14,6))

plt.plot(total_df['HDD_25'], color='#CD661D', label='HDD')

plt.plot(total_df['CDD_15'], color='#3D59AB', label='CDD')

plt.plot(total_df['temp_mean'], color='black', label='Mean Temperature')

plt.legend()

plt.xlabel('day')

plt.title('Weather Data - HDD, CDD and Mean Temperature (ºC)', fontsize = 20);
# here we will plot the rolling mean with a 7 window day period to analyze the weekly frequency

plt.figure(figsize=(16,6))

plt.plot(total_df['totalkW_mean'].rolling(window=7,center=False).mean(),label='Rolling Mean') # analyze the rolling mean considering a 7 days window

plt.plot(total_df['totalkW_mean'].rolling(window=7,center=False).std(),label='Rolling Std Dev')# analyze the rolling std dev considering a 7 days window

plt.legend()

plt.xlabel('day')

plt.title('7 day window - Mean & Std. Deviation (kW)', fontsize=20);
N, M = 12, 6

fig, ax = plt.subplots(figsize=(N, M))

corr_plot = plot_acf(total_df['totalkW_mean'].dropna(), lags=40, ax=ax)

#plt.show()

plt.title('Autocorrelation', fontsize=20)

plt.xlabel('lag');
Consumo_WD_M = total_df[['WD','Month','totalkW_mean']].groupby(by=['WD','Month']).mean()['totalkW_mean'].unstack()

plt.figure(figsize=(14,6))

sns.heatmap(Consumo_WD_M.astype(int),cmap='viridis', annot=True, fmt='g')

plt.title("Mean (kW)", fontsize=20);
plt.figure(figsize=(12,5))

sns.boxplot(x='WD', y='totalkW_mean', data=total_df, color='#3D59AB')

plt.title("Day of the Week Mean (kW)", fontsize=20);



plt.figure(figsize=(12,5))

sns.boxplot(x='Month', y='totalkW_mean', data=total_df, color='#CD661D')

plt.title("Monthly Mean (kW)", fontsize=20);
 # this plot shows the time during the day where it has the greatest power

count=total_df[['WD','totalkW_time_max','totalkW_max']].groupby(by=['WD','totalkW_time_max']).count()['totalkW_max'].unstack()

count[np.isnan(count)] = 0

plt.figure(figsize=(13,5))

sns.heatmap(count.dropna(), cmap='viridis')

plt.title("Count of maximum power per hour", fontsize=20);
total = total_df['totalkW_time_max'].value_counts()

count  = pd.concat([total], axis=1, keys=['Count'])

count.reset_index(drop=False, inplace=True)

count.head(8)

count.sort_values(by=['index'], inplace=True, ascending=True)

plt.figure(figsize=(14,4))

sns.barplot(x='index', y='Count', data=count, color='#3D59AB')

plt.title("Count of maximum power per hour", fontsize=20);

plt.xlabel('totalkW_time_max');



#count.sort_values(by=['Count'], inplace=True, ascending=False)

#plt.figure(figsize=(14,4))

#sns.barplot(x='index', y='Count', data=count, color='#3D59AB')

#plt.xlabel('totalkW_time_max')

#plt.title("Count of maximum power per hour - descendant", fontsize=20);
plt.figure(figsize=(12,4))

sns.heatmap(total_df.groupby(by='WD')['totalkW_mean'].describe().astype(int), cmap='viridis', annot=True , fmt='g');

plt.title('TotalkW_mean description', fontsize=20);
total_df['is_weekend'] = np.where(total_df['WD'].isin(['5','6']),1,0);
plt.figure(figsize=(10,4))

sns.distplot(total_df['totalkW_mean'].loc[(total_df['is_weekend'] == 0)])

sns.distplot(total_df['totalkW_mean'].loc[(total_df['is_weekend'] == 1)])

plt.ylabel('p.u.')

plt.title('Distribution - Weekends and weekdays', fontsize=20);
# X is the independent variable - aka INPUT

# y is the dependent variable - aka OUTPUT



X = total_df.drop(['Date','Year','Day','totalkW_mean','totalkW_max','totalkW_time_max','insolation','RH'], axis=1).values

y = total_df['totalkW_mean'].values



# the model only works with arrays, not DF, that's why the ".values" on the code
# splitting the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# since there are not too many data, I thought it was better to have a bigger training set,    not the usual 2/3.



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

model_ann = Sequential()

model_ann.add(LSTM( units=4, input_shape=(1, 12), activation='sigmoid', return_sequences=True))

# first layer receiving the input. Sigmoid was choosing as an activation funcion in order to work properly for LSTM



model_ann.add(LSTM( units=2, input_shape=(1, 12), activation='linear',recurrent_activation='linear', return_sequences=False))

# the second layer work with a linear activation function to behave more in tune with our input data



model_ann.add(Dense(units=1,activation='linear'))

# finally a last layer with only one output, which is the respective mean power of the day



model_ann.compile(loss='mse', optimizer='adam')

# for the optimization model, the Minimum Squared Error is minimized as the objective function

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)

# added an early stop that is trigger when the difference of the validation loss at each epoch is minimum.
model_ann.fit(x = X_train, 

          y = y_train, 

          epochs = 8000, 

          validation_data = (X_test, y_test), 

          batch_size = 128,

          callbacks=[early_stop],

          verbose=0

         );



# 8000 epochs to train the model

# 128 cases for updating the model parameters, which is the batch size
model_loss = pd.DataFrame(model_ann.history.history)

print(model_loss.tail(1))

plt.figure(figsize=(12,6))

plt.plot(model_loss['loss'], label = 'Training loss')

plt.plot(model_loss['val_loss'], label = 'Validation loss')

plt.legend()

plt.ylabel("loss")

plt.xlabel("epoch")

plt.title("Losses - Training & Validation", fontsize=20);
pd.options.display.float_format = "{:.2f}".format



predictions = model_ann.predict(X_test)



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % total_df['totalkW_mean'].astype(float).mean())

print("y_test Mean = %.2f" % y_test.astype(float).mean())

print("y_test Std Dev = %.2f" % y_test.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(y_test, lower_p)) # retrieve observation at lower percentile

#print('y_test %.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(y_test, upper_p))

#print('y_test %.1fth percentile = %.2f' % (upper_p, upper))

print("Predictions Mean = %.2f" % predictions.astype(float).mean())

print("Predictions Std Dev = %.2f" % predictions.astype(float).std())



model_1_MAE = mean_absolute_error(y_test,predictions)

model_1_RMSE = np.sqrt(mean_squared_error(y_test,predictions))

model_1_EVS = explained_variance_score(y_test,predictions)



print("MAE = %.2f" % model_1_MAE)

print("RMSE = %.2f" % model_1_RMSE)

print("EVS = %.2f" % model_1_EVS)
plt.figure(figsize=(12,5))

plt.scatter(y_test,predictions) # Our predictions



plt.plot(y_test,y_test, 'orange') # y=x line

plt.plot(y_test,y_test+y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.plot(y_test,y_test-y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.xlabel("y_test")

plt.ylabel("Predictions")

plt.title("Scatter - y_test vs. Predictions (kW)", fontsize=20);



plt.figure(figsize=(12,5)) # Plot predictions and the real values together

sns.distplot(y_test, label='y_test', bins=30)

sns.distplot(predictions, label='Predictions', bins=30)

plt.title("Distribution Model - y_test vs. Predicted values", fontsize=20);

plt.legend()

plt.ylabel("p.u.")

plt.xlabel("kW");



plt.figure(figsize=(14,5))

plt.plot(y_test, '#CD661D', linewidth=.5, label='y_test')

plt.plot(predictions, '#3D59AB',linewidth=1, label='Predictions')

plt.title(" y_test vs. Predictions (kW)", fontsize=20);

plt.legend()

plt.ylabel("kW")

plt.xlabel("days");
y_test_pd = DataFrame(y_test)

predictions_pd = DataFrame(predictions)

comparison  = pd.concat([y_test_pd, predictions_pd], axis=1, keys=['y_test', 'predictions'])

comparison['resid'] = comparison['y_test'] -  comparison['predictions']

comparison['resid_ratio'] = (comparison['predictions'] / comparison['y_test'])

comparison['residsqr'] = comparison['resid']*comparison['resid']



plt.figure(figsize=[12,4])

sns.distplot((comparison['resid']), bins=30)

plt.title('Distribution - Residuals', fontsize=20);

plt.xlabel("kW")

plt.ylabel("p.u.");



print("Skewness = %.2f" % comparison['resid'].skew())

print("Kurtosis = %.2f" % comparison['resid'].kurt())



plt.figure(figsize=[12,4])

plt.scatter(comparison['resid'],comparison['resid_ratio'])

plt.title('Scatter - Residuals vs. Residuals Ratio', fontsize=20);

plt.xlabel("Residuals")

plt.ylabel("Residuals Ratio");
N, M = 12, 6

fig, ax = plt.subplots(figsize=(N, M))

corr_plot = plot_acf(comparison['resid'].dropna(), lags=40, ax=ax)

#plt.show()

plt.title('Autocorrelation - Residuals', fontsize=20)

plt.xlabel('lag');



N, M = 12, 6

fig, ax = plt.subplots(figsize=(N, M))

corr_plot = plot_acf(comparison['residsqr'].dropna(), lags=40, ax=ax)

#plt.show()

plt.title('Autocorrelation - Residuals Squared', fontsize=20)

plt.xlabel('lag');
X_df = total_df.drop(['Date','Day','Year','totalkW_max','totalkW_mean','insolation','totalkW_time_max','RH'], axis=1)

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

    nl4 = nl4.append(pd.Series([nl3.loc[i][0]-nl3.loc[i-1][0]] + [nl3.loc[i][0]-nl3.loc[i-7][0]] + [nl3.loc[i][0]/nl3.loc[i-1][0]] + [nl3.loc[i][0]/nl3.loc[i-7][0]] + [X_df.loc[i][4]] + [nl4.loc[i-6][5]] + [X_df.loc[i][6]] + [X_df.loc[i][7]] + [X_df.loc[i][8]] + [X_df.loc[i][9]] + [X_df.loc[i][10]] + [nl4.loc[i-6][11]],  index=nl4.columns), ignore_index=True)

    #print(nl4)

    nly = nl4.tail(1).values

    nly = scaler.transform(nly)

    nly = np.reshape(nly, (nly.shape[0], 1, nly.shape[1]))

    pred = float(model_ann.predict(nly))

    #print(pred)

    nl3 = nl3.append(pd.Series(pred, index=nl3.columns), ignore_index=True)

    j=j+1

    i=i+1
df_real = y_df['t']

df_real = df_real.tail(steps)

df_pred = nl3.tail(steps)



model_2_df = nl4

model_2_df['predictions'] = df_pred

model_2_df['real'] = df_real



# join the predictions and the real values to the used DF, so we can correlate all the dependent and independent variables
plt.figure(figsize=(10,4))

plt.plot(nl3.head(initial_values),label='Real Series')

plt.plot(nl3.tail(steps), label='Predicted Values')

plt.legend()

plt.title("Prediction Model - Application (kW)", fontsize=20)

plt.ylabel("kW")

plt.xlabel("day");
df_real = y_df['t']

df_real = df_real.tail(steps)

df_pred = nl3.tail(steps)



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % df_real.astype(float).mean())

print("Real Std Dev = %.2f" % df_real.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(df_real, lower_p)) # retrieve observation at lower percentile

#print('Real %.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(df_real, upper_p))

#print('Real %.1fth percentile = %.2f' % (upper_p, upper))

print("Predictions Mean = %.2f" % df_pred.astype(float).mean())

print("Predictions Std Dev = %.2f" % df_pred.astype(float).std())



model_1_app_MAE = mean_absolute_error(df_real,df_pred)

model_1_app_RMSE = np.sqrt(mean_squared_error(df_real,df_pred))

model_1_app_EVS = explained_variance_score(df_real,df_pred)

                           

print("MAE = %.2f" % model_1_app_MAE)

print("RMSE = %.2f" % model_1_app_RMSE)

print("EVS = %.2f" % model_1_app_EVS)
plt.figure(figsize=(12,5)) 

sns.scatterplot(x='real', y='predictions', data=model_2_df, hue='is_weekend')

plt.plot(df_real,df_real,'black')  # y=x line

plt.plot(y_test,y_test+y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.plot(y_test,y_test-y_test_std_dev,'grey', linewidth=.1) # y=x line

plt.title("Scatter Prediction Model - Real Series vs. Predicted values (kW)", fontsize=20)

plt.xlabel("Real Series")

plt.ylabel("Predictions");



plt.figure(figsize=(12,5)) # Plot predictions and the real values together

sns.distplot(df_real, label='Real Series', bins=30)

sns.distplot(df_pred, label='Predictions', bins=30)

plt.title("Distribution Prediction Model - Real Series vs. Predicted values", fontsize=20)

plt.legend()

plt.ylabel("p.u.")

plt.xlabel("kW");



plt.figure(figsize=(18,6)) # Plot predictions and the real values together

plt.plot(df_real[200:400], '#CD661D', label= 'Real Series', linewidth=.7) # get only a part of it

plt.plot(df_pred[200:400], '#3D59AB', label= 'Predictions', linewidth=1)

plt.title("Prediction Model - Real Series vs. Predicted Values - day 200 to 400 (kW)", fontsize=20)

plt.xlabel("days")

plt.ylabel("kW");

plt.legend();

model_2_df['resid'] = model_2_df['real'] -  model_2_df['predictions']

model_2_df['resid_ratio'] = (model_2_df['predictions'] / model_2_df['real'])

model_2_df['residsqr'] = model_2_df['resid']*model_2_df['resid']



plt.figure(figsize=[12,4])

sns.distplot(model_2_df['resid'].loc[(model_2_df['is_weekend'] == 0)], bins=30, label='weekday')

sns.distplot(model_2_df['resid'].loc[(model_2_df['is_weekend'] == 1)], bins=30, label='weekend')

plt.title('Distribution - Residuals', fontsize=20)

plt.xlabel("kW")

plt.xlim(-220,120)

plt.legend()

plt.ylabel("p.u.");



print("Skewness: %.2f" % model_2_df['resid'].skew())

print("Kurtosis: %.2f" % model_2_df['resid'].kurt())



plt.figure(figsize=[12,4])

sns.scatterplot(x='resid', y='resid_ratio', data=model_2_df, hue='is_weekend')

plt.title('Scatter - Residuals vs. Residuals Ratio', fontsize=20)

plt.xlabel("Residuals")

plt.xlim(-230,110)

plt.ylabel("Residuals Ratio");
# comparing the distribution of the following values



plt.figure(figsize=[18,8])

plt.subplot(221)

sns.distplot(nl4['totalkW_d/1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_d/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Total kW [d/1]", fontsize=20)

plt.xlabel("");



plt.subplot(223)

sns.distplot(nl4['totalkW_w/1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_w/1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Total kW [w/1]", fontsize=20)

plt.xlabel("");



plt.subplot(222)

sns.distplot(nl4['totalkW_d-1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_d-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Total kW [d-1]", fontsize=20)

plt.xlabel("");



plt.subplot(224)

sns.distplot(nl4['totalkW_w-1'].head(initial_values), label='Real Series')

sns.distplot(nl4['totalkW_w-1'].tail(steps), label='Predictions')

plt.legend()

plt.title("Total kW [w-1]", fontsize=20)

plt.xlabel("");
plt.figure(figsize=(12,4))

sns.lineplot(x='WD', y='real', data=model_2_df, label='Real Series')

sns.lineplot(x='WD', y='predictions', data=model_2_df, label='Predictions');

plt.title("Prediction Model - Real Series vs. Predicted Values (kW)", fontsize=20)

plt.ylabel("kW")

plt.legend();

g = sns.FacetGrid(model_2_df, col='WD', xlim=(50,450), ylim=(0,30))

g = g.map(plt.hist, 'real')

h = sns.FacetGrid(model_2_df, col='WD', xlim=(50,450), ylim=(0,30))

h = h.map(plt.hist, 'predictions')
plt.figure(figsize=(14,1))

rwd = model_2_df.pivot_table(values='real',columns='WD')

sns.heatmap(rwd.astype(int), annot=True,linecolor='white',linewidths=1, vmin=200, vmax=400, cmap='viridis', fmt='g')

plt.title("Prediction Model - Real Series (kW)", fontsize=20)

plt.ylabel(" ");



plt.figure(figsize=(14,1));

pwd = model_2_df.pivot_table(values='predictions',columns='WD')

sns.heatmap(pwd.astype(int), annot=True,linecolor='white',linewidths=1, vmin=200, vmax=400, cmap='viridis', fmt='g')

plt.title("Prediction Model - Predictions (kW)", fontsize=20)

plt.ylabel(" ");
Pred_Power_WD_M = model_2_df[['WD','Month','predictions']].groupby(by=['WD','Month']).mean()['predictions'].unstack()

Consumo_WD_M = total_df[['WD','Month','totalkW_mean']].groupby(by=['WD','Month']).mean()['totalkW_mean'].unstack()

plt.figure(figsize=(14,6))

comp=(Pred_Power_WD_M/Consumo_WD_M)

sns.heatmap(comp,cmap='viridis', annot=True, vmin=0.8, vmax=1.2)

plt.title("Prediction Model - Predictions / Real Series", fontsize=20);

plt.figure(figsize=(14,1))

sns.heatmap(model_2_df.corr()[-4:-3:18], annot=True,linecolor='white',linewidths=1, cmap='viridis')

plt.title("Real Series - Correlations", fontsize=20);



plt.figure(figsize=(14,1))

sns.heatmap(model_2_df.corr()[-5:-4:18], annot=True,linecolor='white',linewidths=1, cmap='viridis')

plt.title("Prediction Model - Correlations", fontsize=20);
# Models

from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

from xgboost import XGBRegressor



# Metrics and Grid Search

from sklearn import model_selection, metrics

from sklearn.model_selection import GridSearchCV
X = total_df.drop(['Date','Year','Day','totalkW_mean','totalkW_max','totalkW_time_max','insolation','RH'], axis=1).values

y = total_df['totalkW_mean'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)



scaler = MinMaxScaler()

scaler.fit(X_train) 

X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test) 
# Creating a predefined function to test the models

def modelfit(model):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MAE = %.2f" % mean_absolute_error(y_test,preds))

    print("RMSE = %.2f" % np.sqrt(mean_squared_error(y_test,preds)))

    print("EVS = %.2f" % explained_variance_score(y_test,preds));
# Linear Regression



lm = LinearRegression(n_jobs = 10000)

modelfit(lm)
# Random Forest Regressor



rf = RandomForestRegressor(n_jobs = 1000, verbose=0)

modelfit(rf)
# XGBoost

xg = XGBRegressor(learning_rate=0.1, n_estimators=5000)

modelfit(xg)
# Decision Tree

dt = DecisionTreeRegressor()

modelfit(dt)
# Bayesian Linear Model

br = BayesianRidge(alpha_1=1e-06, alpha_2=0.5, compute_score=False, copy_X=True,

              fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=10,

              normalize=False, tol=0.1, verbose=False)

modelfit(br)
# Lasso Lars

ls = LassoLars()

modelfit(ls)
#apply an array of learning rate values in order to get the one with the lowest error

import numpy as np

size=100

lr = np.linspace(0.01,1,size)

j = 0

MAE = np.zeros(size)

RMSE = np.zeros(size)

EVS = np.zeros(size)



for j in range(0,len(lr)-1):

    xg = XGBRegressor(learning_rate=lr[j], n_estimators=5000)

    xg.fit(X_train, y_train)

    preds = xg.predict(X_test)

    MAE[j] = mean_absolute_error(y_test,preds)

    RMSE[j] = np.sqrt(mean_squared_error(y_test,preds))

    EVS[j] = explained_variance_score(y_test,preds)



xg_df = pd.DataFrame(MAE[:-1], columns = ['MAE']) 

xg_df['RMSE'] = RMSE[:-1]

xg_df['EVS'] = EVS[:-1]

xg_df['lr'] = lr[:-1]
xg_df.sort_values(by=['MAE'], inplace=True, ascending=True)



model_alt_MAE = xg_df['MAE'].min()

model_alt_RMSE = xg_df['RMSE'].min()

model_alt_EVS = xg_df['EVS'].max()



print("MAE = %.2f" % model_alt_MAE)

print("RMSE = %.2f" % model_alt_RMSE)

print("EVS = %.2f" % model_alt_EVS)

print("learning rate = %.2f" % xg_df['lr'].head(1))

lr_ideal = xg_df['lr'].head(1).astype(float).values

xg_df.head(3)
alt_model = XGBRegressor(learning_rate=0.14, n_estimators=5000)

alt_model.fit(X_train, y_train)
X_df = total_df.drop(['Date','Day','Year','totalkW_max','totalkW_mean','insolation','totalkW_time_max','RH'], axis=1)

X_df = X_df.reset_index(drop=True)



y_df = total_df['totalkW_mean']

y_df = pd.concat([y_df], axis=1, keys="total")

y_df = y_df.reset_index(drop=True)



initial_values = 30



nl1 = y_df.head(initial_values)

nl2 = X_df.head(initial_values)
i=len(nl1)-1

j=1

pred=0

steps=564

for j in range(1,steps):

    nl2 = nl2.append(pd.Series([nl1.loc[i][0]-nl1.loc[i-1][0]] + [nl1.loc[i][0]-nl1.loc[i-7][0]] + [nl1.loc[i][0]/nl1.loc[i-1][0]] + [nl1.loc[i][0]/nl1.loc[i-7][0]] + [X_df.loc[i][4]] + [nl2.loc[i-6][5]] + [X_df.loc[i][6]] + [X_df.loc[i][7]] + [X_df.loc[i][8]] + [X_df.loc[i][9]] + [X_df.loc[i][10]] + [nl2.loc[i-6][11]],  index=nl2.columns), ignore_index=True)

    #print(nl4)

    nly = nl2.tail(1).values

    nly = scaler.transform(nly)

    #nly = np.reshape(nly, (nly.shape[0], 1, nly.shape[1]))

    pred = float(alt_model.predict(nly))

    #print(pred)

    nl1 = nl1.append(pd.Series(pred, index=nl3.columns), ignore_index=True)

    j=j+1

    i=i+1

df_real_alt = y_df['t']

df_real_alt = df_real.tail(steps)

df_pred_alt = nl1.tail(steps)



model_alt_df = nl2

model_alt_df['predictions'] = df_pred_alt

model_alt_df['real'] = df_real_alt



y_test_std_dev=y_test.astype(float).std()

print("Real Mean = %.2f" % df_real_alt.astype(float).mean())

print("Real Std Dev = %.2f" % df_real_alt.astype(float).std())



alpha = 5.0

lower_p = alpha / 2.0 # calculate lower percentile (e.g. 2.5)

lower = max(0.0, percentile(df_real_alt, lower_p)) # retrieve observation at lower percentile

#print('Real %.1fth percentile = %.2f' % (lower_p, lower)) # calculate upper percentile (e.g. 97.5)

upper_p = (100 - alpha) + (alpha / 2.0) # retrieve observation at upper percentile

upper = max(1.0, percentile(df_real_alt, upper_p))

#print('Real %.1fth percentile = %.2f' % (upper_p, upper))

print("Predictions Mean = %.2f" % df_pred_alt.astype(float).mean())

print("Predictions Std Dev = %.2f" % df_pred_alt.astype(float).std())



model_alt_app_MAE = mean_absolute_error(df_real_alt,df_pred_alt)

model_alt_app_RMSE = np.sqrt(mean_squared_error(df_real_alt,df_pred_alt))

model_alt_app_EVS = explained_variance_score(df_real_alt,df_pred_alt)

                             

print("MAE = %.2f" % model_alt_app_MAE)

print("RMSE = %.2f" % model_alt_app_RMSE)

print("EVS = %.2f" % model_alt_app_EVS)
plt.figure(figsize=(18,6)) # Plot predictions and the real values together

plt.plot(df_real_alt[200:400], '#CD661D', label= 'Real Series', linewidth=.7)

plt.plot(df_pred_alt[200:400], '#3D59AB', label= 'Predictions', linewidth=1)

plt.title("Prediction Model - Real Series vs. Predicted Values (kW)", fontsize=20)

plt.xlabel("days")

plt.ylabel("kW");

plt.legend();
model_alt_df['resid'] = model_alt_df['real'] -  model_alt_df['predictions']

model_alt_df['resid_ratio'] = (model_alt_df['predictions'] / model_alt_df['real'])

model_alt_df['residsqr'] = model_alt_df['resid']*model_alt_df['resid']



plt.figure(figsize=[12,4])

sns.distplot(model_alt_df['resid'].loc[(model_alt_df['is_weekend'] == 0)], bins=30, label='weekday')

sns.distplot(model_alt_df['resid'].loc[(model_alt_df['is_weekend'] == 1)], bins=30, label='weekend')

plt.title('Distribution - Residuals', fontsize=20)

plt.xlabel("kW")

plt.xlim(-220,120)

plt.legend()

plt.ylabel("p.u.");



print("Skewness: %.2f" % model_alt_df['resid'].skew())

print("Kurtosis: %.2f" % model_alt_df['resid'].kurt())



plt.figure(figsize=[12,4])

sns.scatterplot(x='resid', y='resid_ratio', data=model_alt_df, hue='is_weekend')

plt.title('Scatter - Residuals vs. Residuals Ratio', fontsize=20)

plt.xlabel("Residuals")

plt.xlim(-230,110)

plt.ylabel("Residuals Ratio");
plt.figure(figsize=(18,6)) # Plot predictions and the real values together

plt.plot(df_real[250:350], '#A9A9A9', label= 'Real Series', linewidth=.3, marker='o', markersize=5)

plt.plot(df_pred[250:350], '#3D59AB', label= 'Predictions RNN LSTM', linewidth=.5, marker='o', markersize=5)

plt.plot(df_pred_alt[250:350], '#CD661D', label= 'Predictions XGBoost', linewidth=.5, marker='o', markersize=5)

plt.title("Prediction Model Comparison - Real Series vs. Predicted Values - day 250 to 350 (kW)", fontsize=20)

plt.xlabel("days")

plt.ylabel("kW");

plt.legend();
plt.figure(figsize=(12,5)) # Plot predictions and the real values together

sns.distplot(df_real, label='Real Series', bins=40, color='#A9A9A9')

sns.distplot(df_pred, label='RNN LSTM', bins=40, color='#3D59AB')

sns.distplot(df_pred_alt, label='XGBoost', bins=40, color='#CD661D')



plt.title("Distribution Prediction Model - Real Series vs. Predicted values", fontsize=20)

plt.legend()

plt.ylabel("p.u.")

plt.xlabel("kW");
plt.figure(figsize=[12,4])

sns.distplot(model_alt_df['resid'], bins=40, label='XGBoost')

sns.distplot(model_2_df['resid'], bins=40, label='RNN LSTM')



plt.title('Distribution Model Comparison - Residuals', fontsize=20)

plt.xlabel("kW")

plt.xlim(-220,120)

plt.legend()

plt.ylabel("p.u.");



print("Skewness RNN+LSTM = %.2f" % model_2_df['resid'].skew())

print("Kurtosis RNN+LSTM = %.2f" % model_2_df['resid'].kurt())

print("Skewness XGBoost = %.2f" % model_alt_df['resid'].skew())

print("Kurtosis XGBoost = %.2f" % model_alt_df['resid'].kurt())
Model_alt_app = pd.DataFrame([model_alt_app_MAE, model_alt_app_RMSE, model_alt_app_EVS], columns = ['Model_alt_app'])

Model_alt = [model_alt_MAE, model_alt_RMSE, model_alt_EVS]

model_1_app = [model_1_app_MAE, model_1_app_RMSE, model_1_app_EVS ]

model_1 = [model_1_MAE, model_1_RMSE, model_1_EVS ]



Model_perf_df = pd.DataFrame(['RNN+LSTM',  'XGBoost','RNN+LSTM Applied', 'XGBoost Applied' ], columns = ['Model'])

Model_perf_df['MAE'] = [model_1_MAE, model_alt_MAE,model_1_app_MAE,model_alt_app_MAE ]

Model_perf_df['RMSE'] = [model_1_RMSE, model_alt_RMSE, model_1_app_RMSE, model_alt_app_RMSE ]

Model_perf_df['EVS'] = [model_1_EVS, model_alt_EVS,model_1_app_EVS, model_alt_app_EVS ]



Model_perf_df