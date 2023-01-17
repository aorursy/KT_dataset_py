import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression



sns.set()
df_covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
cols = df_covid.columns

df_covid.columns = [col.lower() for col in cols]
df_covid.rename(columns={

    'observationdate' : 'observation_date',

    'country/region' : 'country',

    'province/state' : 'province_state', 

    'last update' : 'last_update',

}, inplace=True)
df_covid['observation_date'] = pd.to_datetime(df_covid['observation_date'])



df_covid.sort_values('observation_date', inplace=True)
df_covid['diseased'] = df_covid['confirmed'] - df_covid['recovered'] - df_covid['deaths']



df_series = df_covid.groupby('observation_date').agg({

    'country' : 'nunique',

    'confirmed' : 'sum',

    'deaths' : 'sum',

    'recovered' : 'sum',

    'diseased' : 'sum',

})
df_covid.drop(['sno', 'last_update'], axis=1, inplace=True)
for i in range(7, 15):

    df_series[f'confirmed_lag_{i}'] = df_series['confirmed'].shift(i)

    df_series[f'deaths_lag_{i}'] = df_series['deaths'].shift(i)

    df_series[f'recovered_lag_{i}'] = df_series['recovered'].shift(i)

    df_series[f'diseased_lag_{i}'] = df_series['diseased'].shift(i)
sns.set(style="white")



fig, ax = plt.subplots(figsize=(11, 9))



# Create correlation matrix

corr = df_series.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)



# Plot correlation matrix without the diagonal and upper part

sns.heatmap(corr, mask=mask, cmap=cmap, linewidths=.5)
sns.set()



# Get percentages of recovered, deaths and diseased

df_series['pct_recovered'] = round(df_series['recovered'] / df_series['confirmed'], 4)

df_series['pct_deaths'] = round(df_series['deaths'] / df_series['confirmed'], 4)

df_series['pct_diseased'] = round(df_series['diseased'] / df_series['confirmed'], 4)
df_series['country_pct_change'] = df_series['country'].pct_change()

df_series['recovered_pct_change'] = df_series['recovered'].pct_change()

df_series['deaths_pct_change'] = df_series['deaths'].pct_change()

df_series['diseased_pct_change'] = df_series['diseased'].pct_change()
df_country = df_covid.groupby(['country', 'observation_date']).sum().reset_index()



df_country['confirmed_log'] = np.log10(df_country['confirmed'])

df_country['deaths_log'] = np.log10(df_country['deaths'])
df_country['day_cnt'] = 0 

for country in df_country['country'].unique():

    day_cnt = [i for i in range(1, df_country[df_country['country'] == country][df_country['confirmed'] > 0].shape[0] + 1)]

    

    df_country.loc[(df_country['country'] == country) & (df_country['confirmed'] > 0) , 'day_cnt'] = day_cnt
# Column to plot

target = 'confirmed_log'



# It'd be better if this dataframe was outside this cell, but that's fine

countries_to_plot = df_country.groupby('country').sum().sort_values(target, ascending=False).index[:10]



# Plot top 10

plt.figure(figsize=(13, 11))

sns.lineplot(x='day_cnt', y=target, hue='country', data=df_country[df_country[target] > 0][df_country['country'].isin(countries_to_plot)])



plt.show()
# Column to plot

target = 'deaths'



# It'd be better if this dataframe was outside this cell, but that's fine

countries_to_plot = df_country.groupby('country').sum().sort_values(target, ascending=False).index[:10]



# Plot top 10

plt.figure(figsize=(13, 11))

sns.lineplot(x='day_cnt', y=target, hue='country', data=df_country[df_country[target] > 0][df_country['country'].isin(countries_to_plot)])



plt.show()
df_series[['deaths_pct_change', 'country_pct_change']].plot(figsize=(14, 5))
df_series.dropna().shape
train_cols = [col for col in df_series.columns if 'deaths_lag_' in col] 



# Just to see if i'm not cheating ;D

', '.join(train_cols) 
num_split = 7



X = np.log10(df_series.dropna()[train_cols])

y = np.log10(df_series.dropna()['deaths'])



X_train = X[:-num_split]

y_train = y[:-num_split]

X_test = X[-num_split:]

y_test = y[-num_split:]
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)



df_predictions = pd.DataFrame()

df_predictions['y_pred_log'] = predictions

df_predictions['y_true_log'] = y_test.values

df_predictions['y_pred'] = 10 ** predictions

df_predictions['y_true'] = 10 ** y_test.values



df_predictions['absolute_pct_error'] = abs((df_predictions['y_pred'] - df_predictions['y_true']) / df_predictions['y_true']) * 100
f"MAPE: {round(df_predictions['absolute_pct_error'].mean())}%"
fig, ax = plt.subplots(figsize=(14, 5))



ax.plot(y_train, 'bo--')

ax.plot(y_test, 'go--')

ax.plot(pd.Series(predictions, index = y_test.index), 'ro--')



plt.title('Log10 values')

plt.show()
fig, ax = plt.subplots(figsize=(14, 5))



ax.plot(10 ** y_train, 'bo--')

ax.plot(10 ** y_test, 'go--')

ax.plot(10 ** pd.Series(predictions, index = y_test.index), 'ro--')



plt.title('Absolute values')

plt.show()
df_predictions['y_pred'] = round(df_predictions['y_pred'])

df_predictions