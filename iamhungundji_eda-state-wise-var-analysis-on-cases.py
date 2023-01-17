import matplotlib.pyplot as plt

import seaborn as sns

import plotly.io as pio

import plotly.graph_objects as go

import plotly.express as px

%matplotlib inline

import numpy as np

import pandas as pd



from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.api import VAR



from sklearn.metrics import mean_squared_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/covid-tracking-project-racial-data-tracker/Race Data Entry - CRDT.csv')

data.head()
data.fillna(0, inplace=True)

data['Cases_Total'] = data['Cases_Total'].str.replace(',','')

data['Cases_Total'] = data['Cases_Total'].astype('float64')

data['Cases_White'] = data['Cases_White'].str.replace(',','')

data['Cases_White'] = data['Cases_White'].astype('float64')
state_date = data[['State','Date']]

state_date.head()
cases_data = data[data.columns[pd.Series(data.columns).str.startswith('Cases_')]]

cases_data = pd.concat([cases_data, state_date], axis=1)

cases_data['Date'] = pd.to_datetime(cases_data['Date'], format='%Y%m%d')

cases_data = cases_data.sort_values(by='Date', ascending=True).reset_index(drop=True)

cases_data.index = cases_data.Date

cases_data = cases_data.drop('Date', axis=1)

cases_data.head()
latest_cases = pd.DataFrame(cases_data.groupby('State')['Cases_Total'].max()).reset_index()

fig = px.choropleth(latest_cases,

                    locations="State",

                    color="Cases_Total",

                    hover_name="State",

                    locationmode = 'USA-states')

fig.update_layout(

    title_text = 'Total Cases in all States',

    geo_scope='usa',

)

fig.show()
daily_total_cases = cases_data.reset_index().dropna()

daily_total_cases['Date'] = daily_total_cases['Date'].astype('str')

fig = px.scatter_geo(daily_total_cases, 

                     locations="State", 

                     color="Cases_Black", 

                     hover_name="State", 

                     size="Cases_Black",

                     animation_frame="Date", 

                     locationmode = 'USA-states')

fig.update_layout(

    title_text = 'Total Daily Cases in all States',

    geo_scope='usa',

)

fig.show()
fig = px.pie(cases_data.groupby('State').sum()[['Cases_Total']], 

             values='Cases_Total', names=cases_data.groupby('State').sum().index,

             title='Total Cases in all States',

             hover_data=['Cases_Total'])

fig.update_traces(textposition='inside', textinfo='percent+label',showlegend=False)

fig.show()
cases_data.groupby('State').mean()
state_ = 'OR'

state = cases_data[cases_data['State'] == state_].drop('State', axis=1)



state_pc = pd.DataFrame()

state_pc['Cases'] = state.columns[1:]

state_pc['Ratio'] = np.round(state.sum()[1:]/np.sum(state.sum()[1:]) * 100, 2).values

state_pc = state_pc.sort_values(by='Ratio', ascending=False).reset_index(drop=True)



plt.figure(figsize=(12, 6))

for col in state.columns[1:]:

    plt.plot(state[col], label=col)

plt.title('Cases in State: '+state_)

plt.legend()

plt.show()

    

state_pc.style.background_gradient()
plt.figure(figsize=(14,7))

ax = sns.heatmap(state[state.columns[1:]].corr(), annot=True, cmap="mako")

plt.title('Correlation Matrix of various cases in state '+state_)

plt.show()
melt = pd.melt(state[state.columns[1:]].reset_index(), id_vars=['Date'])

melt.dropna(inplace=True)

melt['Date'] = melt['Date'].astype('str')

melt = melt.rename({"variable":'Cases_Type', 'value':'Cases'}, axis=1)

fig = px.bar(melt, x='Cases', y='Cases_Type', color='Cases',

             animation_frame="Date", height=850, orientation='h',

             title="Detailed COVID-19 Daily Cases Analysis of all types in state "+state_)

fig.show()
state_sum = pd.DataFrame(state.sum()[1:]).reset_index().rename({'index':'Type', 0:'Cases'}, axis=1)

fig = px.pie(state_sum,

             values='Cases', names='Type',

             title='Ratio of cases type in state '+state_)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
def adfuller_test(series, signif=0.05, name='', verbose=False):

    r = adfuller(series, autolag='AIC')

    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}

    p_value = output['pvalue']

    def adjust(val, length= 6): return str(val).ljust(length)



    # Print Summary

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)

    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')

    print(f' Significance Level    = {signif}')

    print(f' Test Statistic        = {output["test_statistic"]}')

    print(f' No. Lags Chosen       = {output["n_lags"]}')



    for key,val in r[4].items():

        print(f' Critical value {adjust(key)} = {round(val, 3)}')



    if p_value <= signif:

        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")

        print(f" => Series is Stationary.")

    else:

        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")

        print(f" => Series is Non-Stationary.")

    return



def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

    for c in df.columns:

        for r in df.index:

            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)

            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]

            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')

            min_p_value = np.min(p_values)

            df.loc[r, c] = min_p_value

    df.columns = [var + '_x' for var in variables]

    df.index = [var + '_y' for var in variables]

    return df
df = pd.DataFrame(state[state.columns[1:]])

df.fillna(0, inplace=True)

df = df.loc[:, (df != 0).any(axis=0)]



print("Columns: ", df.columns.tolist())
print("Plot for each variables")



nrow = 6

ncol = 2

k = 0

cols = df.columns



fig = plt.figure(figsize=(20, 19))

for i in range(1, len(cols)+1):

    plt.subplot(nrow, ncol, i)

    plt.plot(df.index, df[cols[k]])

    plt.title(cols[k])

    k = k + 1

plt.tight_layout()

plt.show()
print("Grangers Casualty Test")



maxlag = 4

test = 'ssr_chi2test'



grangers_causation_matrix(df, variables = df.columns)
nobs = 7

df_train, df_test = df[0:-nobs], df[-nobs:]

df_train = df_train.loc[:, (df_train != 0).any(axis=0)]

df_test = df_test[df_train.columns]

cols = df_train.columns

#df_train = df

print(df_train.shape)

print(df_test.shape)
for name, column in df.iteritems():

    adfuller_test(column, name=column.name)

    print('\n')
model = VAR(df_train.reset_index(drop=True))

model_fitted = model.fit(maxlag)

model_fitted.summary()
lag_order = model_fitted.k_ar



print("Lag Order: ",lag_order)



forecast_input = df_train.values[-lag_order:]

forecast_input



fc = model_fitted.forecast(y=forecast_input, steps=nobs)



df_result = pd.concat([df_test, pd.DataFrame(fc, 

                                             index=df_test.index, 

                                             columns=['Predicted_'+str(i) for i in range(df_test.shape[1])])], 

                      axis=1)



df_result
n = 0

for col in df_test.columns.tolist():

    rmse = np.sqrt(mean_squared_error(df_result[col], df_result['Predicted_'+str(n)]))

    print("RMSE of variable", col, ": ",np.round(rmse, 4))

    n = n + 1
k = 0

n = 0

fig = plt.figure(figsize=(22, 20))

for i in range(1, len(cols)+1):

    z = np.polyfit(df_train.reset_index(drop=True).index, df_train[cols[k]], 1)

    p = np.poly1d(z)

    trend = pd.DataFrame(p(df_train.reset_index(drop=True).index), 

                         columns=['Trend'], index=df_train.index)

    plt.subplot(nrow, ncol, i)

    plt.plot(df_train[cols[k]], label='Train')

    plt.plot(df_result[cols[k]], label='Validation')

    plt.plot(df_result['Predicted_'+str(n)], label='VAR Forecasting')

    plt.plot(trend['Trend'], label='Trendline')

    plt.legend()

    plt.title(cols[k])

    n = n + 1

    k = k + 1

plt.tight_layout()

plt.show()