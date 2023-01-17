import os, gc, pickle, copy, datetime, warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn import metrics

from IPython.display import Image

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 999)

warnings.filterwarnings('ignore')
df_test = pd.read_csv("../input/my-covid-pred/test_week2.csv")

df_test.head()
df_week4 = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

df_week4.head()
df_test = pd.merge(df_test, df_week4, on=['Province_State', 'Country_Region', 'Date'], how='left')

df_test.head()
df_test['Date'] = pd.to_datetime(df_test['Date'])

df_test['day'] = df_test['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_week4['Date'] = pd.to_datetime(df_week4['Date'])

df_week4['day'] = df_week4['Date'].apply(lambda x: x.dayofyear).astype(np.int16)

df_test.head()
# concat Country/Region and Province/State

def func(x):

    try:

        x_new = x['Country_Region'] + "/" + x['Province_State']

    except:

        x_new = x['Country_Region']

    return x_new

        

df_test['place_id'] = df_test.apply(lambda x: func(x), axis=1)

df_week4['place_id'] = df_week4.apply(lambda x: func(x), axis=1)

df_test.head()
tmp = df_test[pd.isna(df_test['ConfirmedCases'])==False]['Date'].max()

print("last day of existing true data: {}".format(tmp))
df_oscii_fix = pd.read_csv("../input/my-covid-pred/submission_osciiart2_fixed.csv") # my final submission with bug

df_beluga = pd.read_csv("../input/my-covid-pred/submission_beluga.csv")

df_kaz = pd.read_csv("../input/my-covid-pred/submission_Kaz.csv")

df_vopani = pd.read_csv("../input/my-covid-pred/submission_Vopani.csv")

df_rapid = pd.read_csv("../input/my-covid-pred/submission_rapids.ai KGMON.csv")



df_oscii_fix.head()
day_before_private = 92

# list of places

places_sort = df_test[['place_id', 'ConfirmedCases']][df_test['day']==day_before_private]

places_sort = places_sort.sort_values('ConfirmedCases', ascending=False).reset_index(drop=True)['place_id'].values

print(len(places_sort))

for i, place in enumerate(places_sort):

    print(i, place)
def show_graph(place):

    sns.set()

    sns.set_style('ticks')

    fig, ax = plt.subplots(figsize = (15,6)) 

    plt.subplot(1,2,1)

    x_pred = df_test[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Date'].values

    y_oscii_fix = df_oscii_fix[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values

    y_beluga = df_beluga[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values

    y_kaz = df_kaz[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values

    y_vopani = df_vopani[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values

    y_rapid = df_rapid[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values



    x_true= df_week4[(df_week4['place_id']==place)]['Date'].values

    y_true = df_week4[(df_week4['place_id']==place)]['ConfirmedCases'].values



    ax = sns.lineplot(x=x_pred, y=y_oscii_fix, label='Oscii fix')

    #     ax.set(yscale="log")

    sns.lineplot(x=x_pred, y=y_beluga, label='Beluga')

    sns.lineplot(x=x_pred, y=y_kaz, label='Kaz')

    sns.lineplot(x=x_pred, y=y_vopani, label='Vopani')

    sns.lineplot(x=x_pred, y=y_rapid, label='Rapid')

    sns.lineplot(x=x_true, y=y_true, label='true')

    plt.ylim(-1, y_true.max()*2)

    plt.title("{}/ConfirmedCases".format(place))



    fig.autofmt_xdate()

    plt.subplot(1,2,2)

    x_pred = df_test[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Date'].values



    y_oscii_fix = df_oscii_fix[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values

    y_beluga = df_beluga[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values

    y_kaz = df_kaz[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values

    y_vopani = df_vopani[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values

    y_rapid = df_rapid[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values

    x_true= df_week4[(df_week4['place_id']==place)]['Date'].values

    y_true = df_week4[(df_week4['place_id']==place)]['Fatalities'].values

    ax = sns.lineplot(x=x_pred, y=y_oscii_fix, label='Oscii fix')

    #     ax.set(yscale="log")

    sns.lineplot(x=x_pred, y=y_beluga, label='Beluga')

    sns.lineplot(x=x_pred, y=y_kaz, label='Kaz')

    sns.lineplot(x=x_pred, y=y_vopani, label='Vopani')

    sns.lineplot(x=x_pred, y=y_rapid, label='Rapid')

    sns.lineplot(x=x_true, y=y_true, label='true')

    plt.ylim(-1, y_true.max()*2)

    plt.title("{}/Fatalities".format(place))

    fig.autofmt_xdate()

    plt.show()
show_graph(places_sort[0])

show_graph(places_sort[1])

show_graph(places_sort[2])

show_graph(places_sort[3])

show_graph(places_sort[4])

show_graph(places_sort[10])

show_graph(places_sort[20])

show_graph(places_sort[50])

show_graph(places_sort[100])

show_graph(places_sort[150])

show_graph(places_sort[200])

show_graph("Japan")

show_graph("Korea, South")
country_data = pd.read_csv('../input/countryinfo/covid19countryinfo.csv')
df_test.loc[df_test.ConfirmedCases < 0,'ConfirmedCases' ] = 0

df_test.loc[df_test.Fatalities < 0,'Fatalities' ] = 0
evaluate_period = (pd.isna(df_test['ConfirmedCases'])==False) & (df_test['day']>day_before_private)
df_test2 = df_test[evaluate_period]
df_test2['predicted_cases'] = df_oscii_fix[evaluate_period]['ConfirmedCases'].values

df_test2['predicted_fatalities'] = df_oscii_fix[evaluate_period]['Fatalities'].values
country_data = country_data.rename(columns={"region": "Province_State", "country": "Country_Region"})

df_test2 = df_test2.merge(country_data)
df_test2['case_square_error'] = (np.log(df_test2['predicted_cases'].values[:]+1) - np.log(df_test2['ConfirmedCases'].values.clip(0, 1e10)+1))**2

df_test2['fatalities_square_error'] = (np.log(df_test2['predicted_fatalities'].values[:]+1) - np.log(df_test2['Fatalities'].values.clip(0, 1e10)+1))**2

df_test2['case_absolute_error'] = (np.log(df_test2['predicted_cases'].values[:]+1) - np.log(df_test2['ConfirmedCases'].values.clip(0, 1e10)+1))

df_test2['fatalities_absolute_error'] = (np.log(df_test2['predicted_fatalities'].values[:]+1) - np.log(df_test2['Fatalities'].values.clip(0, 1e10)+1))



df_test2['pop'] = df_test2['pop'].str.replace(',', '').astype(float)

df_test2['gdp2019'] = df_test2['gdp2019'].str.replace(',', '').astype(float)
df_test2.head()
out_df = df_test2[~pd.isna(df_test2['pop'])].groupby('Country_Region').mean()[['case_square_error','fatalities_square_error','pop','gdp2019']]
sns.regplot(np.log(out_df['pop'].values),out_df['case_square_error'],robust=True)

plt.xlabel('log(population)')

plt.ylim([0,0.2])
sns.regplot(np.log(out_df['pop'].values),out_df['fatalities_square_error'],robust=True)

plt.xlabel('log(population)')

plt.ylim([0,0.2])
sns.regplot(np.log(out_df['gdp2019'].values),out_df['case_square_error'],robust=True)

plt.xlabel('log(GDP 2019)')

plt.ylim([0,0.2])
sns.regplot(np.log(out_df['gdp2019'].values),out_df['fatalities_square_error'],robust=True)

plt.xlabel('log(GDP 2019)')

plt.ylim([0,0.2])
df_test2.groupby('Country_Region')['case_square_error'].mean().sort_values()
df_test2.groupby('Country_Region')['case_absolute_error'].mean().sort_values()
df_test2.groupby('Country_Region')['fatalities_square_error'].mean().sort_values()
df_test2.groupby('Country_Region')['fatalities_absolute_error'].mean().sort_values()
Image('../input/temp-image/shap.png')