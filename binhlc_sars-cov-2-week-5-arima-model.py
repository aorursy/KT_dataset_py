# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_test = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")

df_train = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")

df_sub = pd.read_csv("../input/covid19-global-forecasting-week-5/submission.csv")
def CreateGeographyFeature(df):

    df["geography"] = df.Country_Region + "_" + df.Province_State + "_" + df.County

    df.loc[df.County.isna(), "geography"] = df[df.County.isna()].Country_Region + "_" + df[df.County.isna()].Province_State

    df.loc[df.Province_State.isna(), "geography"] = df[df.Province_State.isna()].Country_Region

    return df

df_train = CreateGeographyFeature(df_train)

df_test = CreateGeographyFeature(df_test)
!pip install pmdarima
import pmdarima as pm
from tqdm.notebook import tqdm

import warnings



QUANTILE = [0.05, 0.5, 0.95]

period = df_test[df_test.Date > df_train.Date.max()]['Date'].unique().shape[0]

min_test_date = df_test.Date.min()



df_sub_all = pd.DataFrame()



with warnings.catch_warnings():

    warnings.filterwarnings("ignore")

    for country in tqdm(df_train.Country_Region.unique()):

    #for country in ['Andorra']:    

        df_country = df_train[df_train.Country_Region == country].groupby(['Date','Target']).sum()[['TargetValue']].reset_index()



        y = df_country.loc[df_country['Target'] == 'ConfirmedCases']['TargetValue'].values

       

        df_country_cc = pd.DataFrame()

        for alpha in QUANTILE:

            # ConfirmedCases

            model = pm.auto_arima(y, alpha=alpha, suppress_warnings=True, seasonal=False, error_action="ignore")

            y_pred = model.predict(n_periods=period, exogenous=None, return_conf_int=False)

            # By Country           

            df_country_alpha = pd.DataFrame({'Date':df_test[df_test.Date > df_train.Date.max()]['Date'].unique(),'TargetValue':y_pred})

            df_country_alpha['Country_Region'] = country

            df_country_alpha['Target'] = 'ConfirmedCases'

            df_country_alpha['Quantile'] = alpha

            df_country_cc = df_country_cc.append(df_country_alpha)

            

        y = df_country.loc[df_country['Target'] == 'Fatalities']['TargetValue'].values

        

        df_country_ft = pd.DataFrame()

        for alpha in QUANTILE:

            model = pm.auto_arima(y, alpha=alpha, suppress_warnings=True, seasonal=False, error_action="ignore")

            y_pred = model.predict(n_periods=period, exogenous=None, return_conf_int=False)

            # By Country           

            df_country_alpha = pd.DataFrame({'Date':df_test[df_test.Date > df_train.Date.max()]['Date'].unique(),'TargetValue':y_pred})

            df_country_alpha['Country_Region'] = country

            df_country_alpha['Target'] = 'Fatalities'

            df_country_alpha['Quantile'] = alpha

            df_country_ft = df_country_ft.append(df_country_alpha)            



        df_country_pred = pd.concat([df_country_cc,df_country_ft])

        df_country_pred.columns = ['Date','CountryValue','Country_Region','Target','Quantile']



        df_geography_pred = df_test.loc[(df_test['Country_Region'] == country) & (df_test.Date > df_train.Date.max())][['ForecastId','Date','Target','geography','Country_Region']]

        df_weight = df_train[(df_train.Country_Region == country) & (df_train.Date > '2020-05-01')].groupby(['Target','geography'])[['TargetValue']].sum()

        df_weight = df_weight / df_weight.groupby(level='Target').sum()

        # fix sum = 0

        

        df_weight = df_weight.reset_index()

        df_weight.columns = ['Target','geography','Weight']

        df_weight['Weight'] = df_weight['Weight'].fillna(0)



        df_sub = pd.merge(df_geography_pred, df_country_pred, how='left', on=['Date','Country_Region','Target'])

        df_sub = pd.merge(df_sub,df_weight, how = 'left', on = ['geography','Target'])

        df_sub['TargetValue'] = df_sub['CountryValue'] * df_sub['Weight']

        #Merge with train data

        df_sub_train_alpha = pd.merge(df_test[df_test.Country_Region == country],df_train[df_train.Country_Region == country][['geography','Date','Target','TargetValue']],how = 'inner', on = ['geography','Date','Target'])

        df_sub_train = pd.DataFrame()

        for alpha in QUANTILE:

            df_sub_train_alpha['Quantile'] = alpha

            df_sub_train = df_sub_train.append(df_sub_train_alpha)

        

        df_sub = pd.concat([df_sub_train[['ForecastId','Quantile','TargetValue']],df_sub[['ForecastId','Quantile','TargetValue']]])

        df_sub_all = df_sub_all.append(df_sub)

df_sub_all.loc[df_sub_all['TargetValue']<0,'TargetValue'] = 0

df_sub_all['ForecastId_Quantile'] = df_sub_all['ForecastId'].astype(str) + '_' + df_sub_all['Quantile'].astype(str)

df_sub_all[['ForecastId_Quantile','TargetValue']].to_csv("submission.csv", index = False)
from bokeh.plotting import figure, show, output_notebook

from bokeh.models import NumeralTickFormatter

from bokeh.palettes import Spectral11

output_notebook()



def plotCountry(country):

    df_country = pd.merge(left=df_test[df_test['Country_Region'] == country], right=df_sub_all, left_on='ForecastId', right_on='ForecastId')

    df_country = df_country.groupby(['Date','Target','Quantile']).sum().reset_index()

    df_country.Date = pd.to_datetime(df_country.Date)

    mypalette=Spectral11[0:3]

    p = figure(title=country + " Confirmed Cases Forecast", x_axis_label='Date', x_axis_type='datetime', y_axis_label='Confirmed Cases')

    i = 0

    for alpha in QUANTILE:

        df_quantile = df_country[(df_country['Target'] == 'ConfirmedCases') & (df_country['Quantile'] == alpha)]   

        p.line(df_quantile['Date'], df_quantile['TargetValue'], legend_label=f"Confirmed Cases - Quantile {alpha}", line_width=2, line_color=mypalette[i])

        i += 1

    p.legend.location = "top_left"

    p.yaxis.formatter=NumeralTickFormatter(format="‘0.0a")    

    show(p)



    mypalette=Spectral11[0:3]

    p = figure(title=country + " Fatalities Forecast", x_axis_label='Date', x_axis_type='datetime', y_axis_label='Fatalities')

    i = 0

    for alpha in QUANTILE:

        df_quantile = df_country[(df_country['Target'] == 'Fatalities') & (df_country['Quantile'] == alpha)]   

        p.line(df_quantile['Date'], df_quantile['TargetValue'], legend_label=f"Fatalities - Quantile {alpha}", line_width=2, line_color=mypalette[i])

        i += 1

    p.legend.location = "top_left"

    p.yaxis.formatter=NumeralTickFormatter(format="‘0.0a")    

    show(p)
plotCountry('US')
plotCountry('Vietnam')