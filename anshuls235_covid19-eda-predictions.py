#Libraries to import

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

import pycountry

import plotly_express as px

sns.set_style('darkgrid')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance, plot_tree
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv') 

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
display(df_train.head())

display(df_train.describe())

display(df_train.info())
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d')

df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d')
print('Minimum date from training set: {}'.format(df_train['Date'].min()))

print('Maximum date from training set: {}'.format(df_train['Date'].max()))
print('Minimum date from test set: {}'.format(df_test['Date'].min()))

print('Maximum date from test set: {}'.format(df_test['Date'].max()))
df_map = df_train.copy()

df_map['Date'] = df_map['Date'].astype(str)

df_map = df_map.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()
def get_iso3_util(country_name):

    try:

        country = pycountry.countries.get(name=country_name)

        return country.alpha_3

    except:

        if 'Congo' in country_name:

            country_name = 'Congo'

        elif country_name == 'Diamond Princess' or country_name == 'Laos':

            return country_name

        elif country_name == 'Korea, South':

            country_name = 'Korea, Republic of'

        elif country_name == 'Taiwan*':

            country_name = 'Taiwan'

        country = pycountry.countries.search_fuzzy(country_name)

        return country[0].alpha_3



d = {}

def get_iso3(country):

    if country in d:

        return d[country]

    else:

        d[country] = get_iso3_util(country)

    

df_map['iso_alpha'] = df_map.apply(lambda x: get_iso3(x['Country_Region']), axis=1)
df_map['ln(ConfirmedCases)'] = np.log(df_map.ConfirmedCases + 1)

df_map['ln(Fatalities)'] = np.log(df_map.Fatalities + 1)
px.choropleth(df_map, 

              locations="iso_alpha", 

              color="ln(ConfirmedCases)", 

              hover_name="Country_Region", 

              hover_data=["ConfirmedCases"] ,

              animation_frame="Date",

              color_continuous_scale=px.colors.sequential.dense, 

              title='Daily Confirmed Cases growth(Logarithmic Scale)')
px.choropleth(df_map, 

              locations="iso_alpha", 

              color="ln(Fatalities)", 

              hover_name="Country_Region",

              hover_data=["Fatalities"],

              animation_frame="Date",

              color_continuous_scale=px.colors.sequential.OrRd,

              title = 'Daily Deaths growth(Logarithmic Scale)')
#Get the top 10 countries

last_date = df_train.Date.max()

df_countries = df_train[df_train['Date']==last_date]

df_countries = df_countries.groupby('Country_Region', as_index=False)['ConfirmedCases','Fatalities'].sum()

df_countries = df_countries.nlargest(10,'ConfirmedCases')

#Get the trend for top 10 countries

df_trend = df_train.groupby(['Date','Country_Region'], as_index=False)['ConfirmedCases','Fatalities'].sum()

df_trend = df_trend.merge(df_countries, on='Country_Region')

df_trend.drop(['ConfirmedCases_y','Fatalities_y'],axis=1, inplace=True)

df_trend.rename(columns={'Country_Region':'Country', 'ConfirmedCases_x':'Cases', 'Fatalities_x':'Deaths'}, inplace=True)

#Add columns for studying logarithmic trends

df_trend['ln(Cases)'] = np.log(df_trend['Cases']+1)# Added 1 to remove error due to log(0).

df_trend['ln(Deaths)'] = np.log(df_trend['Deaths']+1)
px.line(df_trend, x='Date', y='Cases', color='Country', title='COVID19 Cases growth for top 10 worst affected countries')
px.line(df_trend, x='Date', y='Deaths', color='Country', title='COVID19 Deaths growth for top 10 worst affected countries')
px.line(df_trend, x='Date', y='ln(Cases)', color='Country', title='COVID19 Cases growth for top 10 worst affected countries(Logarithmic Scale)')
px.line(df_trend, x='Date', y='ln(Deaths)', color='Country', title='COVID19 Deaths growth for top 10 worst affected countries(Logarithmic Scale)')
# Dictionary to get the state codes from state names for US

us_state_abbrev = {

    'Alabama': 'AL',

    'Alaska': 'AK',

    'American Samoa': 'AS',

    'Arizona': 'AZ',

    'Arkansas': 'AR',

    'California': 'CA',

    'Colorado': 'CO',

    'Connecticut': 'CT',

    'Delaware': 'DE',

    'District of Columbia': 'DC',

    'Florida': 'FL',

    'Georgia': 'GA',

    'Guam': 'GU',

    'Hawaii': 'HI',

    'Idaho': 'ID',

    'Illinois': 'IL',

    'Indiana': 'IN',

    'Iowa': 'IA',

    'Kansas': 'KS',

    'Kentucky': 'KY',

    'Louisiana': 'LA',

    'Maine': 'ME',

    'Maryland': 'MD',

    'Massachusetts': 'MA',

    'Michigan': 'MI',

    'Minnesota': 'MN',

    'Mississippi': 'MS',

    'Missouri': 'MO',

    'Montana': 'MT',

    'Nebraska': 'NE',

    'Nevada': 'NV',

    'New Hampshire': 'NH',

    'New Jersey': 'NJ',

    'New Mexico': 'NM',

    'New York': 'NY',

    'North Carolina': 'NC',

    'North Dakota': 'ND',

    'Northern Mariana Islands':'MP',

    'Ohio': 'OH',

    'Oklahoma': 'OK',

    'Oregon': 'OR',

    'Pennsylvania': 'PA',

    'Puerto Rico': 'PR',

    'Rhode Island': 'RI',

    'South Carolina': 'SC',

    'South Dakota': 'SD',

    'Tennessee': 'TN',

    'Texas': 'TX',

    'Utah': 'UT',

    'Vermont': 'VT',

    'Virgin Islands': 'VI',

    'Virginia': 'VA',

    'Washington': 'WA',

    'West Virginia': 'WV',

    'Wisconsin': 'WI',

    'Wyoming': 'WY'

}
df_us = df_train[df_train['Country_Region']=='US']

df_us['Date'] = df_us['Date'].astype(str)

df_us['state_code'] = df_us.apply(lambda x: us_state_abbrev.get(x.Province_State,float('nan')), axis=1)

df_us['ln(ConfirmedCases)'] = np.log(df_us.ConfirmedCases + 1)

df_us['ln(Fatalities)'] = np.log(df_us.Fatalities + 1)
px.choropleth(df_us,

              locationmode="USA-states",

              scope="usa",

              locations="state_code",

              color="ln(ConfirmedCases)",

              hover_name="Province_State",

              hover_data=["ConfirmedCases"],

              animation_frame="Date",

              color_continuous_scale=px.colors.sequential.Darkmint,

              title = 'Daily Cases growth for USA(Logarithmic Scale)')
px.choropleth(df_us,

              locationmode="USA-states",

              scope="usa",

              locations="state_code",

              color="ln(Fatalities)",

              hover_name="Province_State",

              hover_data=["Fatalities"],

              animation_frame="Date",

              color_continuous_scale=px.colors.sequential.OrRd,

              title = 'Daily deaths growth for USA(Logarithmic Scale)')
df_train.Province_State.fillna('NaN', inplace=True)
df_plot = df_train.groupby(['Date','Country_Region','Province_State'], as_index=False)['ConfirmedCases','Fatalities'].sum()
df = df_plot.query("Country_Region=='India'")

px.line(df, x='Date', y='ConfirmedCases', title='Daily Cases growth for India')
px.line(df, x='Date', y='Fatalities', title='Daily Deaths growth for India')
ch_geojson = "../input/china-regions-map/china-provinces.json"

df_plot['day'] = df_plot.Date.dt.dayofyear

df_plot['Province_ch'] = "新疆维吾尔自治区"
df = df_plot.query("Country_Region=='China'")

fig = px.choropleth_mapbox(df,

              geojson=ch_geojson,

              #scope="asia",

              color="ConfirmedCases",

              locations="Province_ch",

              featureidkey="objects.CHN_adm1.geometries.properties.NL_NAME_1",

              #featureidkey="features.properties.name",

              animation_frame="day")

fig.update_geos(fitbounds="locations", visible=False)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
df = df_plot.query("Country_Region=='China'")

px.line(df, x='Date', y='ConfirmedCases', color='Province_State', title='Daily Cases growth for China')
px.line(df, x='Date', y='Fatalities', color='Province_State', title='Daily Deaths growth for China')
def categoricalToInteger(df):

    #convert NaN Province State values to a string

    df.Province_State.fillna('NaN', inplace=True)

    #Define Ordinal Encoder Model

    oe = OrdinalEncoder()

    df[['Province_State','Country_Region']] = oe.fit_transform(df.iloc[:,1:3])

    return df
def create_features(df):

    df['day'] = df['Date'].dt.day

    df['month'] = df['Date'].dt.month

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['quarter'] = df['Date'].dt.quarter

    df['weekofyear'] = df['Date'].dt.weekofyear

    return df
def cum_sum(df, date, country, state):

    sub_df = df[(df['Country_Region']==country) & (df['Province_State']==state) & (df['Date']<=date)]

    display(sub_df)

    return sub_df['ConfirmedCases'].sum(), sub_df['Fatalities'].sum()
def train_dev_split(df):

    date = df['Date'].max() - dt.timedelta(days=7)

    return df[df['Date'] <= date], df[df['Date'] > date]
df_train = categoricalToInteger(df_train)

df_train = create_features(df_train)
df_train, df_dev = train_dev_split(df_train)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region','ConfirmedCases','Fatalities']

df_train = df_train[columns]

df_dev = df_dev[columns]
train = df_train.values

dev = df_dev.values

X_train, y_train = train[:,:-2], train[:,-2:]

X_dev, y_dev = dev[:,:-2], dev[:,-2:]
'''train = df_train.values

X_train, y_train = train[:,:-2], train[:,-2:]'''
def modelfit(alg, X_train, y_train,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(X_train, label=y_train)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

            metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_stdv=False)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(X_train, y_train,eval_metric='rmse')

        

    #Predict training set:

    predictions = alg.predict(X_train)

    #predprob = alg.predict_proba(X_train)[:,1]

        

    #Print model report:

    print("\nModel Report")

    #print("Accuracy : %.4g" % metrics.accuracy_score(y_train, predictions))

    print("RMSE Score (Train): %f" % metrics.mean_squared_error(y_train, predictions))

                    

    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)

    feat_imp.plot(kind='bar', title='Feature Importances')

    plt.ylabel('Feature Importance Score')
'''model1 = XGBRegressor(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'reg:squarederror',

 scale_pos_weight=1)

modelfit(model1, X_train, y_train[:,0])'''
'''model2 = XGBRegressor(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'reg:squarederror',

 scale_pos_weight=1)

modelfit(model2, X_train, y_train[:,1])'''
model1 = XGBRegressor(n_estimators=1000)

model2 = XGBRegressor(n_estimators=1000)
model1.fit(X_train, y_train[:,0],

           eval_set=[(X_train, y_train[:,0]), (X_dev, y_dev[:,0])],

           verbose=False)
model2.fit(X_train, y_train[:,1],

           eval_set=[(X_train, y_train[:,1]), (X_dev, y_dev[:,1])],

           verbose=False)
plot_importance(model1);
plot_importance(model2);
df_train = categoricalToInteger(df_test)

df_train = create_features(df_test)
columns = ['day','month','dayofweek','dayofyear','quarter','weekofyear','Province_State', 'Country_Region']

df_test = df_test[columns]
y_pred1 = model1.predict(df_test.values)

y_pred2 = model2.predict(df_test.values)
df_submit = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
df_submit.ConfirmedCases = y_pred1

df_submit.Fatalities = y_pred2
'''df_submit.ConfirmedCases = df_submit.ConfirmedCases.apply(lambda x:max(0,round(x,0)))

df_submit.Fatalities = df_submit.Fatalities.apply(lambda x:max(0,round(x,0)))'''
df_submit.to_csv(r'submission.csv', index=False)