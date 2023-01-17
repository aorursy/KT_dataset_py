# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import io

import requests



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from plotly import tools

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')



from datetime import tzinfo, timedelta, datetime, date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sdds = pd.read_csv("../input/eirgridsystemdemand/SystemDemand_14.Apr.2020.00.00_13.May.2020.23.59.csv", header=0, na_values='-')

sdds2 = pd.read_csv("/kaggle/input/eirgridmetrics/SystemDemand_06.Jun.2020.00.00_05.Jul.2020.23.59.csv", na_values='-')

sdds = pd.concat([sdds, sdds2])

co2ds = pd.read_csv("/kaggle/input/eirgridmetrics/Co2Intensity_14.Apr.2020.00.00_13.May.2020.21.45.csv", na_values='-')

co2ds2 = pd.read_csv("/kaggle/input/eirgridmetrics/Co2Intensity_06.Jun.2020.00.00_05.Jul.2020.23.15.csv", na_values='-')

co2ds = pd.concat([co2ds, co2ds2])



co2ds['date'] = pd.to_datetime(co2ds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

co2ds

sgds = pd.read_csv("/kaggle/input/eirgridmetrics/SystemGeneration_14.Apr.2020.00.00_13.May.2020.21.45.csv")

sgds2 = pd.read_csv("/kaggle/input/eirgridmetrics/SystemGeneration_06.Jun.2020.00.00_05.Jul.2020.23.00.csv")

sgds = pd.concat([sgds, sgds2])



wgds = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_14.Apr.2020.00.00_13.May.2020.23.59.csv")

wgds2 = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_06.Jun.2020.00.00_05.Jul.2020.23.59.csv")

wgds = pd.concat([wgds, wgds2])

wgds.dtypes
#wg0621ds = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_21.Jun.2020.00.00_21.Jun.2020.23.59.csv")

icds = pd.read_csv("/kaggle/input/eirgridmetrics/Interconnection_14.Apr.2020.00.00_13.May.2020.22.00.csv")

icds2 = pd.read_csv("/kaggle/input/eirgridmetrics/Interconnection_06.Jun.2020.00.00_05.Jul.2020.23.00.csv")

icds = pd.concat([icds, icds2])

sdds
#co2ds['date'] = pd.to_datetime(co2ds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

sgds['date'] = pd.to_datetime(sgds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

wgds['date'] = pd.to_datetime(wgds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

icds['date'] = pd.to_datetime(icds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

#wg0621ds['date'] = pd.to_datetime(wg0621ds["DATE & TIME"], infer_datetime_format=True, errors='ignore')

sdds['date'] = pd.to_datetime(sdds["DATE & TIME"], infer_datetime_format=True, errors='ignore')
sdds = sdds.drop('DATE & TIME', axis=1)

co2ds = co2ds.drop('DATE & TIME', axis=1)

sgds = sgds.drop('DATE & TIME', axis=1)

wgds = wgds.drop('DATE & TIME', axis=1)

icds = icds.drop('DATE & TIME', axis=1)

#wg0621ds = wg0621ds.drop('DATE & TIME', axis=1)
icds.columns
sgds.columns
left = sdds.set_index(['date', ' REGION'])

right = co2ds.set_index(['date', ' REGION'])

#right.index = right.index.tz_convert(None)



newdf = left.join(right)



right = sgds.set_index(['date', ' REGION'])

newdf = newdf.join(right)



right = wgds.set_index(['date', ' REGION'])

newdf = newdf.join(right)

newdf = newdf.reset_index()

right = icds.set_index(['date'])

left = newdf.set_index(['date'])

newdf = left.join(right)



newdf = newdf.reset_index()
newdf.columns
newdf[' ACTUAL DEMAND(MW)']
newdf.dtypes
newdf.loc[newdf['  ACTUAL WIND(MW)']=='-']
newdf.loc[newdf[' ACTUAL GENERATION(MW)'].isnull()]
newdf.dropna(axis=0, subset=[' ACTUAL GENERATION(MW)'], inplace=True)

newdf['  ACTUAL WIND(MW)'] = newdf['  ACTUAL WIND(MW)'].astype('int32')

newdf[' ACTUAL DEMAND(MW)'] = newdf[' ACTUAL DEMAND(MW)'].astype('int32')

#newdf[' NET TOTAL(MW)'] = newdf[' NET TOTAL(MW)'].astype('int32')

#newdf[' EWIC(MW)'] = newdf[' EWIC(MW)'].astype('int32')

#newdf[' MOYLE(MW)'] = newdf[' MOYLE(MW)'].astype('int32')





    
newdf.describe()
# Check intuition that Actual Wind Electricity Generation correlates to overall electricity generation CO2 intensity

import plotly.express as px



df = px.data.tips()

#fig = px.scatter(newdf, x=' FORECAST DEMAND(MW)', y=' CO2 INTENSITY (gCO2/kWh)', trendline="ols")



fig = px.scatter(newdf, x='  ACTUAL WIND(MW)', y=' CO2 INTENSITY (gCO2/kWh)', trendline="ols")

fig.show()
df = px.data.tips()

#fig = px.scatter(newdf, x=' FORECAST DEMAND(MW)', y=' CO2 INTENSITY (gCO2/kWh)', trendline="ols")



fig = px.scatter(newdf, x='date', y=' CO2 INTENSITY (gCO2/kWh)')

fig.show()
weatherds = pd.read_csv("../input/ireland-weather/IrelandWeatherData (1).csv")
weatherds.columns
weatherds['date'].describe()


# Reduce timeframe

weatherds = weatherds[(weatherds.date >= "2020-04-14 00:00:00") & (weatherds.date < "2020-07-05 14:45:00")]
# Perform EDA, visualize, etc...

fig = make_subplots(

    rows=1, cols=1,

    specs=[[{"secondary_y": True}]],

    subplot_titles=("Belmullet"))





fig.add_trace(go.Scatter(x=weatherds['date'], y=weatherds['wdsp'],

                    mode='lines',

                    name='Wind Speed',showlegend=True), 1, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=weatherds['date'], y=weatherds['temp'],

                    mode='lines',

                    name='Temperature',showlegend=True), 1, 1, secondary_y=True)





fig.update_layout(

    title_text="Wind speed vs. Temperatures",

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=True,

    ),

    autosize=False,

    margin=dict(

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white',

    width=4000,

    height=500

)

fig.show()
import plotly.express as px



df = px.data.tips()

#fig = px.scatter(newdf, x='wdsp', y='wdsp_BEL', trendline="ols")

fig = px.scatter(weatherds, x='wdsp', y='temp', trendline="ols")

fig.show()
right = newdf.set_index(['date'])

left = weatherds.set_index(['date'])

newdf = left.join(right)



newdf = newdf.reset_index()
newdf.columns
df = px.data.tips()

fig = px.scatter(newdf, x='wdsp', y='  ACTUAL WIND(MW)', trendline="ols")

fig.show()
newdf.wdsp.describe()
df = px.data.tips()

fig = px.scatter(newdf, x='wdsp_COR', y='  ACTUAL WIND(MW)', trendline="ols")

fig.show()
df = px.data.tips()

fig = px.scatter(newdf, x='wdsp_BEL', y='  ACTUAL WIND(MW)', trendline="ols")

fig.show()
# Perform EDA, visualize, etc...

fig = make_subplots(

    rows=1, cols=1,

    specs=[[{"secondary_y": True}]],

    subplot_titles=("newdf"))





fig.add_trace(go.Scatter(x=newdf['date'], y=(newdf['wdsp_COR']^2),

                    mode='lines',

                    name='Wind Speed ^ 2',showlegend=True), 1, 1, secondary_y=False)



#fig.add_trace(go.Scatter(x=newdf['date'], y=newdf[' ACTUAL DEMAND(MW)'],

#                    mode='lines',

#                    name='Actual Demand (MW)',showlegend=True), 1, 1, secondary_y=True)



fig.add_trace(go.Scatter(x=newdf['date'], y=(newdf['  ACTUAL WIND(MW)'] / newdf[' ACTUAL DEMAND(MW)']),

                    mode='lines',

                    name='Wind Generation by Actual Demand',showlegend=True), 1, 1, secondary_y=True)





fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=True,

    ),

    autosize=False,

    margin=dict(

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white',

    width=2000,

    height=500

)

fig.show()
newdf["hour"] = 0

newdf["isweekday"] = False

newdf["month"] = 0

def settimeatt(row):

    row["hour"] = row.date.hour

    row["isweekday"] = (row.date.isoweekday() < 6) 

    row["month"] = row.date.month

    return row

newdf = newdf.apply(settimeatt, axis='columns')

newdf["Wind_gen_ratio"] = (newdf['  ACTUAL WIND(MW)'] / newdf[' ACTUAL DEMAND(MW)'])

newdf['wdsp_BEL2'] = (newdf['wdsp_BEL'] ^ 2)

newdf['wdsp_COR2'] = (newdf['wdsp_COR'] ^ 2)

newdf['wdsp_DUB2'] = (newdf['wdsp_DUB'] ^ 2)

newdf['wdsp2'] = (newdf['wdsp']^2)

import plotly.express as px



#df = px.data.tips()

fig = px.scatter(newdf, x='hour', y=' ACTUAL DEMAND(MW)', facet_col='isweekday')

fig.show()
fig = px.scatter(newdf, x='hour', y=' ACTUAL DEMAND(MW)', facet_col='month', color='isweekday')

fig.show()
newdf = newdf.rename(columns={' ACTUAL DEMAND(MW)' : 'ActualDemandMW'})

newdf = newdf.rename(columns={'  ACTUAL WIND(MW)' : 'ActualWindMW'})

newdf.columns
redds = newdf.loc[:, ['date', 'wdsp_BEL', 'wddir_BEL', 'wdsp', 'wddir','wdsp_COR', 'wddir_COR', 'wdsp_DUB', 'wddir_DUB', 'hour', 'isweekday', 'month', 'ActualWindMW']]
import seaborn as sns



num_df = redds.select_dtypes(['int', 'float'])

# Compute the correlation matrix

corr = num_df.corr()

corr = np.round(corr, 3)

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.6, center=0, annot= True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

from sklearn.model_selection import train_test_split



# Remove rows with missing target, separate target from predictors

redds.dropna(axis=0, subset=['ActualWindMW'], inplace=True)

y = redds.ActualWindMW

redds.drop(['ActualWindMW'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(redds, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64', 'bool']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_train.dtypes
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds))

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d: %s (%f)" % (f + 1, indices[f], X_train.columns[f], importances[indices[f]]))



# Plot the impurity-based feature importances of the forest

plt.figure()

plt.title("Feature importances range(X_train.shape[1])")

plt.bar(X_train.columns, importances[indices],

        color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), indices)

plt.xlim([-1, X_train.shape[1]])

plt.show()
fig = px.scatter(x=preds, y=(y_valid.array), trendline="ols")

fig.show()
y_valid.describe()
wfds = pd.read_csv("/kaggle/input/ireland-weather-forecast/WeatherForecast.csv")

wfds['date'] = pd.to_datetime(wfds["date"], infer_datetime_format=True, errors='ignore')

wfds.wdsp_BEL.describe()
wfds['date'].describe()
# Weather Hourly data wind speed in knt, for some reason, 1 knt = 0.514 m/s

knot = 0.514

wfds.wdsp_BEL = wfds.wdsp_BEL / knot

wfds.wdsp = wfds.wdsp / knot

wfds.wdsp_COR = wfds.wdsp_COR / knot

wfds.wdsp_DUB = wfds.wdsp_DUB / knot

wfds["hour"] = 0

wfds["isweekday"] = False

wfds["month"] = 0

wfds = wfds.apply(settimeatt, axis='columns')

# TODO Compatible types

redtds = wfds.loc[:, ['wdsp_BEL', 'wddir_BEL', 'wdsp', 'wddir','wdsp_COR', 'wddir_COR', 'wdsp_DUB', 'wddir_DUB', 'hour', 'isweekday', 'month']]

redtds.dtypes
preds = clf.predict(redtds)
preds
redtds
fig = go.Figure()

fig.add_trace(go.Box(y=newdf['wdsp_BEL'], name='BEL'))

fig.add_trace(go.Box(y=wfds['wdsp_BEL'], name='BEL forecast'))

fig.add_trace(go.Box(y=newdf['wdsp_DUB'], name = 'DUB'))

fig.add_trace(go.Box(y=wfds['wdsp_DUB'], name='DUB forecast'))

fig.add_trace(go.Box(y=newdf['wdsp'], name = 'SHA'))

fig.add_trace(go.Box(y=wfds['wdsp'], name='SHA forecast'))



fig.add_trace(go.Box(y=newdf['wdsp_COR'], name = 'CORK'))

fig.add_trace(go.Box(y=wfds['wdsp_COR'], name='CORK forecast'))

fig.show()
wfds["preds"] = preds
wfds
wgds3 = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_30.Jun.2020.00.00_29.Jul.2020.23.59.csv", na_values='-')

wgds3['date'] = pd.to_datetime(wgds3["DATE & TIME"], infer_datetime_format=True, errors='ignore')

wgds3 = wgds3.drop("DATE & TIME", axis=1)
wgds2 = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_27.Jul.2020.00.00_27.Jul.2020.23.59OLD.csv", na_values='-')

wgds28 = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_28.Jul.2020.00.00_28.Jul.2020.23.59OLD.csv", na_values='-')

wgds2 = pd.concat([wgds2, wgds28])

wgds29 = pd.read_csv("/kaggle/input/eirgridmetrics/WindGeneration_29.Jul.2020.00.00_29.Jul.2020.23.59.csv", na_values='-')

wgds2 = pd.concat([wgds2, wgds29])
wgds2['date'] = pd.to_datetime(wgds2["DATE & TIME"], infer_datetime_format=True, errors='ignore')

wgds2 = wgds2.drop("DATE & TIME", axis=1)

# Reduce time frame to common weather forecast and Eirgrid wind generation forecast

wgds2 = wgds2[(wgds2.date >= "2020-07-27 13:00:00") & (wgds2.date < "2020-07-30 00:00:00")]

wgds3 = wgds3[(wgds3.date >= "2020-07-27 13:00:00") & (wgds3.date < "2020-07-30 00:00:00")]

wfds = wfds[(wfds.date >= "2020-07-27 13:00:00") & (wfds.date < "2020-07-30 00:00:00")]

wgds2
wgds2 = wgds2.rename(columns={' FORECAST WIND(MW)' : 'ForecastWindMW'})

wgds2 = wgds2.rename(columns={'  ACTUAL WIND(MW)' : 'ActualWindMW'})

wgds3 = wgds3.rename(columns={' FORECAST WIND(MW)' : 'ForecastWindMW'})

wgds3 = wgds3.rename(columns={'  ACTUAL WIND(MW)' : 'ActualWindMW'})
wgds3.columns
right = wgds2.set_index(['date'])

left = wfds.set_index(['date']).copy()

fulldf = left.join(right, lsuffix="", rsuffix="_wf")



fulldf = fulldf.reset_index()
right = wgds3.set_index(['date'])

left = fulldf.set_index(['date']).copy()

fulldf = left.join(right, lsuffix="", rsuffix="_3")



fulldf = fulldf.reset_index()
fulldf.columns
# Perform EDA, visualize, etc...

fig = make_subplots(

    rows=1, cols=1,

    specs=[[{"secondary_y": True}]],

    subplot_titles=(""))





fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['preds']),

                    mode='lines',

                    name='Random Forest prediction',showlegend=True), 1, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['ForecastWindMW']),

                    mode='lines',

                    name='Eirgrid ForecastWindMW',showlegend=True), 1, 1, secondary_y=False)



#fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['ForecastWindMW_3']),

#                    mode='lines',

#                    name='Eirgrid ForecastWindMW v3',showlegend=True), 1, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['ActualWindMW_3'] ),

                    mode='lines', line=dict(width=3),

                    name='Eirgrid Wind Generation Actual',showlegend=True), 1, 1, secondary_y=False)







fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=True,

    ),

    autosize=False,

    margin=dict(

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white',

    width=1000,

    height=500

)

fig.show()
# Perform EDA, visualize, etc...

fig = make_subplots(

    rows=2, cols=1,

    specs=[[{"secondary_y": True}],

           [{"secondary_y": True}]],

    subplot_titles=(""))





fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['preds']),

                    mode='lines',

                    name='Random Forest prediction',showlegend=True), 1, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['ForecastWindMW']),

                    mode='lines',

                    name='Eirgrid ForecastWindMW',showlegend=True), 1, 1, secondary_y=False)





fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['ActualWindMW_3'] ),

                    mode='lines',

                    name='Eirgrid Wind Generation',showlegend=True), 1, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['wdsp_BEL'] ),

                    mode='lines',

                    name='Belmullet Wind Speed',showlegend=True), 2, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['wdsp'] ),

                    mode='lines',

                    name='Shannon Wind Speed',showlegend=True), 2, 1, secondary_y=False)



fig.add_trace(go.Scatter(x=fulldf['date'], y=(fulldf['wdsp_COR'] ),

                    mode='lines',

                    name='Cork Airport Wind Speed',showlegend=True), 2, 1, secondary_y=False)







fig.update_layout(

    xaxis=dict(

        showline=True,

        showgrid=False,

        showticklabels=True,

        linecolor='rgb(204, 204, 204)',

        linewidth=2,

        ticks='outside',

        tickfont=dict(

            family='Arial',

            size=12,

            color='rgb(82, 82, 82)',

        ),

    ),

    yaxis=dict(

        showgrid=False,

        zeroline=False,

        showline=False,

        showticklabels=True,

    ),

    autosize=False,

    margin=dict(

        l=100,

        r=20,

        t=110,

    ),

    plot_bgcolor='white',

    width=1000,

    height=500

)

fig.show()
fulldf.loc[:, ['date', 'preds', 'ForecastWindMW', 'ActualWindMW_3']].tail()
fulldf.shape
data_without_missing_values = fulldf.dropna(subset=["ActualWindMW_3"], axis=0)

#data_without_missing_values = data_without_missing_values.drop('ActualWindMW_3', axis =1)

data_without_missing_values.shape
print('MAE:', mean_absolute_error(data_without_missing_values['ActualWindMW_3'], data_without_missing_values['preds']))

dayaheadpred = fulldf.iloc[0:24]  
print('MAE:', mean_absolute_error(dayaheadpred['ActualWindMW_3'], dayaheadpred['preds']))
