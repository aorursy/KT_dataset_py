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
import numpy as np 

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import matplotlib.gridspec as gridspec



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import cross_val_score

from sklearn import metrics

import statsmodels.api as sm



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error



from keras.layers import Flatten 

data = pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
data.head(5)
data.info()
data.columns = (["country","region","population","area","density","coastline_area_ratio","net_migration","infant_mortality","gdp_per_capita",

                  "literacy","phones","arable","crops","other","climate","birthrate","deathrate","agriculture","industry",

                  "service"])
data.country = data.country.astype('category')



data.region = data.region.astype('category')



data.density = data.density.astype(str)

data.density = data.density.str.replace(",",".").astype(float)



data.coastline_area_ratio = data.coastline_area_ratio.astype(str)

data.coastline_area_ratio = data.coastline_area_ratio.str.replace(",",".").astype(float)



data.net_migration = data.net_migration.astype(str)

data.net_migration = data.net_migration.str.replace(",",".").astype(float)



data.infant_mortality = data.infant_mortality.astype(str)

data.infant_mortality = data.infant_mortality.str.replace(",",".").astype(float)



data.literacy = data.literacy.astype(str)

data.literacy = data.literacy.str.replace(",",".").astype(float)



data.phones = data.phones.astype(str)

data.phones = data.phones.str.replace(",",".").astype(float)



data.arable = data.arable.astype(str)

data.arable = data.arable.str.replace(",",".").astype(float)



data.crops = data.crops.astype(str)

data.crops = data.crops.str.replace(",",".").astype(float)



data.other = data.other.astype(str)

data.other = data.other.str.replace(",",".").astype(float)



data.climate = data.climate.astype(str)

data.climate = data.climate.str.replace(",",".").astype(float)



data.birthrate = data.birthrate.astype(str)

data.birthrate = data.birthrate.str.replace(",",".").astype(float)



data.deathrate = data.deathrate.astype(str)

data.deathrate = data.deathrate.str.replace(",",".").astype(float)



data.agriculture = data.agriculture.astype(str)

data.agriculture = data.agriculture.str.replace(",",".").astype(float)



data.industry = data.industry.astype(str)

data.industry = data.industry.str.replace(",",".").astype(float)



data.service = data.service.astype(str)

data.service = data.service.str.replace(",",".").astype(float)
data.info()
data.shape
data.head(5)
fig = plt.figure(figsize=(16,30))

features= ["population","area", "density", "coastline_area_ratio","net_migration","infant_mortality", "literacy", "phones", "arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"]



for i in range(len(features)):

    fig.add_subplot(9, 5, i+1)

    sns.boxplot(y=data[features[i]])

plt.tight_layout()

plt.show()
fig = plt.figure(figsize=(16,30))

features= ["population","area", "density", "coastline_area_ratio","net_migration","infant_mortality", "literacy", "phones", "arable","crops","other","climate","birthrate","deathrate","agriculture","industry","service"]



for i in range(len(features)):

    fig.add_subplot(9, 5, i+1)

    sns.distplot(data[features[i]])

plt.tight_layout()

plt.show()
# boxplot for distribution analysis of region (categorical data) with GDP per Capita



sns.boxplot(y=data['gdp_per_capita'],x= data['region'])

plt.tight_layout()

plt.xticks(rotation=90)

plt.show()
fig = plt.figure(constrained_layout=True, figsize=(16,6))

grid = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)

ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('Histogram')

sns.distplot(data.loc[:,'gdp_per_capita'], norm_hist=True, ax = ax1)

ax3 = fig.add_subplot(grid[:, 2])

ax3.set_title('Box Plot')

sns.boxplot(data.loc[:,'gdp_per_capita'], orient='v', ax = ax3)

plt.show()
#skewness and kurtosis

print("Skewness: %f" % data['gdp_per_capita'].skew())

print("Kurtosis: %f" % data['gdp_per_capita'].kurt())
#missing data

total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum()/data.isnull().count()).sort_values(ascending=False))*100

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=percent.index, y=percent)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
sns.heatmap(data.isnull()).set(title = 'Missing Data', xlabel = 'Columns', ylabel = 'Data Points')
data.head(20)
print(data.isnull().sum())
data['gdp_per_capita'].isnull()
data['country'].iloc[223]
data[data['literacy'].isnull()].index.tolist()
print(data['country'].iloc[66])

print(data['region'].iloc[66])
data[data['phones'].isnull()].index.tolist()
print(data['country'].iloc[52])

print(data['region'].iloc[52])
data['climate']
data['agriculture']
data[data['agriculture'].isnull()].index.tolist()
print(data.iloc[3])

print(data.iloc[4])

print(data.iloc[78])

print(data.iloc[80])

print(data.iloc[83])

print(data.iloc[134])

print(data.iloc[140])

print(data.iloc[144])

print(data.iloc[153])

print(data.iloc[171])

print(data.iloc[174])

print(data.iloc[177])

print(data.iloc[208])

print(data.iloc[221])

print(data.iloc[223])
data[data['industry'].isnull()].index.tolist()
print(data.iloc[138])
data['net_migration'].fillna(0, inplace=True)

data['infant_mortality'].fillna(0, inplace=True)

data['gdp_per_capita'].fillna(2500, inplace=True)

data['literacy'].fillna(data.groupby('region')['literacy'].transform('median'), inplace= True)

data['phones'].fillna(data.groupby('region')['phones'].transform('median'), inplace= True)

data['arable'].fillna(0, inplace=True)

data['crops'].fillna(0, inplace=True)

data['other'].fillna(0, inplace=True)

data['climate'].fillna(0, inplace=True)

data['birthrate'].fillna(data.groupby('region')['birthrate'].transform('mean'), inplace= True)

data['deathrate'].fillna(data.groupby('region')['deathrate'].transform('median'), inplace= True)
# For monaco, i will set the value for industry and service to be 0.05 and 0.78 respectively 

data['industry'][138] = 0.05

data['service'][138] = 0.78

print(data['industry'][138])

print(data['service'][138])



# For western sahara, i will set the value for agriculture and industry to be 0.35 and 0.25 respectively.

data['industry'][223] = 0.25

data['agriculture'][223] = 0.35

print(data['industry'][223])

print(data['agriculture'][223])
data['agriculture'].fillna(0.15, inplace=True)

data['service'].fillna(0.8, inplace=True)

data['industry'].fillna(0.05, inplace= True)
print(data.isnull().sum())
fig, ax = plt.subplots(figsize=(16,16)) 

sns.heatmap(data.corr(), annot=True, ax=ax).set(

    title = 'Feature Correlation', xlabel = 'Columns', ylabel = 'Columns')

plt.show()

fig = plt.figure(figsize=(12, 4))

data.groupby('region')['gdp_per_capita'].mean().sort_values().plot(kind='bar')

plt.title('Regional Average GDP per Capita')

plt.xlabel("Region")

plt.ylabel('Average GDP per Capita')

plt.show()
sns.boxplot(x="region",y="gdp_per_capita",data=data,width=0.7,palette="Set3",fliersize=5)

plt.xticks(rotation=90)

plt.title("Regional Average GDP per Capita")
fig = plt.figure(figsize=(12, 12))

sns.jointplot(data= data, x= 'literacy', y= 'gdp_per_capita', kind= 'scatter')

plt.title('GDP Analysis: GDP per capita vs Literacy')

plt.show()
fig = plt.figure(figsize=(12, 12))

sns.jointplot(data= data, x= 'phones', y= 'gdp_per_capita', kind= 'scatter')

plt.title('GDP Analysis: GDP per capita vs Literacy')

plt.show()
fig = plt.figure(figsize=(12, 12))

sns.jointplot(data= data, x= 'phones', y= 'gdp_per_capita', kind= 'hex')

plt.title('GDP Analysis: GDP per capita vs Literacy')

plt.show()
fig = plt.figure(figsize=(12, 12))

sns.jointplot(data= data, x= 'infant_mortality', y= 'gdp_per_capita', kind= 'scatter')

plt.title('GDP Analysis: GDP per capita vs infant_mortality ')

plt.show()
fig = plt.figure(figsize=(12, 12))

sns.jointplot(data= data, x= 'birthrate', y= 'infant_mortality', kind= 'scatter')

plt.title('GDP Analysis: birthrate vs infant_mortality')

plt.show()
gdp=data.sort_values(["gdp_per_capita"],ascending=False)



# prepare data frame

df = gdp.iloc[:100,:]



# Creating trace1

trace1 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.birthrate,

                    mode = "lines",

                    name = "Birthrate",

                    marker = dict(color = 'rgba(235,66,30, 0.8)'),

                    text= df.country)

# Creating trace2

trace2 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.deathrate,

                    mode = "lines+markers",

                    name = "Deathrate",

                    marker = dict(color = 'rgba(10,10,180, 0.8)'),

                    text= df.country)

z = [trace1, trace2]

layout = dict(title = 'Birthrate and Deathrate of World Countries (Top 100)',

              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)

             )

fig = dict(data = z, layout = layout)

iplot(fig)
gdp=data.sort_values(["gdp_per_capita"],ascending=False)
# prepare data frame

df = gdp.iloc[127:227,:]



# Creating trace1

trace1 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.birthrate,

                    mode = "lines",

                    name = "Birthrate",

                    marker = dict(color = 'rgba(235,66,30, 0.8)'),

                    text= df.country)

# Creating trace2

trace2 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.deathrate,

                    mode = "lines+markers",

                    name = "Deathrate",

                    marker = dict(color = 'rgba(10,10,180, 0.8)'),

                    text= df.country)

z = [trace1, trace2]

layout = dict(title = 'Birthrate and Deathrate Percentage of World Countries (Last 100)',

              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)

             )

fig = dict(data = z, layout = layout)

iplot(fig)
# prepare data frame

df = gdp.iloc[:100,:]



# Creating trace1

trace1 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.agriculture,

                    mode = "lines+markers",

                    name = "AGRICULTURE",

                    marker = dict(color = 'rgba(235,66,30, 0.8)'),

                    text= df.country)

# Creating trace2

trace2 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.industry,

                    mode = "lines+markers",

                    name = "INDUSTRY",

                    marker = dict(color = 'rgba(10,10,180, 0.8)'),

                    text= df.country)

# Creating trace3

trace3 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.service,

                    mode = "lines+markers",

                    name = "SERVICE",

                    marker = dict(color = 'rgba(10,250,60, 0.8)'),

                    text= df.country)





z = [trace1, trace2,trace3]

layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (TOP 100)',

              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)

             )

fig = dict(data = z, layout = layout)

iplot(fig)
# prepare data frame

df = gdp.iloc[127:227,:]



# Creating trace1

trace1 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.agriculture,

                    mode = "lines+markers",

                    name = "AGRICULTURE",

                    marker = dict(color = 'rgba(235,66,30, 0.8)'),

                    text= df.country)

# Creating trace2

trace2 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.industry,

                    mode = "lines+markers",

                    name = "INDUSTRY",

                    marker = dict(color = 'rgba(10,10,180, 0.8)'),

                    text= df.country)

# Creating trace3

trace3 = go.Scatter(

                    x = df.gdp_per_capita,

                    y = df.service,

                    mode = "lines+markers",

                    name = "SERVICE",

                    marker = dict(color = 'rgba(10,250,60, 0.8)'),

                    text= df.country)





z = [trace1, trace2,trace3]

layout = dict(title = 'Service , Industry and Agriculture Percentage of World Countries (LAST 100)',

              xaxis= dict(title= 'GDP',ticklen= 5,zeroline= False)

             )

fig = dict(data = z, layout = layout)

iplot(fig)
lit = data.sort_values("literacy",ascending=False).head(7)
trace1 = go.Bar(

                x = lit.country,

                y = lit.agriculture,

                name = "agriculture",

                marker = dict(color = 'rgba(255, 20, 20, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lit.gdp_per_capita)

trace2 = go.Bar(

                x = lit.country,

                y = lit.service,

                name = "service",

                marker = dict(color = 'rgba(20, 20, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = lit.gdp_per_capita)

z = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = z, layout = layout)

iplot(fig)
x = lit.country



trace1 = {

  'x': x,

  'y': lit.industry,

  'name': 'industry',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': lit.service,

  'name': 'service',

  'type': 'bar'

};

z = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 7 country'},

  'barmode': 'relative',

  'title': 'industry and service percentage of top 7 country (literacy)'

};

fig = go.Figure(data = z, layout = layout)

iplot(fig)
#Population per country

z = dict(type='choropleth',

            locations = data.country,

            locationmode = 'country names', z = data.population,

            text = data.country, colorbar = {'title':'Population'},

            colorscale = 'Blackbody', reversescale = True)



layout = dict(title='Population per country',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [z],layout = layout)

iplot(choromap,validate=False)
#Infant motality per country

z = dict(type='choropleth',

        locations = data.country,

        locationmode = 'country names', z = data.infant_mortality,

        text = data.country, colorbar = {'title':'Infant Mortality'},

        colorscale = 'YlOrRd', reversescale = True)

layout = dict(title='Infant Mortality per Country',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [z],layout = layout)

iplot(choromap,validate=False)
data.head(5)
#Population per country

z = dict(type='choropleth',

locations = data.country,

locationmode = 'country names', z = data.gdp_per_capita,

text = data.country, colorbar = {'title':'GDP per Capita'},

colorscale = 'Hot', reversescale = True)

layout = dict(title='GDP per Capita of World Countries',

geo = dict(showframe=False,projection={'type':'natural earth'}))

choromap = go.Figure(data = [z],layout = layout)

iplot(choromap,validate=False)
fig = plt.figure(figsize=(18, 24))

plt.title('Regional Analysis')

ax1 = fig.add_subplot(4, 1, 1)

ax2 = fig.add_subplot(4, 1, 2)

ax3 = fig.add_subplot(4, 1, 3)

ax4 = fig.add_subplot(4, 1, 4)

sns.countplot(data= data, y= 'region', ax= ax1)

sns.barplot(data= data, y= 'region', x= 'gdp_per_capita', ax= ax2, ci= None)

sns.barplot(data= data, y= 'region', x= 'net_migration', ax= ax3, ci= None)

sns.barplot(data= data, y= 'region', x= 'population', ax= ax4, ci= None)

plt.show()
data_final = pd.concat([data,pd.get_dummies(data['region'], prefix='region')], axis=1)

#dropping the redundant region column

data_final.drop(['region'],axis=1,inplace=True)

print(data_final.info())
data_final.head(10)
y = data_final['gdp_per_capita']

X = data_final.drop(['gdp_per_capita','country'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#StandardScaler will transform data such that its distribution will have a mean value 0 and standard deviation of 1.

sc_X = StandardScaler()



X2_train = sc_X.fit_transform(X_train)

X2_test = sc_X.fit_transform(X_test)

y2_train = y_train

y2_test = y_test
data_final[data_final.columns[1:]].corr()['gdp_per_capita'][:]
y3 = y

X3 = data_final.drop(['gdp_per_capita','country','population', 'area', 'coastline_area_ratio', 'arable',

                      'crops', 'other', 'climate', 'deathrate', 'industry'], axis=1)



X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=101)
sc_X4 = StandardScaler()



X4_train = sc_X4.fit_transform(X3_train)

X4_test = sc_X4.fit_transform(X3_test)

y4_train = y3_train

y4_test = y3_test
lm1 = LinearRegression()

lm1.fit(X_train,y_train)



lm2 = LinearRegression()

lm2.fit(X2_train,y2_train)



lm3 = LinearRegression()

lm3.fit(X3_train,y3_train)



lm4 = LinearRegression()

lm4.fit(X4_train,y4_train)
lm1_pred = lm1.predict(X_test)

lm2_pred = lm2.predict(X2_test)

lm3_pred = lm3.predict(X3_test)

lm4_pred = lm4.predict(X4_test)
print('Linear Regression Performance:')



print('\nall features, No scaling:')

print('MAE:', mean_absolute_error(y_test, lm1_pred))

print('RMSE:', np.sqrt(mean_squared_error(y_test, lm1_pred)))

print('R2_Score: ', r2_score(y_test, lm1_pred))



print('\nall features, with scaling:')

print('MAE:', mean_absolute_error(y2_test, lm2_pred))

print('RMSE:', np.sqrt(mean_squared_error(y2_test, lm2_pred)))

print('R2_Score: ', r2_score(y2_test, lm2_pred))



print('\nselected features, No scaling:')

print('MAE:', mean_absolute_error(y3_test, lm3_pred))

print('RMSE:', np.sqrt(mean_squared_error(y3_test, lm3_pred)))

print('R2_Score: ', r2_score(y3_test, lm3_pred))



print('\nselected features, with scaling:')

print('MAE:', mean_absolute_error(y4_test, lm4_pred))

print('RMSE:', np.sqrt(mean_squared_error(y4_test, lm4_pred)))

print('R2_Score: ', r2_score(y4_test, lm4_pred))

fig = plt.figure(figsize=(12, 6))

plt.scatter(y4_test,lm4_pred,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predicted GDP per Capita') 

plt.title('Linear Regression Prediction Performance (features selected and scaled)') 

plt.grid()

plt.show()
model = sm.OLS(y4_train, X4_train).fit()

print_model = model.summary()

print(print_model)



print(X3.columns[1])
#what the coefficient values refers to

i=1

for col in X3.columns: 

    print("x"+str(i)+":"+str(col))

    i=i+1
print(lm4.coef_)

from matplotlib import pyplot



importance = lm4.coef_

# summarize feature importance

# for i,v in enumerate(importance):

#     print("Feature:" + str(X3.columns[i]) + ", Score: %.5f" % (i,v))

# plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)

pyplot.show()
features = {'Feature': ['density', 'net_migration','infant_mortality','literacy','phones', 'birthrate', 'agriculture','service','region_ASIA (EX. NEAR EAST)','region_BALTICS', 'region_C.W. OF IND. STATES', 'region_EASTERN EUROPE', 'region_LATIN AMER. & CARIB', 'region_NEAR EAST', 'region_NORTHERN AFRICA', 'region_NORTHERN AMERICA', 'region_OCEANIA', 'region_SUB-SAHARAN AFRICA', 'region_WESTERN EUROPE'],

        'Coefficient': [-593.27081629, 1789.42167595, -309.12088609, 256.01950704, 5313.19849616, -1318.85076516, -1655.24940852, -706.2682728, -25.78832369, -262.58839876, -957.2313499, -789.87119263, -819.42112542, -230.40994515, -297.50983884, 467.15853813, 132.13764433, 241.54449668, 2086.82900519]}



df_features = pd.DataFrame(features, columns = ['Feature', 'Coefficient'])

print (df)



df_sorted_desc= df_features.sort_values('Coefficient',ascending=False)

plt.figure(figsize=(10,6))

# bar plot with matplotlib

plt.bar('Feature', 'Coefficient',data=df_sorted_desc)

plt.xticks(rotation=90)

plt.xlabel("Feature", size=15)

plt.ylabel("Feature Coefficient", size=15)

plt.title("Coefficient of features (X4)", size=18)
df_features['AbsCoefficient']=""

df_features['AbsCoefficient'] = df_features['Coefficient'].abs()



df_abs_desc= df_features.sort_values('AbsCoefficient',ascending=False)

plt.figure(figsize=(10,6))

# bar plot with matplotlib

plt.bar('Feature', 'AbsCoefficient',data=df_abs_desc)

plt.xticks(rotation=90)

plt.xlabel("Feature", size=15)

plt.ylabel("Feature importance", size=15)

plt.title("Feature importance (X4)", size=18)
print("Ridge Regression performance")

rr = Ridge(alpha=0.01)

rr.fit(X4_train,y4_train) 

pred_test_ridge= rr.predict(X4_test)



print('MAE:', mean_absolute_error(y4_test,pred_test_ridge))

print('RMSE:', np.sqrt(mean_squared_error(y4_test,pred_test_ridge)))

print('R2_Score: ', r2_score(y4_test,pred_test_ridge))
print("Lasso Regression performance")

model_lasso = Lasso(alpha=0.01,tol=0.01)

model_lasso.fit(X4_train,y4_train) 

pred_test_lasso= model_lasso.predict(X4_test)

print('MAE:', mean_absolute_error(y4_test,pred_test_lasso))

print('RMSE:', np.sqrt(mean_squared_error(y4_test,pred_test_lasso)))

print('R2_Score: ', r2_score(y4_test,pred_test_lasso))
svm1 = SVR(kernel='rbf')

svm1.fit(X_train,y_train)



svm2 = SVR(kernel='rbf')

svm2.fit(X2_train,y2_train)



svm3 = SVR(kernel='rbf')

svm3.fit(X3_train,y3_train)



svm4 = SVR(kernel='rbf')

svm4.fit(X4_train,y4_train)
svm1_pred = svm1.predict(X_test)

svm2_pred = svm2.predict(X2_test)

svm3_pred = svm3.predict(X3_test)

svm4_pred = svm4.predict(X4_test)
print('SVM Performance:')



print('\nall features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y_test, svm1_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svm1_pred)))

print('R2_Score: ', metrics.r2_score(y_test, svm1_pred))



print('\nall features, with scaling:')

print('MAE:', metrics.mean_absolute_error(y2_test, svm2_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, svm2_pred)))

print('R2_Score: ', metrics.r2_score(y2_test, svm2_pred))



print('\nselected features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y3_test, svm3_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y3_test, svm3_pred)))

print('R2_Score: ', metrics.r2_score(y3_test, svm3_pred))



print('\nselected features, with scaling:')

print('MAE:', metrics.mean_absolute_error(y4_test, svm4_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y4_test, svm4_pred)))

print('R2_Score: ', metrics.r2_score(y4_test, svm4_pred))



fig = plt.figure(figsize=(12, 6))

plt.scatter(y3_test,svm3_pred,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Unoptimized SVM prediction Performance (with feature selection, and scaling)') 

plt.grid()

plt.show()
param_grid = {'C': [1, 10, 100], 'gamma': [0.01,0.001,0.0001], 'kernel': ['rbf']} 

grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=3)
grid.fit(X4_train,y4_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X4_test)
print("Optimized SVM Performance:")

print('MAE:', metrics.mean_absolute_error(y4_test, grid_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y4_test, grid_predictions)))

print('R2_Score: ', metrics.r2_score(y4_test, grid_predictions))



fig = plt.figure(figsize=(12, 6))

plt.scatter(y4_test,grid_predictions,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Optimized SVM prediction Performance (with feature selection, and scaling)') 

plt.grid()

plt.show()
rf1 = RandomForestRegressor(random_state=101, n_estimators=200)

rf3 = RandomForestRegressor(random_state=101, n_estimators=200)



rf1.fit(X_train, y_train)

rf3.fit(X3_train, y3_train)

rf1_pred = rf1.predict(X_test)

rf3_pred = rf3.predict(X3_test)
print('Random Forest Performance:')



print('\nall features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y_test, rf1_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf1_pred)))

print('R2_Score: ', metrics.r2_score(y_test, rf1_pred))



print('\nselected features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y3_test, rf3_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y3_test, rf3_pred)))

print('R2_Score: ', metrics.r2_score(y3_test, rf3_pred))



fig = plt.figure(figsize=(12, 6))

plt.scatter(y_test,rf1_pred,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Random Forest prediction Performance (No feature selection)') 

plt.grid()

plt.show()
rf_param_grid = {'max_features': ['sqrt', 'auto'],

              'min_samples_leaf': [1, 3, 5],

           # 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

                 'min_samples_split': [2, 5, 10],

              'n_estimators': [100,500,1000],

             'bootstrap': [False, True]}
rf_grid = GridSearchCV(estimator= RandomForestRegressor(), param_grid = rf_param_grid,  n_jobs=-1, verbose=0)
rf_grid.fit(X_train,y_train)
rf_grid.best_params_
rf_grid.best_estimator_
rf_grid_predictions = rf_grid.predict(X_test)
print('Random Forest with optimization performance')



print('MAE:', metrics.mean_absolute_error(y_test, rf_grid_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rf_grid_predictions)))

print('R2_Score: ', metrics.r2_score(y_test, rf_grid_predictions))

fig = plt.figure(figsize=(12, 6))

plt.scatter(y_test,rf_grid_predictions,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Optimized Random Forest prediction Performance (No feature selection)') 

plt.grid()

plt.show()
gbm1 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=3,

                                 subsample=1.0, max_features= None, random_state=101)

gbm3 = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=3,

                                 subsample=1.0, max_features= None, random_state=101)



gbm1.fit(X_train, y_train)

gbm3.fit(X3_train, y3_train)



gbm1_pred = gbm1.predict(X_test)

gbm3_pred = gbm3.predict(X3_test)
print('Gradient Boosting Performance:')



print('\nall features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y_test, gbm1_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, gbm1_pred)))

print('R2_Score: ', metrics.r2_score(y_test, gbm1_pred))



print('\nselected features, No scaling:')

print('MAE:', metrics.mean_absolute_error(y3_test, gbm3_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y3_test, gbm3_pred)))

print('R2_Score: ', metrics.r2_score(y3_test, gbm3_pred))



fig = plt.figure(figsize=(12, 6))

plt.scatter(y_test,gbm1_pred,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Gradiant Boosting prediction Performance (No feature selection)') 

plt.grid()

plt.show()
gbm_param_grid = {'learning_rate':[1,0.1, 0.01, 0.001], 

           'n_estimators':[100, 500, 1000],

          'max_depth':[3, 5, 8],

          'subsample':[0.7, 1], 

          'min_samples_leaf':[1, 20],

          'min_samples_split':[10, 20],

          'max_features':[4, 7]}



gbm_tuning = GridSearchCV(estimator =GradientBoostingRegressor(random_state=101),

                          param_grid = gbm_param_grid,

                          n_jobs=-1,

                          cv=5)



gbm_tuning.fit(X_train,y_train)

print(gbm_tuning.best_params_)
gbm_grid_predictions = gbm_tuning.predict(X_test)
print("Gradient Boosting with optimization performance")



print('MAE:', metrics.mean_absolute_error(y_test, gbm_grid_predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, gbm_grid_predictions)))

print('R2_Score: ', metrics.r2_score(y_test, gbm_grid_predictions))

fig = plt.figure(figsize=(12, 6))

plt.scatter(y_test,gbm_grid_predictions,color='coral', linewidths=2, edgecolors='k')

plt.xlabel('True GDP per Capita') 

plt.ylabel('Predictions') 

plt.title('Optimized Gradient Boosting prediction Performance') 

plt.grid()

plt.show()
data = {'Linear Regression (all features, No scaling)':[330027.15, 1568861.27, -29787.03],

        'Linear Regression (all features, with scaling)':[568426.40, 1281949.16, -19888.06],

        'Linear Regression (selected features, No scaling)':[2948.38, 4109.82, 0.80],

        'Linear Regression (selected features, with scaling)':[2854.65, 3760.87, 0.83],

        'Ridge Regression':[2854.61, 3760.74, 0.83],

       'Lasso Regression':[2852.57, 3784.96, 0.83],

       'SVM Regression (all features, No scaling)':[7049.98, 9811.74, -0.17],

       'SVM Regression (all features, with scaling)':[7042.73, 9800.41, -0.16],

       'SVM Regression (selected features, No scaling)':[7047.71, 9807.98, -0.16],

       'SVM Regression (selected features, with scaling)':[7040.05, 9794.57, -0.16],

       'Optimized SVM':[6386.99, 9131.48, -0.0091],

       'Random Forest (all features, No scaling)':[2127.80, 3065.70, 0.89],

       'Random Forest (selected features, No scaling)':[2462.71, 3630.00,  0.84],

       'Optimized Random Forest':[2329.73, 3180.77,  0.88],

       'Gradiant Boosting (all features, No scaling)':[2093.38, 3124.43,   0.88],

       'Gradiant Boosting (selected features, No scaling)':[2355.61, 3609.05,  0.84],

       'Optimized Gradiant Boosting':[2334.88, 3391.41,  0.86]}



df = pd.DataFrame(data)



df.index = ['MAE', 'RMSE', 'R2_Score'] 



print(df)
df_MAE = df.sort_values(by =['MAE'], axis=1,ascending=False)



print(df_MAE)
df_RMSE = df.sort_values(by =['RMSE'], axis=1,ascending=False)



print(df_RMSE)
df_R2_Score = df.sort_values(by =['R2_Score'], axis=1)



print(df_R2_Score)
best_model = {

        'Linear Regression':[2854.65, 3760.87, 0.83],

        'Ridge Regression':[2854.61, 3760.74, 0.83],

       'Lasso Regression':[2852.57, 3784.96, 0.83],

       'Optimized SVM':[6386.99, 9131.48, -0.0091],

       'Random Forest':[2127.80, 3065.70, 0.89],

       'Gradiant Boosting':[2093.38, 3124.43,   0.88]}



df_best_model = pd.DataFrame(best_model)



df_best_model.index = ['MAE', 'RMSE', 'R2_Score'] 



print(df_best_model)
ax = df_best_model.iloc[0].plot.bar(rot=90)
ax = df_best_model.iloc[1].plot.bar(rot=90)
ax = df_best_model.iloc[2].plot.bar(rot=90)