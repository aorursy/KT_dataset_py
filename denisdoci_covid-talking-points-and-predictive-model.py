import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#loading loading weather data

weatherAndCases = pd.read_csv('../input/covidandpopulationdata/training_data_with_weather_info_week_4.csv')

weatherAndCases.head()
#loading testing data 

covidTestingData = pd.read_csv('../input/covidandpopulationdata/owid-covid-data.csv')

covidTestingData.head()
#poplulation data 

populationdata = pd.read_csv('../input/covidandpopulationdata/WPP2019_TotalPopulationBySex.csv')

populationdata.head()
# all of our data is in 2020 and some columns are superflous 

populationdata = populationdata[populationdata['Time']==2020]

populationdata = populationdata[['Location','Time','PopMale','PopFemale','PopTotal','PopDensity']]

populationdata = populationdata.drop_duplicates()

populationdata.head()
#join the population and testing data

populationAndTesting = populationdata.merge(covidTestingData, left_on='Location', right_on='location')

populationAndTesting.head()
populationAndTesting.dtypes
populationAndTesting.head()
#clearly some countries dont match up so lets fix that



print(len(populationdata['Location'].unique()))

print(len(covidTestingData['location'].unique()))

print(len(populationAndTesting['Location'].unique()))

print(set(covidTestingData['location'])-set(populationAndTesting['Location']))
#most of these we can disregard since they are pretty small countries we wont use in our analyssi

#but some countries we cant look past

#United States, Hong Kong, Bolivia, Russia, etc.. 

#we will have to do a map and transform on these data points



def changecountryname(x):

    countryMap = {"United States of America":"United States",

             "China, Hong Kong SAR": "Hong Kong",

             "Bolivia (Plurinational State of)":"Bolivia",

             "Russian Federation":"Russia",

             "Viet Nam": "Vietnam",

             "China, Taiwan Province of China":"Taiwan",

             "Iran (Islamic Republic of)":"Iran",

             "Republic of Korea":"South Korea",

             "United Republic of Tanzania": "Tanzania",

             "Czechia":"Czech Republic",

             "Democratic Republic of the Congo":"Democratic Republic of Congo",

             "Syrian Arab Republic":"Syria",

             "Venezuela (Bolivarian Republic of)":"Venezuela"}

    if x in countryMap.keys():

        return countryMap[x]

    else:

        return x

populationdata['New_Country_Code'] = populationdata.Location.apply(lambda x: changecountryname(x))
populationdata.head()
#join the population and testing data

populationAndTesting = populationdata.merge(covidTestingData, left_on='New_Country_Code', right_on='location')

populationAndTesting.head()
#Test new missing set

print(set(covidTestingData['location'])-set(populationAndTesting['New_Country_Code']))
populationAndTesting.dtypes
#the weather data we're good with but we will be adding some 

#population information to standardize our points

populationandweather = populationdata.merge(weatherAndCases, left_on='New_Country_Code', right_on='Country_Region')

populationandweather.head()
#clearly some countries dont match up so lets fix that

print(len(populationdata['Location'].unique()))

print(len(weatherAndCases['Country_Region'].unique()))

print(len(populationandweather['Location'].unique()))

print(set(weatherAndCases['Country_Region'])-set(populationandweather['New_Country_Code']))
#diamond princess and MS Zaandam are a cruise liners so we obvioulsy dont need that

#from this list to simplify things we're only going to change the some of the countries



def changecountryname2(x):

    countryMap = {

        "United States":"US",

        "Taiwan":"Taiwan*",

        "South Korea":"Korea, South",

        "Czech Republic":"Czechia"}

    if x in countryMap.keys():

        return countryMap[x]

    else:

        return x

populationdata['New_Country_Code_Weather'] = populationdata.New_Country_Code.apply(lambda x: changecountryname2(x))
#the weather data we're good with but we will be adding some 

#population information to standardize our points

populationandweather = populationdata.merge(weatherAndCases, left_on='New_Country_Code_Weather', right_on='Country_Region')

populationandweather.head()
#clearly some countries dont match up so lets fix that

print(len(populationdata['Location'].unique()))

print(len(weatherAndCases['Country_Region'].unique()))

print(len(populationandweather['Location'].unique()))

print(set(weatherAndCases['Country_Region'])-set(populationandweather['New_Country_Code_Weather']))
from dateutil import parser

populationAndTesting['DTDate'] = populationAndTesting.date.apply(lambda x: parser.parse(x))

populationAndTesting.head()
populationAndTesting.dtypes
#cleaning up data

populationAndTesting = populationAndTesting[['Location','PopTotal','PopDensity',

                                             'total_cases','total_deaths',

                                             'total_tests','DTDate']]
ax = sns.lineplot(x="DTDate", y="total_deaths", hue="Location",

                  data=populationAndTesting)
populationAndTesting = populationAndTesting.sort_values(by='PopTotal', ascending=False)

populationAndTesting.head()
populationAndTestingWorld = populationAndTesting[populationAndTesting['Location']=='World']

populationAndTesting = populationAndTesting[populationAndTesting['Location']!='World']

populationAndTesting.head()
topCountries = populationAndTesting.Location.unique()[:25]

topCountries = np.concatenate((topCountries,['Sweden', 'Spain']))

topCountries
populationAndTestingTop = populationAndTesting[populationAndTesting['Location'].isin(topCountries)]

populationAndTestingTop.head()
ax = sns.lineplot(x="DTDate", y="total_deaths", hue="Location",

                  data=populationAndTestingTop)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = sns.lineplot(x="DTDate", y="total_tests", hue="Location",

                  data=populationAndTestingTop)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

#lets standardize total deaths and cases by population 

#population is in 1000

populationAndTestingTop['total_cases_perpopulation'] = populationAndTestingTop['total_cases'] / (1000*populationAndTestingTop['PopTotal'])

populationAndTestingTop['total_tests_perpopulation'] = populationAndTestingTop['total_tests'] / (1000*populationAndTestingTop['PopTotal'])

populationAndTestingTop['total_deaths_perpopulation'] = populationAndTestingTop['total_deaths'] / (1000*populationAndTestingTop['PopTotal'])

populationAndTestingTop.head()

ax = sns.lineplot(x="DTDate", y="total_cases_perpopulation", hue="Location",

                  data=populationAndTestingTop.dropna())

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = sns.lineplot(x="DTDate", y="total_tests_perpopulation", hue="Location",

                  data=populationAndTestingTop.dropna())

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

ax = sns.lineplot(x="DTDate", y="total_deaths_perpopulation", hue="Location",

                  data=populationAndTestingTop)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

#lets control deaths and cases for tests

populationAndTestingTop['deaths_per_test'] = populationAndTestingTop['total_deaths'] / populationAndTestingTop['total_tests']

populationAndTestingTop.head()

set(populationAndTestingTop.dropna().Location)

ax = sns.lineplot(x="DTDate", y="deaths_per_test", hue="Location",

                  data=populationAndTestingTop.dropna())

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.setp(ax.get_xticklabels(), fontsize=5)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

corr = populationAndTestingTop.corr()

corr 
mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
corr = populationAndTesting.corr()

corr 
mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax = sns.lineplot(x="PopDensity", y="deaths_per_test",

                  data=populationAndTestingTop.dropna())

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

#lets use our standarized deaths measure with the full dataset

populationAndTesting['total_deaths_perpopulation'] = populationAndTesting['total_deaths'] / (1000*populationAndTesting['PopTotal'])

populationAndTesting.head()

#lets create a new measure if a country is sweden or its not

populationAndTesting['Is_Sweden'] = populationAndTesting.Location.apply(lambda x: "Sweden" in x)

populationAndTesting.head()

set(populationAndTesting.Is_Sweden)
# load packages

import scipy.stats as stats

# stats f_oneway functions takes the groups as input and returns F and P-value

fvalue, pvalue = stats.f_oneway(populationAndTesting[populationAndTesting['Is_Sweden']==True].total_deaths_perpopulation,

                               populationAndTesting[populationAndTesting['Is_Sweden']==False].total_deaths_perpopulation)

print(fvalue, pvalue)
#lets instead take a subset of some of the most populous top 20 developed countries as based on the HDI index.

#this seems to be a more fair comparison



developed = ['Norway', 'Ireland', 'Germany', 

            'Australia', 'Iceland', 'Sweden',

            'Singapore', 'Netherlands', 'Denmark',

            'Finland', 'Canada', 'New Zealand',

            'United Kingdom', 'United States of America']

populationAndTestingDeveloped = populationAndTesting[populationAndTesting['Location'].isin(developed)]

populationAndTestingDeveloped.head()
#test to see if we got them all

set(populationAndTestingDeveloped.Location)-set(populationAndTestingDeveloped.Location)

#cool it worked
fvalue, pvalue = stats.f_oneway(populationAndTestingDeveloped[populationAndTestingDeveloped['Is_Sweden']==True].total_deaths_perpopulation,

                               populationAndTestingDeveloped[populationAndTestingDeveloped['Is_Sweden']==False].total_deaths_perpopulation)

print(fvalue, pvalue)
ax = sns.lineplot(x="DTDate", y="total_deaths_perpopulation", hue="Is_Sweden",

                  data=populationAndTestingDeveloped.dropna())

plt.setp(ax.get_xticklabels(), fontsize=7)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

#we have to address one annoying thing

populationandweather.dtypes
populationandweather = populationandweather[['Location','PopTotal','Province_State','Date',

                                             'ConfirmedCases','Fatalities','Lat','Long',

                                             'day_from_jan_first','temp','min',

                                             'max','stp','slp','dewp','rh',

                                             'ah','wdsp','prcp','fog']]

populationandweather.head()
set(populationandweather[populationandweather['Province_State'].notnull()].Location)
#Some of our data is more modular so we have to use state/province level data for 

#Australia, Canada, China, United States of America. The rest are just islands so we can honestly drop those



#australia we scraped from

#https://www.abs.gov.au/ausstats/abs@.nsf/mediareleasesbyCatalogue/CA1999BAEAA1A86ACA25765100098A47



#canada 

#https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000901



#china

#http://data.stats.gov.cn/english/easyquery.htm?cn=E0103



#usa

#https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html

australia = pd.read_csv('../input/covidandpopulationdata/australia.csv')

china = pd.read_csv('../input/covidandpopulationdata/china.csv')

canada = pd.read_csv('../input/covidandpopulationdata/canada.csv')

usa = pd.read_csv('../input/covidandpopulationdata/usa.csv')
populationandweather_usa = populationandweather[populationandweather['Location']=='United States of America']

populationandweather_canada = populationandweather[populationandweather['Location']=='Canada']

populationandweather_china = populationandweather[populationandweather['Location']=='China']

populationandweather_australia = populationandweather[populationandweather['Location']=='Australia']
populationandweather = populationandweather[populationandweather['Province_State'].isnull()]
populationandweather_usa.head()
usa.head()
#join the population and testing data

populationandweather_usa = populationandweather_usa.merge(usa, left_on='Province_State', right_on='state')

populationandweather_usa.head()
populationandweather_usa['PopTotal'] = populationandweather_usa['population']

populationandweather_usa = populationandweather_usa[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',

       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',

       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]

populationandweather_usa.head()

#test to see if we got them 

set(populationandweather_usa.Province_State) - set(usa.state)
china = china[['State','Population']]

china.head()
populationandweather_china.head()
#join the population and testing data

populationandweather_china = populationandweather_china.merge(china, left_on='Province_State', right_on='State')

populationandweather_china.head()
populationandweather_china['PopTotal'] = populationandweather_china['Population']

populationandweather_china = populationandweather_china[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',

       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',

       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]

populationandweather_china.head()

#test to see if we got them 

set(populationandweather_china.Province_State) - set(china.State)
australia = australia[['State','population']]

australia.head()
print(set(populationandweather_australia.Province_State))

populationandweather_australia.head()
#join the population and testing data

populationandweather_australia = populationandweather_australia.merge(australia, left_on='Province_State', right_on='State')

populationandweather_australia.head()

populationandweather_australia['PopTotal'] = populationandweather_australia['population']

populationandweather_australia = populationandweather_australia[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',

       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',

       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]

populationandweather_australia.head()

#test to see if we got them 

set(populationandweather_australia.Province_State) - set(australia.State)
canada = canada[['Province','Population']]

canada.head()
#join the population and testing data

populationandweather_canada = populationandweather_canada.merge(canada, left_on='Province_State', right_on='Province')

populationandweather_canada.head()

populationandweather_canada['PopTotal'] = populationandweather_canada['Population']

populationandweather_canada = populationandweather_canada[['Location', 'PopTotal', 'Province_State', 'Date', 'ConfirmedCases',

       'Fatalities', 'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',

       'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog']]

populationandweather_canada.head()

finalweather = populationandweather.append([populationandweather_canada,populationandweather_usa,

                                           populationandweather_australia,populationandweather_china])

finalweather.head()
finalweather['PopTotal']  = finalweather['PopTotal'].replace(',','', regex=True)

finalweather['PopTotal'] = finalweather['PopTotal'].astype(float)

finalweather['ConfirmedCasesPerCapita'] = finalweather['ConfirmedCases']/finalweather['PopTotal']

finalweather['DeathsPerCapita'] = finalweather['Fatalities']/finalweather['PopTotal']
finalweather['DTDate'] = finalweather.Date.apply(lambda x: parser.parse(x))
finalweather.columns
finalweather = finalweather[['PopTotal', 'ConfirmedCases','Fatalities', 

                             'Lat', 'Long', 'day_from_jan_first', 'temp', 'min', 'max',

                             'stp', 'slp', 'dewp', 'rh', 'ah', 'wdsp', 'prcp', 'fog',

       'ConfirmedCasesPerCapita', 'DeathsPerCapita']]
#impute missing data

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
finalweather = finalweather.replace([np.inf, -np.inf], np.nan)

imp = IterativeImputer(max_iter=10, random_state=0)

imp.fit(finalweather)

IterativeImputer(random_state=0)

x = imp.fit_transform(finalweather)

# x.head()

temp = pd.DataFrame(x, columns=finalweather.columns)

temp.head()
from sklearn.preprocessing import MinMaxScaler

#load scaler

scaler = MinMaxScaler()

scaler.fit(temp)

scaled =scaler.fit_transform(temp) 
scaleddf = pd.DataFrame(scaled, columns=finalweather.columns)

scaleddf.head()
scaleddf.dtypes
for i in scaleddf.columns:

    scaleddf[i] = scaleddf[i].astype(int)
scaleddf.dtypes


from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier



y = scaleddf['DeathsPerCapita'] 

X = scaleddf[['ConfirmedCases', 'Lat', 'Long', 

                  'day_from_jan_first', 'temp', 'min',

                  'max','stp', 'slp', 'dewp', 'rh', 'ah',

                  'wdsp', 'prcp', 'fog']]



# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
#lets do the same check but for cases

from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier



y = scaleddf['ConfirmedCases'] 

X = scaleddf[['DeathsPerCapita', 'Lat', 'Long', 

                  'day_from_jan_first', 'temp', 'min',

                  'max','stp', 'slp', 'dewp', 'rh', 'ah',

                  'wdsp', 'prcp', 'fog']]



# Build a forest and compute the feature importances

forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(X.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)

plt.xlim([-1, X.shape[1]])

plt.show()
scaleddf = pd.DataFrame(scaled, columns=finalweather.columns)

scaleddf.head()
scaleddf.columns
scaleddf.head()
#create a testing and training model 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(scaleddf.drop(['DeathsPerCapita'],axis=1), scaleddf['DeathsPerCapita'], test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

pred = reg.predict(X_test)
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_validate

import statistics 



print(mean_absolute_error(pred, y_test))

scores = cross_validate(reg, X_test, y_test, cv=10,

                        scoring=('neg_root_mean_squared_error'),

                        return_train_score=True)  

statistics.mean(abs(scores['test_score']))
#multi layer perceptron



from sklearn.neural_network import MLPRegressor

neuralNetwork = MLPRegressor(

    hidden_layer_sizes=(550, 550),

    shuffle=True, activation='relu',

    learning_rate='adaptive')



neuralNetwork.fit(X_train, y_train)



pred_y_test = neuralNetwork.predict(X_test)

pred_y_train = neuralNetwork.predict(X_train)
print(mean_absolute_error(pred_y_test, y_test))

scores = cross_validate(neuralNetwork, X_test, y_test, cv=10,

                        scoring=('neg_root_mean_squared_error'),

                        return_train_score=True)  

statistics.mean(abs(scores['test_score']))
import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



print(tf.__version__)

def build_model():

  model = keras.Sequential([

    layers.Dense(90, activation='relu', input_shape=[len(X_train.keys())]),

    layers.Dense(90, activation='relu'),

    layers.Dense(1)

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)



  model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

  return model
model = build_model()

history = model.fit(X_train, y_train,

                    batch_size=64,

                    epochs=50)
model.summary()

# test_predictions = history.predict(X_test).flatten()

test_predictions = model.predict(X_test).flatten()



print(mean_absolute_error(test_predictions, y_test))
