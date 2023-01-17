import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split  #Split file

import geopandas

import matplotlib

# Input data files are available in the "../input/" directory.

import os

#print(os.listdir("../input"))

weather_data = pd.read_csv("../input/weatherAUS.csv") 

# Preview the first 5 lines of the loaded data 

weather_data.head()
## Let's what is inside

weather_data["Date"].head()
#Change datatype of Date column

weather_data["Date"] = pd.to_datetime(weather_data["Date"])

## Data discovering

print("Maximum date :: ",weather_data["Date"].max() )

print("Minimun date :: ", weather_data["Date"].min())

print("Count lines :: ", weather_data["Date"].count())
## According to the number of data per column, it's unnecessary to keep if less that 60%

number_lines = weather_data.count().max()

calc_col = weather_data.count().sort_values()/(number_lines)

calc_col.apply(lambda x :'OK' if x > 0.6 else 'NOK') 



## deleting columns under 60% and RISK_MM due to be itself a sort of prediction

weather_data = weather_data.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','RISK_MM','Date'],axis=1)



## drop null values

weather_data = weather_data.dropna(how='any')



##discovering outliers

import seaborn as sns

sns.boxplot(x=weather_data['MaxTemp'])
##Discover outliers with mathematical function   Z-Score

from scipy import stats

z = np.abs(stats.zscore(weather_data._get_numeric_data()))

print(z)

weather_data= weather_data[(z < 3).all(axis=1)]

## Transforming categorical column

weather_data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

weather_data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am','Location']

for col in categorical_columns:

    print(np.unique(weather_data[col]))

# transform the categorical columns

weather_data = pd.get_dummies(weather_data, columns=categorical_columns)

## Letting all in the same Magnitude with preprocessing from SKLEARN

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(weather_data)

weather_data = pd.DataFrame(scaler.transform(weather_data), index=weather_data.index, columns=weather_data.columns)

weather_data.head()
##Selecting the best features using SelectKBest



from sklearn.feature_selection import SelectKBest, chi2

X = weather_data.loc[:,weather_data.columns!='RainTomorrow']

y = weather_data['RainTomorrow']

selector = SelectKBest(chi2, k=5)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)])
X = weather_data[['Rainfall', 'Humidity9am', 'Humidity3pm', 'RainToday', 'WindDir9am_N']] 

y = weather_data[['RainTomorrow']]





X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
## Comparing regression models



import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# prepare configuration for cross validation test harness

seed = 7

# prepare models

models = []

models.append(('LogisticRegression', LogisticRegression()))

models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))

models.append(('KNeighborsClassifier', KNeighborsClassifier()))

models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

models.append(('GaussianNB', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

fig.set_figheight(7)

fig.set_figwidth(14)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline  

import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
fig = plt.figure(figsize=(25, 25))

m = Basemap(projection='lcc',resolution='c',

            width=8E6, height=5E6, 

            lat_0=-25, lon_0=133,)

m.etopo(scale=0.5, alpha=0.5)



x,y = m(144.946457,-37.840935)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Melbourne, VIC', fontsize=12)

x,y = m(138.599503,-34.921230)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Adelaide, SA', fontsize=12)

x,y = m(147.157135,-41.429825)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Launceston, TAS', fontsize=12)

x,y = m(138.593903,-34.906101)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'North Adelaide, SA', fontsize=12)

x,y = m(146.816956,-19.258965)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Townsville City, QLD', fontsize=12)

x,y = m(145.754120,-16.925491)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Cairns City, QLD', fontsize=12)

x,y = m(115.857048,-31.953512)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Perth, WA', fontsize=12)

x,y = m(142.136490,-34.206841)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Mildura, VIC', fontsize=12)

x,y = m(144.880600,-37.649967)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Ziyou Today, Greenvale, Victoria', fontsize=12)

x,y = m(153.114136,-30.296276)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Coffs Harbour NSW 2450', fontsize=12)

x,y = m(149.101273,-33.283577)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Orange, NSW', fontsize=12)

x,y = m(144.278702,-36.757786)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Bendigo, VIC', fontsize=12)

x,y = m(146.916473,-36.080780)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Albury, NSW', fontsize=12)

x,y = m(150.893143,-34.425072)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Wollongong, NSW', fontsize=12)

x,y = m(130.841782,-12.462827)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Darwin, Northern Territory', fontsize=12)

x,y = m(151.224396,-33.683212)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Terrey Hills, NSW', fontsize=12)

x,y = m(151.035889,-33.917290)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Bankstown NSW', fontsize=12)

x,y = m(150.987274,-33.807690)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Westmead, NSW', fontsize=12)

x,y = m(153.021072,-27.470125)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Brisbane, QLD', fontsize=12)

x,y = m(151.268356,-23.843138)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Gladstone QLD', fontsize=12)

x,y = m(149.082977,-35.343784)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Phillip ACT, Canberra', fontsize=12)

x,y = m(151.342224,-33.425018)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Gosford, NSW', fontsize=12)

x,y = m(140.783783,-37.824429)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Mount Gambier, SA', fontsize=12)

x,y = m(151.209900,-33.865143)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Sydney, NSW', fontsize=12)

x,y = m(152.407181,-27.529953)

plt.plot(x, y, 'bo', markersize=5)

plt.text(x,y,'Glenore Grove, Queensland', fontsize=12)
