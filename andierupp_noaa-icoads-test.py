

from google.cloud import bigquery

import sys 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

#access cloud dataset with google bigquery

client = bigquery.Client()

dataset_ref = client.dataset('noaa_icoads', project = 'bigquery-public-data')

dset = client.get_dataset(dataset_ref)



icoads_core_2015 = client.get_table(dset.table('icoads_core_2015'))

#print([i.name+',type:'+i.field_type for i in icoads_core_2015.schema])



#select data in El Nino 3.4 region



QUERY = """

        SELECT latitude, longitude, sea_surface_temp, wind_direction_true, amt_pressure_tend, air_temperature, sea_level_pressure, wave_direction, wave_height, timestamp

        FROM bigquery-public-data.noaa_icoads.icoads_core_2015

        WHERE longitude >= -170 AND longitude <= -120 AND latitude >= -5 AND latitude <= 5 

        """



import pandas as pd

#use query to create dataframe

df = client.query(QUERY).to_dataframe()

#print(df.head(10))



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing, svm

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



#specifiying which columns to use

elNinoData = df[['sea_surface_temp','amt_pressure_tend']]

elNinoData.columns = ['SST','Pressure']

data = elNinoData

#print(elNinoData.head())



#look for relationship between sst and pressure, both signs of el nino activity

#print(data.describe())

sns.lmplot(x = 'SST',y = 'Pressure', data = elNinoData, ci = None)

plt.show()





#not showing a clear relation

#look at sea level pressure and temp

data2 = df[['sea_level_pressure', 'sea_surface_temp']]

data2.columns = ['SLP','SST']





sns.lmplot(x = 'SLP', y='SST', data=data2)
#not showing clear relation either

#sea surface temp v. air temp



data3 = df[['sea_surface_temp','air_temperature']]

data3.columns = ['SST','AirTemp']

sns.lmplot(x='SST',y = 'AirTemp',data = data3)
#appears to have positive correlation

#find and get rid of null values



data3.isnull().sum()

#new_data = data3.dropna(axis = 1)

data3.fillna(method = 'ffill', inplace = True)




#create tensors

x = np.array(data3['SST']).reshape(-1,1)

y = np.array(data3['AirTemp']).reshape(-1,1)



#choose parameters to split data

validation_size = 0.01

seed = 0
#transform null, inf, NaN values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

x = my_imputer.fit_transform(x)

y = my_imputer.fit_transform(y)
#split data and use linear regression model for first round



x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = validation_size, random_state = seed)

#np.isnan(x_train)

#np.isnan(y_train)

model_UnitTest = LinearRegression()

model_UnitTest.fit(x_train, y_train)
#predict and evaluate accuracy

y_pred = model_UnitTest.predict(x_test)

accuracy = model_UnitTest.score(x_test,y_test)

print("Linear Regression Model Accuracy:"+"{:.1%}".format(accuracy))

print("R2 Score:"+"{}".format(r2_score(y_test,y_pred)))
#plot test v. prediction

plt.scatter(x_test, y_test)

plt.plot(x_test, y_pred)
#plot another subset of test v. prediction

x_new = [[30],[36]]

y_pred = model_UnitTest.predict(x_new)

plt.scatter(x_train,y_train, color = 'r')

plt.plot(x_new,y_pred)
#create a list of model names and models

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))
#split data and set seed

validation_size2 = 0.2

seed2 = 5

x_train2, x_test2, y_train2, y_test2 = train_test_split(x,y, test_size = validation_size2, random_state = seed2)

y_train2[1]



y_train2_int_type = y_train2.astype('int')
#run models in a loop and get the accuracy of each

scoring = 'accuracy'

results = []

names = []

for name, model in models:

        kfold = model_selection.KFold(n_splits = 10, random_state = seed2, shuffle = True)

        cv_results = model_selection.cross_val_score(model, x_train2, y_train2_int_type.ravel(), cv = kfold, scoring = scoring)

        results.append(cv_results)

        names.append(name)

        result = '%s: %f (%f)'%(name, cv_results.mean(), cv_results.std())

        print(result)
#compare the models

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

plt.plot([1,2,3])

ax = fig.add_subplot(211)

ax.set_xticklabels(names)

plt.show()
#pick CART/decision tree classifier as best model and use it

model = DecisionTreeClassifier()

model.fit(x_train2, y_train2_int_type.ravel())
#generate prediction and give r score

y_pred2 = model.predict(x_train2)

print("R2 Score:"+ "{:.3}".format(r2_score(y_train2, y_pred2.ravel())))
