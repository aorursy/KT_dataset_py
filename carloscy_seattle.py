# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from scipy.stats.stats import pearsonr  

from pandas.tseries.holiday import USFederalHolidayCalendar

from pandas.tseries.offsets import CustomBusinessDay

from datetime import datetime

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from matplotlib import colors

import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

import sklearn.metrics

from statsmodels.stats.anova import anova_lm

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import LeaveOneOut

from sklearn.cluster import KMeans

from sklearn.cluster import MeanShift, estimate_bandwidth

from sklearn.mixture import GaussianMixture
trips = pd.read_csv('../input/trip.csv', error_bad_lines=False)

trips.head()
trips.shape
trips.isnull().sum()


trips.starttime = pd.to_datetime(trips.starttime, format='%m/%d/%Y %H:%M')


trips['date'] = trips.starttime.dt.date
dates = {}

for d in trips.date:

    if d not in dates:

        dates[d] = 1

    else:

        dates[d] += 1
df2 = pd.DataFrame.from_dict(dates, orient = "index")

df2['date'] = df2.index

df2['trips'] = df2.ix[:,0]

train = df2.ix[:,1:3]

train.reset_index(drop = True, inplace = True)
train.head()
weather = pd.read_csv('../input/weather.csv')
weather.shape
weather.head()
weather.isnull().sum()
weather.dtypes


weather.Date = pd.to_datetime(weather.Date, format='%m/%d/%Y')
print (train.shape)

print (weather.shape)
weather.Events.unique()
weather.loc[weather.Events == 'Rain , Snow', 'Events'] = "Sleet"

weather.loc[weather.Events == 'Rain-Snow', 'Events'] = "Sleet"

weather.loc[weather.Events == 'Rain , Thunderstorm', 'Events'] = "Rain"

weather.loc[weather.Events == 'Rain-Thunderstorm', 'Events'] = "Rain"

weather.loc[weather.Events == 'Fog , Rain', 'Events'] = "Fog"

weather.loc[weather.Events == 'Fog-Rain', 'Events'] = "Fog"

weather.loc[weather.Events.isnull(), 'Events'] = "Normal"

Events = pd.get_dummies(weather.Events)

weather = weather.merge(Events, left_index = True, right_index = True)
weather = weather.drop(['Events'],1)
weather.loc[weather.Max_Gust_Speed_MPH == '-', 'Max_Gust_Speed_MPH'] = "0"

weather.Max_Gust_Speed_MPH = weather['Max_Gust_Speed_MPH'].astype(float)

weather.loc[weather.Max_Gust_Speed_MPH.isnull(), 'Max_Gust_Speed_MPH'] = weather.groupby('Max_Wind_Speed_MPH').Max_Gust_Speed_MPH.apply(lambda x: x.fillna(x.median()))
mv = np.mean(weather.Mean_Temperature_F)

weather.loc[weather.Mean_Temperature_F.isnull(), 'Mean_Temperature_F'] = mv

train = train.merge(weather, on = train.date)
train.head()
train.drop(['key_0','date'],1, inplace= True)

#train.rename(columns={'date':'Date'}, inplace=True)

train.head()
calendar = USFederalHolidayCalendar()

holidays = calendar.holidays(start=train.Date.min(), end=train.Date.max())
#Encontrar todos los dias de trabajo en nuestra data

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

business_days = pd.DatetimeIndex(start=train.Date.min(), end=train.Date.max(), freq=us_bd)



#crear data frame de ellos

business_days = pd.to_datetime(business_days, format='%Y/%m/%d').date

holidays = pd.to_datetime(holidays, format='%Y/%m/%d').date
#hacer el match con nuestra data

train['business_day'] = train.Date.isin(business_days)

train['holiday'] = train.Date.isin(holidays)
#Convertir True a 1 y False a 0

train.business_day = train.business_day.map(lambda x: 1 if x == True else 0)

train.holiday = train.holiday.map(lambda x: 1 if x == True else 0)

train.head()
train.sort_values(by=['trips'], ascending = False).head() 
sns.distplot(train['trips'])
sns.boxplot(data=train[['trips']])
cols = list(train)

sns.set(font_scale=.75)



fig, ax = plt.subplots(figsize=(13,13))

sns.heatmap(train.corr(), ax=ax, cbar=True, annot=True, fmt=".2f", square=True, yticklabels=cols, xticklabels=cols)

plt.show()
train.corr()[['trips']].sort_values('trips') 
target = train.trips

df = train

df = df.drop(columns = 'Date')

train = train.drop(['trips', 'Date'], 1)
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state = 42)

print("Train X set size: ", X_train.shape)

print("Train Y set size: ", y_train.shape)

print('')

print("Test X set size: ", X_test.shape)

print("Test Y set size: ", y_test.shape)
# calculate the intercept of our model

import statsmodels.api as sm

u = sm.add_constant(X_train)


simple_model = sm.OLS(y_train,X_train['Mean_Temperature_F'])

simple_result = simple_model.fit()
print(simple_result.summary())
#X_test = sm.add_constant(X_test)

y_pred_simple = simple_result.predict(X_test['Mean_Temperature_F'])

sns.scatterplot(x = X_test['Mean_Temperature_F'], y = y_test.values.ravel())

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_pred_simple)

plt.title('Regression Trips - Mean Temperature')

plt.xlabel("Mean Temperature ºF")

plt.ylabel("Trips")
simple_model = sm.OLS(y_train,X_train['Mean_Humidity'])

simple_result = simple_model.fit()
print(simple_result.summary())
#X_test = sm.add_constant(X_test)

y_pred_simple = simple_result.predict(X_test['Mean_Humidity'])

sns.scatterplot(x = X_test['Mean_Humidity'], y = y_test.values.ravel())

sns.lineplot(x = X_test['Mean_Humidity'] , y = y_pred_simple)

plt.title('Regression Trips - Mean Humidity')

plt.xlabel("Mean Humidity")

plt.ylabel("Trips")


simple_model = sm.OLS(y_train,X_train['Precipitation_In'])

simple_result = simple_model.fit()

print(simple_result.summary())
#X_test = sm.add_constant(X_test)

y_pred_simple = simple_result.predict(X_test['Precipitation_In'])

sns.scatterplot(x = X_test['Precipitation_In'], y = y_test.values.ravel())

sns.lineplot(x = X_test['Precipitation_In'] , y = y_pred_simple)

plt.title('Regression plot')

plt.title('Regression Trips - Precipitation Inch')

plt.xlabel("Precipitation Inch")

plt.ylabel("Trips")
multiple_model = sm.OLS(y_train,X_train)

multiple_result = multiple_model.fit()

print(multiple_result.summary())
#debido a que el P-value de alguna variables es mayor a 0.05, no parecen ser relevantes para nuestro modelo

X_train = X_train.drop(columns =['Max_Temperature_F','Min_TemperatureF','Max_Dew_Point_F','Min_Dewpoint_F',

                                'Max_Sea_Level_Pressure_In','Mean_Visibility_Miles','Max_Wind_Speed_MPH',

                                'Mean_Wind_Speed_MPH','Max_Gust_Speed_MPH','Normal','Sleet','Snow','holiday'])



X_test = X_test.drop(columns =['Max_Temperature_F','Min_TemperatureF','Max_Dew_Point_F','Min_Dewpoint_F',

                                'Max_Sea_Level_Pressure_In','Mean_Visibility_Miles','Max_Wind_Speed_MPH',

                                'Mean_Wind_Speed_MPH','Max_Gust_Speed_MPH','Normal','Sleet','Snow','holiday'])
multiple_model = sm.OLS(y_train,X_train)

multiple_result = multiple_model.fit()

print(multiple_result.summary())
simple_model = sm.OLS(y_train,X_train[['Mean_Temperature_F']])

simple_result = simple_model.fit()
#predecir y graficar el resultado

X_test = sm.add_constant(X_test)

y_pred_simple = simple_result.predict(X_test['Mean_Temperature_F'])

sns.scatterplot(x = X_test['Mean_Temperature_F'], y = y_test.values.ravel(), color='mediumturquoise')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_pred_simple, color='y')

plt.title('Regression plot')

plt.title('Regression Trips - Mean Temperature ºF (Grado 1)')

plt.xlabel("Mean Temperature ºF")

plt.ylabel("Trips")
#Imprimir tabla de evaluation metrics

print(simple_result.summary().tables[0])
# Create polynomial regression features of nth degree

poly_reg = PolynomialFeatures(degree = 2)

X_poly_train = poly_reg.fit_transform(pd.DataFrame(X_train['Mean_Temperature_F']))

X_poly_test = poly_reg.fit_transform(pd.DataFrame(X_test['Mean_Temperature_F']))

poly_result = poly_reg.fit(X_poly_train, y_train)

    

# Fit linear model now polynomial features

poly_model = LinearRegression()

poly_result = poly_model.fit(X_poly_train, y_train) 

y_poly_pred = poly_model.predict(X_poly_test)

    

# Plot to compare models

sns.scatterplot(x = X_test['Mean_Temperature_F'], y = y_test.values.ravel(), color='mediumturquoise')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_pred_simple, color='y')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_poly_pred.ravel(), color='crimson')

plt.title('Regression Trips - Mean Temperature ºF (Grado 2)')

plt.xlabel("Mean Temperature ºF")

plt.ylabel("Trips")

# Retrain to be able to print summary table

poly_model = sm.OLS(y_train,X_poly_train)

poly_result = poly_model.fit()



# Print model evaluation metrics

print(poly_result.summary().tables[0])
# Comparar los dos modelos usando ANOVA test

anovaResults = anova_lm(simple_result, poly_result)

print(anovaResults)
# Create polynomial regression features of nth degree

poly_reg = PolynomialFeatures(degree = 3)

X_poly_train = poly_reg.fit_transform(pd.DataFrame(X_train['Mean_Temperature_F']))

X_poly_test = poly_reg.fit_transform(pd.DataFrame(X_test['Mean_Temperature_F']))

poly_result = poly_reg.fit(X_poly_train, y_train)

    

# Fit linear model now polynomial features

poly_model = LinearRegression()

poly_result = poly_model.fit(X_poly_train, y_train)

y_poly_pred = poly_model.predict(X_poly_test)

    

# Plot to compare models

sns.scatterplot(x = X_test['Mean_Temperature_F'], y = y_test.values.ravel(), color='mediumturquoise')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_pred_simple, color='y')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_poly_pred.ravel(), color='crimson')

plt.title('Regression Trips - Mean Temperature ºF (Grado 3)')

plt.xlabel("Mean Temperature ºF")

plt.ylabel("Trips")
# Retrain to be able to print summary table

poly_model = sm.OLS(y_train,X_poly_train)

poly_result = poly_model.fit()



# Print model evaluation metrics

print(poly_result.summary().tables[0])
# Compare the two models using an ANOVA test

anovaResults = anova_lm(simple_result, poly_result)

print(anovaResults)
# Create polynomial regression features of nth degree

poly_reg = PolynomialFeatures(degree = 5)

X_poly_train = poly_reg.fit_transform(pd.DataFrame(X_train['Mean_Temperature_F']))

X_poly_test = poly_reg.fit_transform(pd.DataFrame(X_test['Mean_Temperature_F']))

poly_result = poly_reg.fit(X_poly_train, y_train)

    

# Fit linear model now polynomial features

poly_model = LinearRegression()

poly_result = poly_model.fit(X_poly_train, y_train)

y_poly_pred = poly_model.predict(X_poly_test)

    

# Plot to compare models

sns.scatterplot(x = X_test['Mean_Temperature_F'], y = y_test.values.ravel(), color='mediumturquoise')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_pred_simple, color='y')

sns.lineplot(x = X_test['Mean_Temperature_F'] , y = y_poly_pred.ravel(), color='crimson')

plt.title('Regression Trips - Mean Temperature ºF (Grado 5)')

plt.xlabel("Mean Temperature ºF")

plt.ylabel("Trips")
# Retrain to be able to print summary table

poly_model = sm.OLS(y_train,X_poly_train)

poly_result = poly_model.fit()



# Print model evaluation metrics

print(poly_result.summary().tables[0])
clf = DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
X_test = X_test.drop(columns = 'const')

y_pred = clf.predict(X_test)
accScr = []

accScr.append(accuracy_score(y_pred, y_test))

accScr[0]
#n_estimators = Numero de arboles

#n_jobs = cuantos procesadores queremos usar (-1 = todos) 

clf = RandomForestRegressor(n_estimators=2, n_jobs=-1)
clf.fit(X_train, y_train)
rndF = []

rndfRMSE = []

rndF.append(clf.score(X_test,y_test))

tree_predictions = clf.predict(X_test)

rndfRMSE.append(np.sqrt(mean_squared_error(y_test, tree_predictions)))

# con un mayor numero de arboles

clf = RandomForestRegressor(n_estimators=100, n_jobs=-1)

clf.fit(X_train, y_train)

rndF.append(clf.score(X_test,y_test))

tree_predictions = clf.predict(X_test)

rndfRMSE.append(np.sqrt(mean_squared_error(y_test, tree_predictions)))
clf = RandomForestRegressor(n_estimators=200, n_jobs=-1)

clf.fit(X_train, y_train)

rndF.append(clf.score(X_test,y_test))

tree_predictions = clf.predict(X_test)

rndfRMSE.append(np.sqrt(mean_squared_error(y_test, tree_predictions)))
clf = RandomForestRegressor(n_estimators=300, n_jobs=-1)

clf.fit(X_train, y_train)

rndF.append(clf.score(X_test,y_test))

tree_predictions = clf.predict(X_test)

rndfRMSE.append(np.sqrt(mean_squared_error(y_test, tree_predictions)))
sns.set()

sns.lineplot(x = [2,100,200,300] , y = rndF, color= '#c0e033')

plt.title('R^2 con número de árboles')

plt.xlabel("Numbero de Árboles")

plt.ylabel("R^2")
sns.set()

sns.lineplot(x = [2,100,200,300] , y = rndfRMSE, color= '#c0e033')

plt.title('RMSE con número de árboles')

plt.xlabel("Numbero de Árboles")

plt.ylabel("RMSE")
adaboost_reg = AdaBoostRegressor(n_estimators=50, learning_rate=1, loss='linear')

adaboost_reg.fit(X_train, y_train)
# Get training and test predictions

prediction_train = adaboost_reg.score(X_train,y_train)

prediction_test = adaboost_reg.score(X_test,y_test)

adb = []

adbRMSE = []

print('Prediction Train: ',prediction_train)

print('Prediction Test: ',prediction_test)

adb.append(prediction_test)

adb_predictions = adaboost_reg.predict(X_test)

adbRMSE.append(np.sqrt(mean_squared_error(y_test, adb_predictions)))
#Mayor numero de estimators

adaboost_reg = AdaBoostRegressor(n_estimators=190, learning_rate=1, loss='linear')

adaboost_reg.fit(X_train, y_train)

prediction_test = adaboost_reg.score(X_test,y_test)

print('Prediction Train: ',prediction_train)

print('Prediction Test: ',prediction_test)

adb.append(prediction_test)

adb_predictions = adaboost_reg.predict(X_test)

adbRMSE.append(np.sqrt(mean_squared_error(y_test, adb_predictions)))
#Mayor numero de estimators

adaboost_reg = AdaBoostRegressor(n_estimators=230, learning_rate=1, loss='linear')

adaboost_reg.fit(X_train, y_train)

prediction_test = adaboost_reg.score(X_test,y_test)

print('Prediction Train: ',prediction_train)

print('Prediction Test: ',prediction_test)

adb.append(prediction_test)

adb_predictions = adaboost_reg.predict(X_test)

adbRMSE.append(np.sqrt(mean_squared_error(y_test, adb_predictions)))
sns.set()

sns.lineplot(x = [50,190, 230] , y = adb, color= '#c0e033')

plt.title('R^2 con número de árboles')

plt.xlabel("Numbero de Árboles")

plt.ylabel("R^2")
sns.set()

sns.lineplot(x = [50,190, 230] , y = adbRMSE , color= '#c0e033')

plt.title('RMSE con número de árboles')

plt.xlabel("Numbero de Árboles")

plt.ylabel("RMSE")
from keras import Sequential

from keras.layers import Dense

def build_regressor():

    regressor = Sequential()

    regressor.add(Dense(units=13, input_dim=13))

    regressor.add(Dense(units=1))

    #loss function: mean squared error 

    #metrics: mean absolute error & accuracy.

    regressor.compile(optimizer= 'adam',

                      loss= 'mean_squared_error',

                      metrics=['mae','accuracy'])

    return regressor
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=100)
results = regressor.fit(X_train,y_train)
def plot_training_curves(history):

    """

    Plot accuracy and loss curves for training and validation sets.

    Args:

        history: a Keras History.history dictionary

    Returns:

        mpl figure.

    """

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(18,5))

    if 'acc' in history:

        ax_acc.plot(history['acc'], label='acc')

        if 'val_acc' in history:

            ax_acc.plot(history['val_acc'], label='Val acc')

        ax_acc.set_xlabel('epoch')

        ax_acc.set_ylabel('accuracy')

        ax_acc.legend(loc='upper left')

        ax_acc.set_title('Accuracy')



    ax_loss.plot(history['loss'], label='loss')

    if 'val_loss' in history:

        ax_loss.plot(history['val_loss'], label='Val loss')

    ax_loss.set_xlabel('epoch')

    ax_loss.set_ylabel('loss')

    ax_loss.legend(loc='upper right')

    ax_loss.set_title('Loss')



    sns.despine(fig)

    return
plot_training_curves(results.history)
y_pred= regressor.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Viajes del Dataset')

ax.set_ylabel('Viajes Predecidos')

plt.show()
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=500)

results = regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)
plot_training_curves(results.history)
np.sqrt(mean_squared_error(y_test, y_pred))
fig, ax = plt.subplots()

ax.scatter(y_test, y_pred)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

ax.set_xlabel('Viajes del Dataset')

ax.set_ylabel('Viajes Predecidos')

plt.show()
from keras.wrappers.scikit_learn import KerasRegressor

regressor = KerasRegressor(build_fn=build_regressor, batch_size=32,epochs=1000)

results = regressor.fit(X_train,y_train)

y_pred= regressor.predict(X_test)
plot_training_curves(results.history)
np.sqrt(mean_squared_error(y_test, y_pred))