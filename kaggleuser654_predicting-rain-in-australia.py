# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading in csv file and changing dtype of 'Date' column to datetime

weather = pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
#dataset info

weather.info()
#timespan of data

print("The weather data was collected over the time period {0} to {1}.".format(weather.loc[0, 'Date'], weather.loc[142192, 'Date']))
#number of missing values in each variable

weather.isnull().sum().sort_values(ascending=False)
#list of columns

features = weather.columns.tolist()

#check

features
#drop target variable from list of features

features.remove('RainTomorrow')
#drop RISK_MM from list of features

features.remove('RISK_MM')
#how many missing values in 'RainToday'

print('There are {} missing values in "Rain Today", and they make up {:.2f}% of the number of entries.'

      .format(weather['RainToday'].isnull().sum(),(weather['RainToday'].isnull().sum())/(weather.shape[0])*100))
#drop rows with missing values in 'RainToday'

weather = weather[weather['RainToday'].notnull()]

#check

weather['RainToday'].unique()
#any missing values in the target variable?

weather['RainTomorrow'].isnull().sum()
#for checking purposes

RToday_values_before = weather['RainToday'].value_counts()

RTomorrow_values_before = weather['RainTomorrow'].value_counts()

print('Value count in "Rain Today":\n {}'.format(RToday_values_before))

print('Value count in "Rain Tomorrow":\n {}'.format(RTomorrow_values_before))
#binarise 'RainToday' and 'RainTomorrow'

weather['RainToday'] = (weather['RainToday'] == 'Yes')*1

weather['RainTomorrow'] = (weather['RainTomorrow'] == 'Yes')*1
#check

print('Value count in "Rain Today":\n {}'.format(weather['RainToday'].value_counts()))

print('\nValue count in "Rain Tomorrow":\n {}'.format(weather['RainTomorrow'].value_counts()))
before_rows = weather.shape[0]

copy_data = weather.copy()

mod_data = copy_data.dropna()

after_rows = mod_data.shape[0]

print('{} rows are dropped.'.format(before_rows - after_rows))

print('\nDropping rows with at least one missing value will reduce our dataset by {:.2f} %.'.format((before_rows - after_rows)/before_rows*100))
weather[features].describe()
plt.figure(figsize=(15,10))

cont_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation','Sunshine',

                 'WindGustSpeed','WindSpeed9am','WindSpeed3pm', 'Humidity9am',

                'Humidity3pm', 'Pressure9am', 'Pressure3pm','Cloud9am', 'Cloud3pm',

                'Temp9am','Temp3pm']

corr_weather = weather[cont_features].corr()

mask = np.triu(np.ones_like(corr_weather, dtype=bool))

sns.heatmap(corr_weather, mask=mask,annot=True, cmap='YlOrRd')

plt.show()
#'RainTomorrow' value counts

weather['RainTomorrow'].value_counts()
#pairwise scatter plots of temperature variables

plt.figure(figsize=(15,15))

sns.pairplot(weather[['MinTemp','MaxTemp','Temp9am','Temp3pm','RainTomorrow']], hue = 'RainTomorrow' )

plt.show()
#pairwise scatter plots of 'Sunshine','Cloud9am,'Cloud3pm', and 'Evaporation'

plt.figure(figsize=(15,15))

sns.pairplot(weather[['Sunshine','Evaporation','Cloud9am','Cloud3pm','RainTomorrow']], hue = 'RainTomorrow' )

plt.show()
#pairwise scatter plots of wind variables

plt.figure(figsize=(15,15))

sns.pairplot(weather[['WindGustSpeed','WindSpeed9am','WindSpeed3pm','RainTomorrow']], hue = 'RainTomorrow' )

plt.show()
#pairwise scatter plots of rainfall, pressure and humidity variables

plt.figure(figsize=(15,15))

sns.pairplot(weather[['Rainfall','Humidity9am','Humidity3pm','Pressure9am', 'Pressure3pm','RainTomorrow']], hue = 'RainTomorrow' )

plt.show()
#scatter plot of'RainToday' and 'Rainfall'

plt.figure(figsize=(10,5))

sns.scatterplot(weather['Rainfall'],weather['RainToday'])

plt.show()
#dropping listed features

to_drop = ['MinTemp', 'MaxTemp', 'Temp9am', 'Evaporation', 'RainToday','WindSpeed9am',

           'WindSpeed3pm','Humidity9am','Pressure9am']

for col in to_drop:

    features.remove(col)

#check

features
import datetime

weather['Year'] = pd.DatetimeIndex(weather['Date']).year

weather['Season'] = pd.DatetimeIndex(weather['Date']).month

weather['Season'].replace({1: 'Summer', 2:'Summer', 3:'Spring', 4:'Spring', 5:'Spring', 6:'Winter',

                          7:'Winter', 8:'Winter', 9:'Autumn', 10:'Autumn', 11:'Autumn', 12:'Summer'}, inplace=True)

#check

weather['Season'].unique()
#graphs

weather.pivot_table('Cloud3pm', index='Season', columns='Year', aggfunc='mean').plot(figsize=(10,7), title = 'Cloud3pm')

weather.pivot_table('Cloud9am', index='Season', columns='Year', aggfunc='mean').plot(figsize=(10,7), title = 'Cloud9am')

weather.pivot_table('Sunshine', index='Season', columns='Year', aggfunc='mean').plot(figsize=(10,7), title = 'Season')

plt.show()
#dropping listed features

to_drop2 = ['Sunshine','Cloud9am','Cloud3pm', 'Date','Location']

for col in to_drop2:

    features.remove(col)

features.append('Season')

#check

features
weather[features].isnull().sum().sort_values(ascending=False)
copy_weather = weather[features].copy()

before = copy_weather.shape[0]

copy_weather = copy_weather.dropna()

after = copy_weather.shape[0]

print('The number of rows dropped will be {} which is {:.2f} % of the remaining dataset.'.format(before - after, ((before-after)/before)*100))
weather = weather.dropna(how='any', subset=features)

weather[features].info()
weather[features].info()
#create dummy variables for 'Season'

dummies_season = pd.get_dummies(weather['Season'], prefix = 'Season')

weather = pd.concat([weather, dummies_season], axis=1)

features.extend(dummies_season.columns.tolist())

features.remove('Season')

#check

weather[features].info()
#unique values in 'WindGustDir', 'WindDir9am', WindDir3pm'

windgust_unique = weather['WindGustDir'].unique().tolist()

winddir9am_unique = weather['WindDir9am'].unique().tolist()

winddir3pm_unique = weather['WindDir3pm'].unique().tolist()

print('{} unique values in "WindGustDir": {}'.format(len(windgust_unique), windgust_unique))

print('{} unique values in "WindDir9am": {}'.format(len(winddir9am_unique), winddir9am_unique))

print('{} unique values in "WindDir3pm": {}'.format(len(winddir3pm_unique), winddir3pm_unique))
#mapping wind direction labels to degrees 

dict_dir = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5}

for col in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:

    weather[col].replace(dict_dir, inplace=True)

#check

weather[['WindDir9am', 'WindDir3pm','WindGustDir']].head()
weather[['WindDir9am', 'WindDir3pm','WindGustDir']].head()
#converting wind direction in degrees to values on unit circle, adding to list of features, removing original feature

for col in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:

    weather[col + '_sin'] = np.sin(weather[col]*(2*np.pi/360))

    weather[col + '_cos'] = np.cos(weather[col]*(2*np.pi/360))

    features.append(col + '_sin')

    features.append(col + '_cos')

    features.remove(col)
#checking

fig, ax = plt.subplots(1,3,figsize = (20,5))

i = 0

for col in ['WindDir9am', 'WindDir3pm', 'WindGustDir']:

    ax[i].scatter(weather[col + '_sin'], weather[col + '_cos'])

    ax[i].set_title(col)

    i+=1

plt.show()
wind_sin_corr = weather[['WindDir9am_sin', 'WindDir3pm_sin','WindGustDir_sin']].corr()

sns.heatmap(wind_sin_corr, annot=True)
wind_sin_corr = weather[['WindDir9am_cos', 'WindDir3pm_cos','WindGustDir_cos']].corr()

sns.heatmap(wind_sin_corr, annot=True)
features.remove('WindGustDir_sin')

features.remove('WindGustDir_cos')

features
print('The selected features for the Logistic Regression model are: \n {}'.format(features))
from sklearn.model_selection import train_test_split



train_X, test_X, train_y, test_y = train_test_split(weather[features], weather['RainTomorrow'], test_size = 0.2, random_state = 0)
columns_to_scale = ['Rainfall', 'WindGustSpeed','Humidity3pm','Pressure3pm','Temp3pm']

train_X[columns_to_scale].describe()
from sklearn.preprocessing import minmax_scale



for col in columns_to_scale:

    train_X[col + "_scaled"] = minmax_scale(train_X[col])

    test_X[col + "_scaled"] = minmax_scale(test_X[col])

    weather[col + "_scaled"] = minmax_scale(weather[col])

    features.append(col + '_scaled')

    features.remove(col)

features
train_X.columns
test_X.columns
from sklearn.linear_model import LogisticRegression





lr = LogisticRegression(max_iter=1000)

lr.fit(train_X, train_y)

predictions = lr.predict(test_X)
from sklearn.metrics import confusion_matrix

import scikitplot as skplt

from sklearn.metrics import accuracy_score
plt.figure()

plt.hist(weather['RainTomorrow'])

plt.title('Histogram of RainTomorrow')

plt.show()
#evaluation function

def evaluation(test_y, predictions):

    

    #accuracy score

    accuracy = accuracy_score(test_y, predictions)

    print("The classification accuracy is {:.2f} %." .format(accuracy*100))

    

  

    y_test_mean = test_y.mean()

    #null accuracy

    null_accuracy = max(y_test_mean, 1-y_test_mean)

    print('The null accuracy is {:.2f} %.'.format(null_accuracy*100))

    

    #confusion matrix

    skplt.metrics.plot_confusion_matrix(test_y, predictions)

    

    conf_matrix = confusion_matrix(test_y, predictions)

    

    TN = conf_matrix[0,0] #true negatives

    FP = conf_matrix[0,1] #false positives

    FN = conf_matrix[1,0] #false negatives

    TP = conf_matrix[1,1] #true positives

    

    #precision

    precision = TP/(TP+FP)*100

    print('The precision is {:.2f} %.'.format(precision))

    #sensitivity/ recall

    recall = TP/(FN+TP)*100

    print('The sensitivity/recall is {:.2f} %.'.format(recall))

    #specificity

    specificity = TN/(FP+TN)*100

    print('The specificity is {:.2f} %.'.format(specificity))

    #F_score

    F_score = (2*precision*recall)/(precision + recall)

    print('The F score is {:.2f} %.'.format(F_score))

    

    return None
evaluation(test_y, predictions)
#splitting data into train, validation, and test set in proportions of 60%, 20%, and 20% respectively

X_set, X_test, y_set, y_test = train_test_split(weather[features], weather['RainTomorrow'], test_size = 0.2, random_state = 0)

X_train, X_cv, y_train, y_cv = train_test_split(X_set, y_set, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import normalize
#create polynomial variables of specified degree and scale them



def PolyAndScale(X, degree):

    

    #initialise polynomial transformer 

    poly = PolynomialFeatures(degree = degree, include_bias = True)

    #fit and return polynomial features

    X_poly = poly.fit_transform(X)

    #normalise X_poly

    X_poly = normalize(X_poly, axis=0)

    

    return X_poly

    
from sklearn.linear_model import LogisticRegression



lr_poly = LogisticRegression(max_iter=2000, fit_intercept=False)
def PredictAndError(X, y):

    

    #predict and  calculate error

    prediction = lr_poly.predict(X)

    #misclassification error

    error = 1 - accuracy_score(y_true = y, y_pred = prediction)

    

    return error

    
def LearningCurvesPoly(X_train, X_cv, y_train, y_cv):



    poly_degrees = [1,2,3,4,5]



    train_costs = []

    cv_costs = []



    for p in poly_degrees:

        X_tr = PolyAndScale(X_train, degree=p)

        lr_poly.fit(X_tr, y_train)

        train_error = PredictAndError(X_tr, y_train)

        train_costs.append(train_error)

        X_crossval = PolyAndScale(X_cv, degree=p)

        cv_error = PredictAndError(X_crossval, y_cv)

        cv_costs.append(cv_error)

        

    

    #plot learning curves

    plt.figure(figsize=(10,10))

    plt.plot(train_costs, label = 'Train')

    plt.plot(cv_costs, label = 'CV')

    plt.legend(loc="upper right")

    plt.xlabel('Polynomial degree')

    plt.xticks(ticks = [0,1,2,3,4], labels = [1,2,3,4,5])

    plt.ylabel('Misclassification Error')

    plt.title('Learning Curves')

    plt.show()

    

    return None

    
LearningCurvesPoly(X_train, X_cv, y_train, y_cv)