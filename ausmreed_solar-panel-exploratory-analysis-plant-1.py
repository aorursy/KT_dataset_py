# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the weather and power readings from the first solar plant

plant1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

weather1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')



plant2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')

weather2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
# Convert the DATE_TIME column to a datetime data type.

plant1['DATE_TIME'] = pd.to_datetime(plant1['DATE_TIME'], dayfirst = True)

weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], yearfirst = True)



plant2['DATE_TIME'] = pd.to_datetime(plant2['DATE_TIME'], dayfirst = True)

weather2['DATE_TIME'] = pd.to_datetime(weather2['DATE_TIME'], yearfirst = True)

# Preliminary plot to check trends. 

import seaborn as sb 

spefplant = plant1[plant1['SOURCE_KEY'] == (plant1.SOURCE_KEY).unique()[0]].sort_values(by = 'DATE_TIME').tail(94).reset_index()



for index, row in spefplant.iterrows(): 

    spefplant.at[index, 'Minutes'] = index*15



#Plotting

sb.lineplot(x = 'Minutes', y = 'TOTAL_YIELD', data = spefplant)



# Plot labeling

plt.title('Daily Solar Panel Yield (Plant 1)')

plt.xlabel('Minutes past midnight')

plt.ylabel('Total Yield')

# Suppress a warning to clean up the output. 

import warnings

with warnings.catch_warnings():

    warnings.simplefilter('ignore')

#------------------------Actual Code Below ------------------------------------

    allplant = plant1[plant1['SOURCE_KEY'] == (plant1.SOURCE_KEY).unique()[0]]

    allplant['Hour'] = allplant['DATE_TIME'].apply(lambda x: x.hour)

    allplant['Minute'] = allplant['DATE_TIME'].apply(lambda x: x.minute)



    for index, row in allplant.iterrows(): 

        minutes = row['Minute']

        hours = row['Hour']

        min_past_midnight = hours*60 + minutes 

        allplant.at[index, 'MinutesPastMidnight'] = min_past_midnight



    



    plt.scatter(allplant['MinutesPastMidnight'], allplant['DAILY_YIELD'], s = 3)

    plt.xlabel('Minutes Past Midnight')

    plt.ylabel('Daily Yield')

    plt.title("Daily yield from single sensor")
# Fit this to a sigmoid curve to get the daily prediction yield.

import pylab 

from scipy.optimize import curve_fit



# Define the sigmoid function with the three parameters and fit. 

def sigmoid(x, a, x0, k):

     y = a*(1 / (1 + np.exp(-k*(x-x0))))

     return y



x = allplant['MinutesPastMidnight'].values

y = allplant['DAILY_YIELD'].values



popt, pcov = curve_fit(sigmoid, x, y)

print(popt)



# Plot the optimized curve.

x = np.linspace(0, 1440, 1000)

y = sigmoid(x, *popt)

plt.plot(x,y, color = 'r')

plt.scatter(allplant['MinutesPastMidnight'], allplant['DAILY_YIELD'], s = 3)

plt.xlabel('MinutesPastMidnight')

plt.ylabel('DAILY_YIELD')

plt.title("Daily yield from single sensor")
for item in (plant1.SOURCE_KEY).unique():

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        allplant = plant1[plant1['SOURCE_KEY'] == item]

        allplant['Hour'] = allplant['DATE_TIME'].apply(lambda x: x.hour)

        allplant['Minute'] = allplant['DATE_TIME'].apply(lambda x: x.minute)



        for index, row in allplant.iterrows(): 

            minutes = row['Minute']

            hours = row['Hour']

            min_past_midnight = hours*60 + minutes 

            allplant.at[index, 'MinutesPastMidnight'] = min_past_midnight



        plt.scatter(allplant['MinutesPastMidnight'], allplant['DAILY_YIELD'], s = 3, alpha = 0.25)

        plt.xlabel('Minutes Past Midnight')

        plt.ylabel('Daily Yield')

plt.plot(x, y, color = 'r')
# Create a "Irradiation Yield" column. 

weather1['TOTAL_IRRADIATION'] = 0

run_sum = 0 

i = 1

while i <= 3182: 

    run_sum = run_sum + weather1['IRRADIATION'].iloc[i-1]

    i += 1 

    weather1.at[i, 'TOTAL_IRRADIATION'] = run_sum

sb.lineplot(x = 'DATE_TIME', y = 'TOTAL_IRRADIATION', data = weather1)

plt.title("Irradiation yield over time")
merged = spefplant.merge(right = weather1, how = 'left', on = 'DATE_TIME')
mergedred = merged[['PLANT_ID_x', 'TOTAL_YIELD', 'TOTAL_IRRADIATION']].dropna()
X = mergedred['TOTAL_IRRADIATION'].values.reshape(-1,1)

y = mergedred['TOTAL_YIELD'].values.reshape(-1,1)
# ML Linear Regression, single plant, past hour



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

# X = sc_X.fit_transform(X)

# y = sc_y.fit_transform(y)







from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()



import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/4)

lin_reg.fit(X_train, y_train)

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')

plt.xlabel('IRRADIATION YIELD')

plt.ylabel('TOTAL POWER OUTPUT')





from sklearn.metrics import r2_score

y_true = y_test

y_pred = lin_reg.predict(X_test)

r2_score(y_true, y_pred)

lin_reg.coef_
# Iterate through all the sensors and compare coefficients. 

coefficients = {}

for sensor in plant1.SOURCE_KEY.unique(): 

    spefplant = plant1[plant1['SOURCE_KEY'] == sensor].tail(96)

    merged = spefplant.merge(right = weather1, how = 'left', on = 'DATE_TIME')

    merged = merged[['PLANT_ID_x', 'TOTAL_YIELD', 'TOTAL_IRRADIATION']].dropna()

    X = merged['TOTAL_IRRADIATION'].values.reshape(-1,1)

    y = merged['TOTAL_YIELD'].values.reshape(-1,1) 



    lin_reg = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 3/4)

    lin_reg.fit(X_train, y_train)



    y_true = y_test

    y_pred = lin_reg.predict(X_test)

    r2_score(y_true, y_pred)

    coefficients[sensor] = lin_reg.coef_





# Distributions of slopes

sb.distplot(list(coefficients.values()))

stdev = np.std(list(coefficients.values()))

for item in coefficients: 

    if coefficients[item] < -(2*stdev) + np.mean(list(coefficients.values())):

        print("The sensor {} might be faulty or dirty! Outside 2 sigmas of performance".format(item))
spefplant = plant1[plant1['SOURCE_KEY'] == (plant1.SOURCE_KEY).unique()[0]].sort_values(by = 'DATE_TIME')

# First, for DC power output

merged = spefplant.merge(right = weather1, on = 'DATE_TIME')

merged_reduced_dc = merged[['DC_POWER', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]



#Train Test Split

X = merged_reduced_dc[['AMBIENT_TEMPERATURE', 'IRRADIATION']].values

y = merged_reduced_dc.iloc[:, 0].values.reshape(-1,1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

merged_reduced_dc



# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

polyreg = PolynomialFeatures(degree = 3)

x_poly = polyreg.fit_transform(X_train)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly, y_train)



# Test on test set and gauge accuracy

y_pred = lin_reg_2.predict(polyreg.transform(X_test))

from sklearn.metrics import r2_score

print("R^2 of the regression model is {}".format(r2_score(y_test,y_pred)))

# Try this model on other sensors and see how well the prediction model performs.

r2_array = []

for item in plant1['SOURCE_KEY'].unique(): 

    spefplant = plant1[plant1['SOURCE_KEY'] == item]

    merged = spefplant.merge(right = weather1, on = 'DATE_TIME')

    merged_reduced_dc = merged[['DC_POWER', 'AMBIENT_TEMPERATURE', 'IRRADIATION']]

    X = merged_reduced_dc[['AMBIENT_TEMPERATURE', 'IRRADIATION']].values

    y = merged_reduced_dc.iloc[:, 0].values.reshape(-1,1)

    y_pred = lin_reg_2.predict(polyreg.transform(sc.transform(X)))

    r2_array.append(r2_score(y, y_pred))

# Print out the distributions of R-Squared

sb.distplot(r2_array)

plt.xlabel('R-Squared Value')