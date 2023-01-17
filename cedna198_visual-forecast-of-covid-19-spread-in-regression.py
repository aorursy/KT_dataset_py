import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.feature_selection import RFE

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dt_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 

dt_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
display(dt_train.head())

display(dt_train.describe())

display(dt_train.info())
viz = dt_train[["ConfirmedCases", "Fatalities"]]

viz.hist()

plt.show
plt.scatter(dt_train.ConfirmedCases, dt_train.Fatalities)

plt.xlabel("ConfirmedCases")

plt.ylabel("Fatalities")

plt.show
plt.scatter(dt_train.Date, dt_train.Fatalities)

plt.xlabel("Date")

plt.ylabel("Fatalities")

plt.show
plt.figure(figsize=(18,50))

plt.scatter(dt_train.ConfirmedCases, dt_train.Country_Region)

plt.xlabel("ConfirmedCases")

plt.ylabel("Country_Region")

plt.show
usa = dt_train[dt_train["Country_Region"]=="US"]
dt_train
usa.head()
usa.tail()
plt.figure(figsize=(10,8))

plt.plot(usa["ConfirmedCases"])

plt.xlabel("Time")

plt.ylabel("The Number of Confirmed Cases in USA")
plt.figure(figsize=(10,8))

plt.plot(usa["Fatalities"])

plt.xlabel("Time")

plt.ylabel("The Number of Fatalities in USA")
tab_info = pd.DataFrame(dt_train.dtypes).T.rename(index={0:'column Type'}) 

tab_info = tab_info.append(pd.DataFrame(dt_train.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info = tab_info.append(pd.DataFrame(dt_train.isnull().sum()/dt_train.shape[0]*100).T.rename(index={0: 'null values (%)'}))

tab_info
usa_states = dt_train[dt_train["Country_Region"]=="US"]["Province_State"].unique()
def province(state, country):

    if state == "nan":

        return country

    return state
dt_train = dt_train.fillna ("nan")
dt_train["Province_State"] = dt_train.apply(lambda x: province(x["Province_State"], x["Country_Region"]), axis = 1)
dt_train
usa_states
import seaborn as sns

corr = dt_train.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr, 

            annot=True, fmt=".3f",

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
sns.pairplot(dt_train, vars=["ConfirmedCases", "Fatalities", "Date", "Province_State", "Country_Region"])
sns.pairplot(dt_train.fillna(0), vars=["ConfirmedCases", "Fatalities", "Date", "Province_State", "Country_Region"])
for name, group in dt_train.groupby(["Province_State", "Country_Region"]):

    plt.title(name)

    plt.scatter(range(len(group)), group["ConfirmedCases"])

    plt.show()

    break
# Using sklearn package to model the data

from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(dt_train[['ConfirmedCases']])

train_y = np.asanyarray(dt_train[['Fatalities']])

regr.fit (train_x, train_y)



# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(dt_train.ConfirmedCases, dt_train.Fatalities,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("ConfirmedCases")

plt.ylabel("Fatalities")
display(dt_test.head())

display(dt_test.describe())

display(dt_test.info())
from sklearn.metrics import r2_score



test_x = np.asanyarray(dt_train[['ConfirmedCases']])

test_y = np.asanyarray(dt_train[['Fatalities']])

test_y_ = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
dt_train.to_csv('submission.csv', index = False)