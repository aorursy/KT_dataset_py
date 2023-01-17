import pandas as pd

survey = pd.read_csv(

    "../input/BoilingSteam_com_LinuxGamersSurvey_Q1_2016_Public_Sharing_Only.csv"

).loc[1:]



import numpy as np



spend = survey.loc[:, ['LinuxGamingHoursPerWeek', 'LinuxGamingSpendingPerMonth']].dropna()

spend = spend.assign(

    LinuxGamingHoursPerWeek=spend.LinuxGamingHoursPerWeek.map(lambda v: int(v) if str.isdigit(v) else np.nan),

    LinuxGamingSpendingPerMonth=spend.LinuxGamingSpendingPerMonth.map(lambda v: float(v) if str.isdecimal(v) else np.nan)

).dropna()
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



sns.jointplot(spend['LinuxGamingHoursPerWeek'], spend['LinuxGamingSpendingPerMonth'])
hours = spend['LinuxGamingHoursPerWeek'].values[:, np.newaxis]

y = spend['LinuxGamingSpendingPerMonth'].values[:, np.newaxis]
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



def predict(degree):

    poly = PolynomialFeatures(degree=degree)

    hours_poly = poly.fit_transform(hours)

    clf = LinearRegression()

    clf.fit(hours_poly, y)

    return clf
y_hat = predict(1).predict(PolynomialFeatures(degree=1).fit_transform(hours))

sortorder = np.argsort(spend['LinuxGamingHoursPerWeek'].values)

s = spend['LinuxGamingHoursPerWeek'].values[sortorder]

y_hat = y_hat[sortorder]



import matplotlib.pyplot as plt

plt.plot(s, y_hat)

plt.scatter(spend['LinuxGamingHoursPerWeek'], spend['LinuxGamingSpendingPerMonth'], color='black')
y_hat = predict(10).predict(PolynomialFeatures(degree=10).fit_transform(hours))

sortorder = np.argsort(spend['LinuxGamingHoursPerWeek'].values)

s = spend['LinuxGamingHoursPerWeek'].values[sortorder]

y_hat = y_hat[sortorder]



import matplotlib.pyplot as plt

plt.plot(s, y_hat)

plt.scatter(spend['LinuxGamingHoursPerWeek'], spend['LinuxGamingSpendingPerMonth'], color='black')
n = len(y)

ratio = 0.8

pivot = int(np.round(0.8 * n))

train = spend.iloc[:pivot]

test = spend.iloc[pivot:]



train_X = train['LinuxGamingHoursPerWeek'].values[:, np.newaxis]

train_y = train['LinuxGamingSpendingPerMonth'].values[:, np.newaxis]

test_X = test['LinuxGamingHoursPerWeek'].values[:, np.newaxis]

test_y = test['LinuxGamingSpendingPerMonth'].values[:, np.newaxis]
test_y_hat= (LinearRegression()

     .fit(PolynomialFeatures(degree=10).fit_transform(train_X), train_y)

     .predict(PolynomialFeatures(degree=10).fit_transform(test_X))

)
from sklearn.metrics import mean_squared_error



mean_squared_error(test_y, test_y_hat)
sortorder = np.argsort(test_X.flatten())

test_X_p = test_X.flatten()[sortorder]

test_y_hat_p = test_y_hat[sortorder]

plt.scatter(test_X, test_y, color='black')

plt.plot(test_X_p, test_y_hat_p)
test_y_hat= (LinearRegression()

     .fit(PolynomialFeatures(degree=2).fit_transform(train_X), train_y)

     .predict(PolynomialFeatures(degree=2).fit_transform(test_X))

)



mean_squared_error(test_y, test_y_hat)
sortorder = np.argsort(test_X.flatten())

test_X_p = test_X.flatten()[sortorder]

test_y_hat_p = test_y_hat[sortorder]

plt.scatter(test_X, test_y, color='black')

plt.plot(test_X_p, test_y_hat_p)
mses = [mean_squared_error(

    test_y, 

    (LinearRegression()

        .fit(PolynomialFeatures(degree=n).fit_transform(train_X), train_y)

        .predict(PolynomialFeatures(degree=n).fit_transform(test_X)))

 ) for n in range(1, 11)]
pd.Series(mses, index=range(1, 11)).plot.line(title='CV MSE per N(Coefficients)')