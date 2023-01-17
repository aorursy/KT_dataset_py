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



X = spend['LinuxGamingHoursPerWeek'].values[:, np.newaxis]

y = spend['LinuxGamingSpendingPerMonth'].values[:, np.newaxis]
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
pipe.steps
pipe.named_steps
pipe.set_params(polynomialfeatures__degree=3)
pipe.fit(X, y)

y_hat = pipe.predict(X)



# Plot the result.

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



sortorder = np.argsort(spend['LinuxGamingHoursPerWeek'].values)

s = spend['LinuxGamingHoursPerWeek'].values[sortorder]

y_hat = y_hat[sortorder]

plt.plot(s, y_hat)

plt.scatter(spend['LinuxGamingHoursPerWeek'], 

            spend['LinuxGamingSpendingPerMonth'], 

            color='black')
from sklearn.grid_search import GridSearchCV



pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

param_grid = dict(polynomialfeatures__degree=list(range(1, 5)))

grid_search = GridSearchCV(pipe, param_grid=param_grid)
grid_search.fit(X, y)
grid_search.estimator