%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np
rng = np.random.RandomState(1)

x = 10 * rng.rand(50) # Added noise

y = 2 * x - 5 + rng.randn(50)

plt.scatter(x, y);
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)



model.fit(x[:, np.newaxis], y)



xfit = np.linspace(0, 10, 1000)

yfit = model.predict(xfit[:, np.newaxis])



plt.scatter(x, y)

plt.plot(xfit, yfit);


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(3),

                           LinearRegression())



rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = np.sin(x) + 0.1 * rng.randn(50)



poly_model.fit(x[:, np.newaxis], y)

yfit = poly_model.predict(xfit[:, np.newaxis])



plt.scatter(x, y)

plt.plot(xfit, yfit);


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(7),

                           LinearRegression())



rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = np.sin(x) + 0.1 * rng.randn(50)



poly_model.fit(x[:, np.newaxis], y)

yfit = poly_model.predict(xfit[:, np.newaxis])



plt.scatter(x, y)

plt.plot(xfit, yfit);


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(20),

                           LinearRegression())



rng = np.random.RandomState(1)

x = 10 * rng.rand(50)

y = np.sin(x) + 0.1 * rng.randn(50)



poly_model.fit(x[:, np.newaxis], y)

yfit = poly_model.predict(xfit[:, np.newaxis])



plt.scatter(x, y)

plt.plot(xfit, yfit);


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(PolynomialFeatures(20),

                           LinearRegression())



rng = np.random.RandomState(1)

x = 10 * rng.rand(1000)

y = np.sin(x) + 0.1 * rng.randn(1000)



poly_model.fit(x[:, np.newaxis], y)

yfit = poly_model.predict(xfit[:, np.newaxis])



plt.scatter(x, y)

plt.plot(xfit, yfit);


from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error



error = []

for i in range(4, 21) :

    

    poly_model = make_pipeline(PolynomialFeatures(i),

                               LinearRegression())

    rng = np.random.RandomState(1)

    x = 10 * rng.rand(50)

    y = np.sin(x) + 0.1 * rng.randn(50)

    poly_model.fit(x[:, np.newaxis], y)

    yfit = poly_model.predict(xfit[:, np.newaxis])

    ytrue = [np.sin(c) for c in xfit]

    error.append(mean_squared_error(ytrue, yfit))



x = [i for i in range(4, 21)]

plt.scatter(x, error)

plt.plot(x, error);


url = 'https://raw.githubusercontent.com/ankitesh97/IMDB-MovieRating-Prediction/master/movie_metadata_filtered_aftercsv.csv'


