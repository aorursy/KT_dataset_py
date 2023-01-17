import numpy as np

import pandas as pd

import random

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

np.random.seed(10)



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
x = np.array([i*np.pi/180 for i in range(60,300,4)])

y = np.sin(x) + np.random.normal(0,0.15,len(x))

curve = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])



fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(curve['x'],curve['y'], '.')

curve.head()
def fit_poly( degree ):

    p = np.polyfit( curve.x, curve.y, deg = degree )

    curve['fit'] = np.polyval( p, curve.x )    

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.regplot(curve.x, curve.y, fit_reg = False, ax=ax)   

    ax.plot(curve.x, curve.fit, label='fit')
fit_poly(1)

fit_poly(3)

fit_poly(20)
def get_rmse(y, y_fit):

    return np.sqrt(mean_squared_error( y, y_fit ))
train_X, test_X, train_y, test_y = train_test_split(curve.x, curve.y, test_size = 0.40, random_state = 100)
rmse_df = pd.DataFrame(columns = ["degree", "rmse_train", "rmse_test"])

for i in range( 1, 15 ):

    p = np.polyfit( train_X, train_y, deg = i )

    rmse_df.loc[i-1] = [ i,

                      get_rmse( train_y, np.polyval( p, train_X ) ),

                      get_rmse( test_y, np.polyval( p, test_X ) ) ]



rmse_df
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(rmse_df.degree, rmse_df.rmse_train, label='train', color = 'r')

ax.plot(rmse_df.degree, rmse_df.rmse_test, label='test', color = 'g')

ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.grid()
curve.head()
estimators = []

estimators.append(('LinearRegression', LinearRegression()))



poly_regression = Pipeline((

    ("poly_features", PolynomialFeatures(degree=14, include_bias=False)),

    ("sgd_reg", LinearRegression()),

))

estimators.append(('PloynomialRegression', poly_regression))
#divide data into feature and target

X = curve.iloc[:, 0:1]

y = curve.iloc[:, 1:2]
def plot_learning_curves(model, title, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):

        model.fit(X_train[:m], y_train[:m])

        y_train_predict = model.predict(X_train[:m])

        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))

        val_errors.append(mean_squared_error(y_val_predict, y_val))

    

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")

    ax.plot(np.sqrt(val_errors), "g-", linewidth=3, label="test") 

    ax.set_xlabel("Training examples")

    ax.set_ylabel("RMSE")

    ax.grid()

    ax.legend()

    ax.set_title(title)

    ax.set_ylim(0, 1)

    ax.set_xlim(0, 50)
for name, model in estimators:

    title = "Learning Curves: " + name   

    plot_learning_curves(model, title, X, y)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):            

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='neg_mean_squared_error', random_state=0)

    train_scores_mean = np.sqrt(np.mean(-train_scores, axis=1))

    train_scores_std = np.std(-train_scores, axis=1)

    test_scores_mean = np.sqrt(np.mean(-test_scores, axis=1))

    test_scores_std = np.std(-test_scores, axis=1)

    

    fig, ax= plt.subplots(figsize=(8, 5))   

    ax.set_title(title)

    if ylim is not None:

        ax.set_ylim(*ylim)

    ax.set_xlabel("Training examples")

    ax.set_ylabel("RMSE")

    ax.grid()

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training RMSE")

    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation RMSE")

    ax.legend(loc="best")

    return plt
for name, model in estimators:

    title = "Learning Curves: " + name

    # Cross validation with 100 iterations to get smoother mean test and train

    # score curves, each time with 20% data randomly selected as a validation set.

    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)    

    plot_learning_curve(model, title, X, y, cv=cv, n_jobs=4)  # ylim=(0.7, 1.01), 