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
# importing necessary libraries

import warnings

warnings.filterwarnings('ignore')



import random

random.seed(42)



import pandas as pd

import numpy as np



from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import TimeSeriesSplit, cross_val_score



import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

%config InlineBackend.figure_format = 'retina'
df = pd.read_csv('../input/ads_hour.csv',

                 index_col=['Date'], parse_dates=['Date'])
with plt.style.context('bmh'):    

    plt.figure(figsize=(15, 8))

    plt.title('Ads watched (hour ticks)')

    plt.plot(df.ads);
def prepareData(data, lag_start=5, lag_end=14, test_size=0.3):

    """

    series: pd.DataFrame

        dataframe with timeseries



    lag_start: int

        initial step back in time to slice target variable 

        example - lag_start = 1 means that the model 

                  will see yesterday's values to predict today



    lag_end: int

        final step back in time to slice target variable

        example - lag_end = 4 means that the model 

                  will see up to 4 days back in time to predict today



    test_size: float

        size of the test dataset after train/test split as percentage of dataset



    """

    data = pd.DataFrame(data.copy())

    data.columns = ["y"]

    

    # calculate test index start position to split data on train test

    test_index = int(len(data) * (1 - test_size))

    

    # adding lags of original time series data as features

    for i in range(lag_start, lag_end):

        data["lag_{}".format(i)] = data.y.shift(i)

        

    # transforming df index to datetime and creating new variables

    data.index = pd.to_datetime(data.index)

    data["hour"] = data.index.hour

    data["weekday"] = data.index.weekday

        

    # since we will be using only linear models we need to get dummies from weekdays 

    # to avoid imposing weird algebraic rules on day numbers

    data = pd.concat([

        data.drop("weekday", axis=1), 

        pd.get_dummies(data['weekday'], prefix='weekday')

    ], axis=1)

        

    # encode hour with sin/cos transformation

    # credits - https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

    data['sin_hour'] = np.sin(2*np.pi*data['hour']/24)

    data['cos_hour'] = np.cos(2*np.pi*data['hour']/24)

    data.drop(["hour"], axis=1, inplace=True)

        



    data = data.dropna()

    data = data.reset_index(drop=True)

    

    

    # splitting whole dataset on train and test

    X_train = data.loc[:test_index].drop(["y"], axis=1)

    y_train = data.loc[:test_index]["y"]

    X_test = data.loc[test_index:].drop(["y"], axis=1)

    y_test = data.loc[test_index:]["y"]

    

    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = prepareData(df, lag_start=12, lag_end=48, test_size=0.3)
df.head()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)
tscv = TimeSeriesSplit(n_splits=5)

cv = cross_val_score(lr, X_train_scaled, y_train, 

                                    cv=tscv, 

                                    scoring="neg_mean_absolute_error")
print(cv.mean())
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def plotModelResults(model, df_train, df_test, y_train, y_test, plot_intervals=False, scale=1.96, cv=tscv):

    """

    Plots modelled vs fact values

    

    model: fitted model 

    

    df_train, df_test: splitted featuresets

    

    y_train, y_test: targets

    

    plot_intervals: bool, if True, plot prediction intervals

    

    scale: float, sets the width of the intervals

    

    cv: cross validation method, needed for intervals

    

    """

    # making predictions for test

    prediction = model.predict(df_test)

    

    plt.figure(figsize=(20, 7))

    plt.plot(prediction, "g", label="prediction", linewidth=2.0)

    plt.plot(y_test.values, label="actual", linewidth=2.0)

    

    if plot_intervals:

        # calculate cv scores

        cv = cross_val_score(

            model, 

            df_train, 

            y_train, 

            cv=cv, 

            scoring="neg_mean_squared_error"

        )

        print(cv.mean())

        

        # calculate cv error deviation

        deviation = np.sqrt(cv.std())

        

        # calculate lower and upper intervals

        lower = prediction - (scale * deviation)

        upper = prediction + (scale * deviation)

        

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)

        plt.plot(upper, "r--", alpha=0.5)

        

    # calculate overall quality on test set

    mae  = mean_absolute_error(prediction, y_test)

    mape = mean_absolute_percentage_error(prediction, y_test)

    plt.title("MAE {}, MAPE {}%".format(round(mae), round(mape, 2)))

    plt.legend(loc="best")

    plt.grid(True)
def getCoefficients(model):

    """Returns sorted coefficient values of the model"""

    coefs = pd.DataFrame(model.coef_, X_train.columns)

    coefs.columns = ["coef"]

    coefs["abs"] = coefs.coef.apply(np.abs)

    return coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)    

    



def plotCoefficients(model):

    """Plots sorted coefficient values of the model"""

    coefs = getCoefficients(model)

    

    plt.figure(figsize=(20, 7))

    coefs.coef.plot(kind='bar')

    plt.grid(True, axis='y')

    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')

    plt.show()
plotModelResults(model=lr,df_train=X_train_scaled,df_test=X_test_scaled,y_train=y_train, y_test=y_test, plot_intervals=True, cv=tscv)

#plotCoefficients(lr)
plt.figure(figsize=(10, 8))

sns.heatmap(X_train.corr());
lasso = LassoCV(cv=tscv)

lasso.fit(X_train_scaled, y_train)
plotModelResults(lasso,X_train_scaled,X_test_scaled,y_train, y_test, plot_intervals=True, cv=tscv)
getCoefficients(lasso)
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline



def plotPCA(pca):

    """

    Plots accumulated percentage of explained variance by component

    

    pca: fitted PCA object

    """

    components = range(1, pca.n_components_ + 1)

    variance = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

    plt.figure(figsize=(20, 10))

    plt.bar(components, variance)

    

    # additionally mark the level of 95% of explained variance 

    plt.hlines(y = 95, xmin=0, xmax=len(components), linestyles='dashed', colors='red')

    

    plt.xlabel('PCA components')

    plt.ylabel('variance')

    plt.xticks(components)

    plt.show()
pca = PCA()

pca.fit_transform(X_train_scaled)

plotPCA(pca)
lr2 = LinearRegression()

pca2=PCA(9)

pca_features_train = pca2.fit_transform(X_train_scaled)

pca_features_test = pca2.transform(X_test_scaled)

lr2.fit(pca_features_train, y_train)
plotModelResults(lr2,pca_features_train,pca_features_test,y_train, y_test, plot_intervals=True)
cv = cross_val_score(lr2, pca_features_train, y_train, 

                                    cv=tscv, 

                                    scoring="neg_mean_absolute_error")
print(cv)
cv.mean()
#prediction = lr.predict(X_test_scaled)

#mean_absolute_error(prediction, y_test)