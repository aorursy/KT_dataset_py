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
import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings("ignore")



#plot import

import matplotlib.pyplot as plt

import seaborn as sns; 

#f, ax = plt.subplots(figsize=(16, 5))



# sklearn machine learning tools import

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet, BayesianRidge

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import mean_absolute_error









from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge, Lasso, RidgeCV

from sklearn.linear_model import BayesianRidge

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor

from sklearn.svm import SVR

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
train.head(2)
test.head(2)
import numpy as np

from sklearn.model_selection import KFold, cross_val_score

def create_number_of_dates_feature(dataset):

    """

    Create a new feature called number_of_days,

    from the dates in the entire dataset.

    dataset: train or test

    """

    data_frame_list = []

    countries = sorted(list(set(dataset["Country_Region"])))

    for k, country_name in enumerate(countries):

        country_dataset = dataset[dataset["Country_Region"] == countries[k]]

        m = len(country_dataset["Date"])

        country_dataset["number_of_days"] = [k for k in range(m)]

        data_frame_list.append(country_dataset)

    dataset = pd.concat(data_frame_list, axis=0)

    return dataset



def country_dataset(dataset, country_index):

    """

    dataset: train or test

    country_index: 1:len(countries)

    """

    countries = sorted(list(set(dataset["Country_Region"])))

    country = dataset[dataset["Country_Region"] == countries[country_index]]

    return country



def cv_rmse(model, X, y):

    kf = KFold(n_splits=12, random_state=42, shuffle=True)

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))

    return rmse.mean()





def get_best_polynomial_degree(degrees, model, X_train, y_train, X_valid,

                                y_valid):

    """

    degrees: list of polynomial degree (degrees = [1,2,3])

    """

    R2 = {}; MAE = {}; cross_val_error = {}

    for d in degrees:

        poly = PolynomialFeatures(d)

        X_train_poly = poly.fit_transform(X_train)

        X_valid_poly = poly.fit_transform(X_valid)

        model.fit(X_train_poly, y_train)

        cv_rms = cv_rmse(model, X_train_poly, y_train)

        y_preds = model.predict(X_valid_poly)

        mae = mean_absolute_error(y_valid, y_preds)

        r2 = r2_score(y_valid, y_preds)

        R2[d] = r2; MAE[d] = mae

        cross_val_error[d] = cv_rms

    R2 = dict(sorted(R2.items(), key=lambda x: x[1], reverse=True))

    MAE = sorted(MAE.items(), key=lambda x: x[1])

    cross_val_error = sorted(cross_val_error.items(), key=lambda x: x[1])

    return R2, MAE, cross_val_error











MODELS = {

"KernelRideg": KernelRidge(), "LinerRegression": LinearRegression(),

"Ridge": Ridge(), "lasso": Lasso(), "RidgeCV": RidgeCV(),

"BayessianRidge": BayesianRidge(), "ElasticNet": ElasticNet(),

"ElasticNetCV": ElasticNetCV(), "RandomForestRegressor":RandomForestRegressor(),

"AdaBoostRegressor": AdaBoostRegressor(),

"GradientBoostingRegressor": GradientBoostingRegressor(),

"BaggingRegressor": BaggingRegressor(), "SVR": SVR(),

"LGBMRegressor": LGBMRegressor(), "XGBRegressor": XGBRegressor()

}

def get_best_model(degree, X_train, y_train, X_valid,y_valid):

    fitted_models = {}

    R2 = {}; MAE = {}; cross_val_error = {}

    poly = PolynomialFeatures(degree)

    for name, model in MODELS.items():

        X_train_poly = poly.fit_transform(X_train)

        X_valid_poly = poly.fit_transform(X_valid)

        model.fit(X_train_poly, y_train)

        fitted_models[name] = model

        cv_rms = cv_rmse(model, X_train_poly, y_train)

        y_preds = model.predict(X_valid_poly)

        mae = mean_absolute_error(y_valid, y_preds)

        r2 = r2_score(y_valid, y_preds)

        R2[name] = r2; MAE[name] = mae

        cross_val_error[name] = cv_rms

    R2 = dict(sorted(R2.items(), key=lambda x: x[1], reverse=True))

    MAE = sorted(MAE.items(), key=lambda x: x[1])

    optimum_model_name  = MAE[0][0]

    optimum_model = fitted_models[optimum_model_name]

    return R2, MAE, cross_val_error, optimum_model







def get_predictions(degree, X_train, y_train, X_valid,

                                y_valid, X_test):

    poly = PolynomialFeatures(degree)

    X_test_poly = poly.fit_transform(X_test)

    _, _, _, optimum_model = get_best_model(degree, X_train,

                                    y_train, X_valid, y_valid)

    #print(optimum_model)

    test_preds = optimum_model.predict(X_test_poly)

    return test_preds









def fatality_prediction_per_country(country_index, train, test):

    train = create_number_of_dates_feature(train)

    test = create_number_of_dates_feature(test)

    country_train = country_dataset(train, country_index)

    country_test = country_dataset(test, country_index)

    scalar_test = StandardScaler()

    # fatalities split train test

    X_test = country_test[["number_of_days"]]

    X_test = scalar_test.fit_transform(X_test)

    X_train_fatalities = country_train[["number_of_days"]]

    y_train_fatalities = country_train["Fatalities"].values.reshape(-1,1)





    X_train_fat, X_valid_fat, y_train_fat, y_valid_fat = train_test_split(X_train_fatalities, y_train_fatalities,

                                                                test_size=0.20, random_state=42)

    # Normalization for fatalities

    scalar_train = StandardScaler()

    X_train_fat = scalar_train.fit_transform(X_train_fat)

    y_train_fat = scalar_train.fit_transform(y_train_fat)



    scalar_valid = StandardScaler()

    X_valid_fat = scalar_valid.fit_transform(X_valid_fat)

    y_valid_fat = scalar_valid.fit_transform(y_valid_fat)

    degrees = [4,5,6,7]

    model = LinearRegression()

    R21, MAE1, cross_val_err1 = get_best_polynomial_degree(degrees, model, X_train_fat, y_train_fat, X_valid_fat, y_valid_fat)

    optimum_degree, mae_score = MAE1[0][0], MAE1[0][1]

    predictions = get_predictions(optimum_degree, X_train_fat, y_train_fat, X_valid_fat, y_valid_fat, X_test)

    y_pr = list(scalar_test.inverse_transform(predictions))

    y_prediction = list(map(lambda x: int(x), y_pr))

    return y_prediction







def confirm_cases_prediction_per_country(country_index, train, test):

    train = create_number_of_dates_feature(train)

    test = create_number_of_dates_feature(test)

    country_train = country_dataset(train, country_index)

    country_test = country_dataset(test, country_index)

    scalar_test = StandardScaler()

    # fatalities split train test

    X_test = country_test[["number_of_days"]]

    X_test = scalar_test.fit_transform(X_test)

    X_train_fatalities = country_train[["number_of_days"]]

    y_train_fatalities = country_train["ConfirmedCases"].values.reshape(-1,1)





    X_train_fat, X_valid_fat, y_train_fat, y_valid_fat = train_test_split(X_train_fatalities, y_train_fatalities,

                                                                test_size=0.20, random_state=42)

    # Normalization for fatalities

    scalar_train = StandardScaler()

    X_train_fat = scalar_train.fit_transform(X_train_fat)

    y_train_fat = scalar_train.fit_transform(y_train_fat)



    scalar_valid = StandardScaler()

    X_valid_fat = scalar_valid.fit_transform(X_valid_fat)

    y_valid_fat = scalar_valid.fit_transform(y_valid_fat)

    degrees = [4,5,6,7]

    model = LinearRegression()

    R21, MAE1, cross_val_err1 = get_best_polynomial_degree(degrees, model, X_train_fat, y_train_fat, X_valid_fat, y_valid_fat)

    optimum_degree, mae_score = MAE1[0][0], MAE1[0][1]

    predictions = get_predictions(optimum_degree, X_train_fat, y_train_fat, X_valid_fat, y_valid_fat, X_test)

    y_pr = list(scalar_test.inverse_transform(predictions))

    y_prediction = list(map(lambda x: int(x), y_pr))

    return y_prediction





def prediction_dataframe(test, train):

    container = []

    countries = sorted(list(set(test["Country_Region"])))

    m = len(countries)

    indexes = [k for k in range(m)]

    for country_index in indexes:

        country = test[test["Country_Region"] == countries[country_index]]

        ForecastId = list(country["ForecastId"])

        fat_preds = fatality_prediction_per_country(country_index, train, test)

        conf_preds = confirm_cases_prediction_per_country(country_index, train, test)

        d = {"ForecastId":ForecastId, "ConfirmedCases":conf_preds,"Fatalities":fat_preds}

        df = pd.DataFrame(d)

        container.append(df)

    df_sub = pd.concat(container, axis=0)

    df_sub.to_csv("submission.csv", index=False)
train = create_number_of_dates_feature(train)

test = create_number_of_dates_feature(test)
prediction_dataframe(test, train)

