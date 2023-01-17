import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR



import statsmodels.api as sm



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max.columns', None)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_vw = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv")

print(data_vw.shape)

data_vw.head()
data_vw.isnull().sum()
data_vw.describe()
sns.countplot(data_vw["transmission"])
print(data_vw["model"].value_counts() / len(data_vw))

sns.countplot(y = data_vw["model"])
sns.countplot(data_vw["fuelType"])
sns.countplot(y = data_vw["year"])
plt.figure(figsize=(15,5),facecolor='w') 

sns.barplot(x = data_vw["year"], y = data_vw["price"])
sns.barplot(x = data_vw["transmission"], y = data_vw["price"])
plt.figure(figsize=(15,10),facecolor='w') 

sns.scatterplot(data_vw["mileage"], data_vw["price"], hue = data_vw["year"])
plt.figure(figsize=(15,5),facecolor='w') 

sns.scatterplot(data_vw["mileage"], data_vw["price"], hue = data_vw["fuelType"])
sns.pairplot(data_vw)
data_vw["age_of_car"] = 2020 - data_vw["year"]

data_vw = data_vw.drop(columns = ["year"])

data_vw.sample(10)
data_vw_expanded = pd.get_dummies(data_vw)

data_vw_expanded.head()
std = StandardScaler()

data_vw_expanded_std = std.fit_transform(data_vw_expanded)

data_vw_expanded_std = pd.DataFrame(data_vw_expanded_std, columns = data_vw_expanded.columns)

print(data_vw_expanded_std.shape)

data_vw_expanded_std.head()
X_train, X_test, y_train, y_test = train_test_split(data_vw_expanded_std.drop(columns = ['price']), data_vw_expanded_std[['price']])

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
column_names = data_vw_expanded.drop(columns = ['price']).columns



no_of_features = []

r_squared_train = []

r_squared_test = []



for k in range(3, 40, 2):

    selector = SelectKBest(f_regression, k = k)

    X_train_transformed = selector.fit_transform(X_train, y_train)

    X_test_transformed = selector.transform(X_test)

    regressor = LinearRegression()

    regressor.fit(X_train_transformed, y_train)

    no_of_features.append(k)

    r_squared_train.append(regressor.score(X_train_transformed, y_train))

    r_squared_test.append(regressor.score(X_test_transformed, y_test))

    

sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')

sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')
selector = SelectKBest(f_regression, k = 23)

X_train_transformed = selector.fit_transform(X_train, y_train)

X_test_transformed = selector.transform(X_test)

column_names[selector.get_support()]
def regression_model(model):

    """

    Will fit the regression model passed and will return the regressor object and the score

    """

    regressor = model

    regressor.fit(X_train_transformed, y_train)

    score = regressor.score(X_test_transformed, y_test)

    return regressor, score
model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])



models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]



for model in models_to_evaluate:

    regressor, score = regression_model(model)

    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)



model_performance
regressor = sm.OLS(y_train, X_train).fit()

print(regressor.summary())



X_train_dropped = X_train.copy()
while True:

    if max(regressor.pvalues) > 0.05:

        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]

        print("Dropping " + drop_variable.index[0] + " and running regression again because pvalue is: " + str(drop_variable[0]))

        X_train_dropped = X_train_dropped.drop(columns = [drop_variable.index[0]])

        regressor = sm.OLS(y_train, X_train_dropped).fit()

    else:

        print("All p values less than 0.05")

        break
print(regressor.summary())
poly = PolynomialFeatures()

X_train_transformed_poly = poly.fit_transform(X_train)

X_test_transformed_poly = poly.transform(X_test)



print(X_train_transformed_poly.shape)



no_of_features = []

r_squared = []



for k in range(10, 277, 5):

    selector = SelectKBest(f_regression, k = k)

    X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)

    regressor = LinearRegression()

    regressor.fit(X_train_transformed, y_train)

    no_of_features.append(k)

    r_squared.append(regressor.score(X_train_transformed, y_train))

    

sns.lineplot(x = no_of_features, y = r_squared)
selector = SelectKBest(f_regression, k = 110)

X_train_transformed = selector.fit_transform(X_train_transformed_poly, y_train)

X_test_transformed = selector.transform(X_test_transformed_poly)
models_to_evaluate = [LinearRegression(), Ridge(), Lasso(), SVR(), RandomForestRegressor(), MLPRegressor()]



for model in models_to_evaluate:

    regressor, score = regression_model(model)

    model_performance = model_performance.append({"Features": "Polynomial","Model": model, "Score": score}, ignore_index=True)



model_performance