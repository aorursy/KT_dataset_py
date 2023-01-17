import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import statsmodels.api as sm



from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, cross_validate, KFold

from sklearn.compose import TransformedTargetRegressor

from sklearn.model_selection import learning_curve





from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.pipeline import Pipeline
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
##Load the dataset

data = pd.read_csv("/kaggle/input/housing/housing.csv")
##Drop the first column

data = data.drop("Unnamed: 0", axis = 1)
data.info()
data.describe()
plt.figure(figsize = (10, 5))

sns.distplot(data["medv"], kde= False)
data[data["medv"] == 50].shape[0]
data = data[data["medv"] < 50]
plt.figure(figsize = (15, 10))

sns.heatmap(data.corr(), annot=True)
plt.figure(figsize = (10, 10))

sns.scatterplot(x = "lstat", y = "medv", data = data)
plt.figure(figsize = (10, 10))

sns.scatterplot(x = "rm", y = "medv", data = data)
plt.figure(figsize = (10, 10))

sns.scatterplot(x = "indus", y = "medv", data = data)
##Extract dependent and independent variables



y = data["medv"]

X = data.drop("medv", axis = 1)
## Split dataset as train/test



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
scaler = StandardScaler()

X_train_l = scaler.fit_transform(X_train)

y_train_l = y_train - np.average(y_train) ##Centering the target data
X_train_l = sm.add_constant(X_train_l)

model = sm.OLS(y_train_l, X_train_l)

results = model.fit()
results.summary()
##List of the column indices to keep



features_keep = list(range(X_train_l.shape[1]))
p_val = 1

while p_val > 0.05:

    model = sm.OLS(y_train_l, X_train_l[:, features_keep])

    results = model.fit()

    ##Find the index corresponding to the maximum p-value

    idx_max = np.argmax(results.pvalues)

    p_val = results.pvalues[idx_max]

    ##In case the p-value is above the threshold, remove the corresponding feature

    if (p_val > 0.05):

        features_keep.pop(idx_max)
model = sm.OLS(y_train_l, X_train_l[:, features_keep])

results = model.fit()

results.summary()
##We drop the first entry in features_keep, which corresponds to the intercept

columns_to_keep = [X_train.columns[i-1] for i in features_keep[1:]]

X_train_l = X_train[columns_to_keep]

X_test_l = X_test[columns_to_keep]
##We will use Linear Regression from Scikit-learn which natively fits into a pipeline for cross-validation



regressor = LinearRegression()

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("regressor", regressor)])
transformer = StandardScaler(with_std= False)



model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )
cv = KFold(n_splits= 5, shuffle = True)

score = cross_validate(model_t, X_train_l, y_train, cv = cv, scoring='r2')

print("The average R^2-score is: ", np.average(score["test_score"]))
cv = KFold(n_splits= 5, shuffle = True)

score = cross_validate(model_t, X_train, y_train, cv = cv, scoring='r2')

print("The average R^2-score is: ", np.average(score["test_score"]))
##We can quickly introduce interactions, and check the averaged R^2-score for different degrees

degrees = [2, 3, 4, 5]

for deg in degrees:

    interactions = PolynomialFeatures(degree = deg)

    regressor = LinearRegression()

    scaler = StandardScaler()

    model = Pipeline([("scaler", scaler), 

                      ("interactions", interactions),

                      ("regressor", regressor)])

    transformer = StandardScaler(with_std= False)

    model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

    cv = KFold(n_splits= 5, shuffle = True)

    score = cross_validate(model_t, X_train_l, y_train, cv = cv, scoring='r2')

    print("The average R^2-score for the model with degree ", deg, "interactions is: ", np.average(score["test_score"]))
neighbors = [2, 3, 4, 5, 6, 8, 10]

for n in neighbors:

    regressor = KNeighborsRegressor(n_neighbors = n)

    scaler = StandardScaler()

    model = Pipeline([("scaler", scaler), 

                      ("regressor", regressor)])

    transformer = StandardScaler(with_std= False)

    model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

    cv = KFold(n_splits= 5, shuffle = True)

    score = cross_validate(model_t, X_train_l, y_train, cv = cv, scoring='r2')

    print("The average R^2-score for the model with ", n, "neighbors is: ", np.average(score["test_score"]))
n_estimators = [10, 20, 30, 50, 100, 150, 200, 300]

for n in n_estimators:

    regressor = RandomForestRegressor(n_estimators = n)

    scaler = StandardScaler()

    model = Pipeline([("scaler", scaler), 

                      ("regressor", regressor)])

    transformer = StandardScaler(with_std= False)

    model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

    cv = KFold(n_splits= 5, shuffle = True)

    score = cross_validate(model_t, X_train_l, y_train, cv = cv, scoring='r2')

    print("The average R^2-score for the model with ", n, "estimators is: ", np.average(score["test_score"]))
n_est = 50

regressor = RandomForestRegressor(n_estimators = n_est)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("regressor", regressor)])

transformer = StandardScaler(with_std= False)

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

cv = KFold(n_splits= 5, shuffle = True)

score = cross_validate(model_t, X_train, y_train, cv = cv, scoring='r2')

print("The average R^2-score for the model with ", n_est, "estimators is: ", np.average(score["test_score"]))
regressor = LinearRegression()

interactions = PolynomialFeatures(degree = 2)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("interactions", interactions),

                  ("regressor", regressor)])

transformer = StandardScaler(with_std= False)

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

cv = KFold(n_splits= 5, shuffle = True)



n_sizes, train_scores, test_scores = learning_curve(model_t, X_train_l, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = cv, scoring= "neg_mean_squared_error")



plt.figure(figsize = (10, 10))

sns.lineplot(n_sizes, np.average(train_scores, axis = 1) * (-1), marker= 'o')

sns.lineplot(n_sizes, np.average(test_scores, axis = 1) * (-1), marker = 'o')
plt.figure(figsize = (10, 10))

sns.lineplot(n_sizes[6:], np.average(train_scores, axis = 1)[6:] * (-1), marker= 'o')

sns.lineplot(n_sizes[6:], np.average(test_scores, axis = 1)[6:] * (-1), marker = 'o')
regressor = KNeighborsRegressor(n_neighbors = 4)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("regressor", regressor)])

transformer = StandardScaler(with_std= False)

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

cv = KFold(n_splits= 5, shuffle = True)



n_sizes, train_scores, test_scores = learning_curve(model_t, X_train_l, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = cv, scoring= "neg_mean_squared_error")



plt.figure(figsize = (10, 10))

sns.lineplot(n_sizes, np.average(train_scores, axis = 1) * (-1), marker= 'o')

sns.lineplot(n_sizes, np.average(test_scores, axis = 1) * (-1), marker = 'o')
regressor = RandomForestRegressor(n_estimators = 50, max_depth = None)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("regressor", regressor)])

transformer = StandardScaler(with_std= False)

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

cv = KFold(n_splits= 5, shuffle = True)



n_sizes, train_scores, test_scores = learning_curve(model_t, X_train_l, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = cv, scoring= "neg_mean_squared_error")



plt.figure(figsize = (10, 10))

sns.lineplot(n_sizes, np.average(train_scores, axis = 1) * (-1), marker= 'o')

sns.lineplot(n_sizes, np.average(test_scores, axis = 1) * (-1), marker = 'o')
regressor = RandomForestRegressor(n_estimators = 50, max_depth = 4)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("regressor", regressor)])

transformer = StandardScaler(with_std= False)

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

cv = KFold(n_splits= 5, shuffle = True)



n_sizes, train_scores, test_scores = learning_curve(model_t, X_train_l, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = cv, scoring= "neg_mean_squared_error")



plt.figure(figsize = (10, 10))

sns.lineplot(n_sizes, np.average(train_scores, axis = 1) * (-1), marker= 'o')

sns.lineplot(n_sizes, np.average(test_scores, axis = 1) * (-1), marker = 'o')
regressor = LinearRegression()

interactions = PolynomialFeatures(degree = 2)

scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                  ("interactions", interactions),

                  ("regressor", regressor)])

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

model_t.fit(X_train_l, y_train)

y_pred = model_t.predict(X_test_l)

score_mean = mean_squared_error(y_test, y_pred)

score_r2 = r2_score(y_test, y_pred)



print("The mean squared error evaluated on the test dataset is: ", score_mean)

print("The R^{2} evaluated on the test dataset is: ", score_r2)
regressor = RandomForestRegressor(n_estimators = 50, max_depth = 4)



scaler = StandardScaler()

model = Pipeline([("scaler", scaler), 

                

                  ("regressor", regressor)])

model_t = TransformedTargetRegressor(regressor = model, transformer = transformer )

model_t.fit(X_train, y_train)

y_pred = model_t.predict(X_test)

score_mean = mean_squared_error(y_test, y_pred)

score_r2 = r2_score(y_test, y_pred)



print("The mean squared error evaluated on the test dataset is: ", score_mean)

print("The R^{2} evaluated on the test dataset is: ", score_r2)