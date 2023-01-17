import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import make_scorer

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, RidgeCV, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_validate

from sklearn.inspection import permutation_importance
train_data = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv",

        index_col=['Id'],

        na_values="?")
train_data.shape
train_data.info()
train_data.head()
corr = train_data.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr, cmap='autumn', annot=True, square=True)

plt.show()
all_columns = ["longitude", "latitude", "median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]

target = "median_house_value"



X_train = train_data[all_columns]

Y_train = train_data[target]
scaler = MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)

X_train.describe()
def rmsle(y, y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))



scorer = make_scorer(rmsle, greater_is_better=False)
reg = LinearRegression()

cv_results = cross_validate(reg, X_train, Y_train, cv=10, n_jobs = -1, scoring = scorer)

score = np.average(cv_results["test_score"])

print("Regressão linear:", score)
reg = RidgeCV(alphas = np.logspace(-3, 2, 100), cv = 10, normalize = False, scoring = scorer).fit(X_train, Y_train)

print(f"Regressão Ridge - alpha = {reg.alpha_}, score = {reg.best_score_}")
best_score = 100

best_alpha = 0

best_model = None

for alpha in np.logspace(-5, 5, 100):

    reg = Lasso(alpha = alpha, normalize = False)



    cv_results = cross_validate(reg, X_train, Y_train, cv=10, scoring = scorer, n_jobs = -1)

    score = np.average(cv_results["test_score"])

    

    if (abs(score) < abs(best_score)):

        best_score = score

        best_alpha = alpha

        best_model = reg

print(f"Regressão Lasso - alpha = {best_alpha}, score = {best_score}")
reg = Lasso(normalize = False, alpha = 756).fit(X_train, Y_train)
test_data = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv",

        index_col=['Id'],

        na_values="?")



X_test = scaler.transform(test_data)
Y_test = reg.predict(X_test)
test_data[target] = Y_test

test_data.to_csv("answers.csv", columns=[target])