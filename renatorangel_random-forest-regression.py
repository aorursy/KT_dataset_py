import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

import seaborn as sns

sns.set(style="ticks", palette="pastel")



from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline



from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import learning_curve
# got this code from here https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html



def plot_learning_curve(estimator, 

                        title, 

                        X, 

                        y,

                        scorer,

                        ylim=None, 

                        cv=None,

                        n_jobs=None, 

                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize=(12,6))

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator,

                                                            X,

                                                            y,

                                                            cv=cv,

                                                            scoring=scorer,

                                                            n_jobs=n_jobs,

                                                            train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
def calculate_error(test_y, predicted, weights):

    return mean_absolute_error(test_y, predicted, sample_weight=weights)
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')

stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

features.head()


stores.head()


train.head()
test.head()
df = pd.merge(train, features, on=['Store','Date','IsHoliday'], how='inner')

df = pd.merge(df, stores, on='Store', how='inner')

df.head()
test = pd.merge(test, features, on=['Store','Date','IsHoliday'], how='inner')

test = pd.merge(test, stores, on='Store', how='inner')

test.head()
df['Date'] = pd.to_datetime(df['Date'])

df['year'] = df['Date'].dt.year

df['week'] = df['Date'].dt.week



test_id = test["Store"].astype(str) + "_" + test["Dept"].astype(str) + "_" + test["Date"].astype(str)



test['Date'] = pd.to_datetime(test['Date'])

test['year'] = test['Date'].dt.year

test['week'] = test['Date'].dt.week
df.describe()
df["Date"].describe()
test["Date"].describe()
plt.figure(figsize=(12, 8))

sns.lineplot(x="Date", hue="year", y="Weekly_Sales", data=df)

plt.xticks(rotation=15)

plt.title('Vendas de cada ano por semana')

plt.show()
plt.figure(figsize=(8, 8))

sns.boxplot(x="IsHoliday", 

            y="Weekly_Sales",

#             orient="h",

            data=df)

sns.despine(offset=10, trim=True)
plt.figure(figsize=(20,8))

sns.boxplot(x="Dept", 

            y="Weekly_Sales",

#             orient="h",

            data=df)

sns.despine(offset=10, trim=True)
plt.figure(figsize=(20,8))

sns.boxplot(x="Store", 

            y="Weekly_Sales",

#             orient="h",

            data=df)

sns.despine(offset=10, trim=True)
plt.figure(figsize=(8, 8))

sns.boxplot(x="Type", 

            y="Weekly_Sales",

            data=df)

sns.despine(offset=10, trim=True)
plt.figure(figsize=(8, 8))

sns.scatterplot(x="Type", y="Size", data=df)
plt.figure(figsize=(8, 8))

sns.scatterplot(x="Size", y="Weekly_Sales", data=df)
plt.figure(figsize=(8, 8))

sns.scatterplot(x="year",

                y="Fuel_Price",

                data=df)
del df['Date']

del test["Date"]
lb_type = LabelEncoder()

df['Type'] = lb_type.fit_transform(df['Type'])

test['Type'] = lb_type.transform(test['Type'])



lb_is_holiday = LabelEncoder()

df['IsHoliday'] = lb_is_holiday.fit_transform(df['IsHoliday'])

test['IsHoliday'] = lb_is_holiday.transform(test['IsHoliday'])
si = SimpleImputer()

si.fit(df[["CPI"]]) 

test["CPI"] = si.transform(test[["CPI"]])



si = SimpleImputer()

si.fit(df[["Unemployment"]]) 

test["Unemployment"] = si.transform(test[["Unemployment"]])
# Compute the correlation matrix

corr = df.corr().round(decimals=2)



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
df.isna().mean() * 100
df.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"], inplace=True)

test.drop(columns=["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"], inplace=True)



df.drop(columns=["Type"], inplace=True)

test.drop(columns=["Type"], inplace=True)



df.drop(columns=["Fuel_Price"], inplace=True)

test.drop(columns=["Fuel_Price"], inplace=True)
train, validation = train_test_split(df, test_size=.3, random_state=1)



def prepare_data(train):

    X_train = train.drop('Weekly_Sales', axis=1)

    y_train = train['Weekly_Sales']

    X_train.loc[:, "extra"] = X_train['IsHoliday'].replace(1, 5).replace(0, 1)

    return X_train, y_train



X_train, y_train = prepare_data(train)

X_validation, y_validation = prepare_data(validation)
del train, validation, si, df, corr, features, stores
linear_regression = LinearRegression()

decision_tree = DecisionTreeRegressor()



linear_regression.fit(X_train.drop(columns=["extra"]), y_train)

predicted = linear_regression.predict(X_validation.drop(columns=["extra"]))

print(calculate_error(y_validation, predicted, X_validation["extra"]))



decision_tree.fit(X_train.drop(columns=["extra"]), y_train)

predicted = decision_tree.predict(X_validation.drop(columns=["extra"]))

print(calculate_error(y_validation, predicted, X_validation["extra"]))
def new_scorer(estimator, X, y, sample_weight=None):

    return calculate_error(y, estimator.predict(X), X["extra"])

    

def remove_extra(X):

    return X.drop(columns=["extra"])



pipe = Pipeline(steps=[("preprocessing", FunctionTransformer(remove_extra, validate=False)),

                      ("regressor", RandomForestRegressor())])



search_space = {"regressor": [RandomForestRegressor()],

                 "regressor__n_estimators": [100, 200, 300, 600],

                 "regressor__max_depth": [10, 20, 30, 40, 50, 60, 70, 80],

                 "regressor__min_samples_leaf": [1, 2, 4],

                 "regressor__min_samples_split": [2, 8, 12]}





rf_random = RandomizedSearchCV(estimator=pipe, 

                               param_distributions=search_space, 

                               n_iter = 1, 

                               cv = 3, 

                               verbose=2, 

                               random_state=42, 

                               n_jobs = -1,

                               scoring=new_scorer)



# Fit the random search model

rf_random.fit(X_train, y_train)



model = rf_random.best_estimator_



print(rf_random.best_estimator_)

print(rf_random.best_score_)
title = "Learning Curves Random Forest"



lc_svm = plot_learning_curve(model, title, X_train, y_train, scorer=new_scorer, ylim=(0.0, 4000.0), cv=3, n_jobs=-1)

lc_svm.show()
calculate_error(model.predict(X_train), y_train, X_train["extra"])
calculate_error(model.predict(X_validation), y_validation, X_validation["extra"])
test.loc[:, "extra"] = test['IsHoliday'].replace(1, 5).replace(0, 1)



predicted = model.predict(test)

test["Id"] = test_id

test["Weekly_Sales"] = predicted

test[["Id", "Weekly_Sales"]].to_csv("submission.csv", index=False)