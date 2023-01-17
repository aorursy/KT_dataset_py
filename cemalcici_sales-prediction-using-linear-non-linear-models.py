# Connect Google Cloud

PROJECT_ID = 'advertising-linear-nonlinear'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
import numpy as np # linear algebra

import pandas as pd # data manipulation

import seaborn as sns # data visualization

from matplotlib import pyplot as plt # data visualization
# import data

advs = pd.read_csv("../input/advertising-dataset/advertising.csv")

df = advs.copy()

df.head()
df.info() # metadata
# describe statistics

df.describe([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
def custom_dist_plots(dataframe):

    for col in dataframe.columns:

        sns.distplot(dataframe[col],hist=False).set_title(f"{col} Distribution Graph")

        plt.axvline(dataframe[col].mean(),color='r',label='mean')

        plt.axvline(np.median(dataframe[col]),color='b',label='median')

        plt.axvline((dataframe[col].mode())[0],color='g',label='mode')

        plt.legend()

        plt.show();



custom_dist_plots(df)
# Sales related other variables

def target_scatter(dataframe):

    cols = [col for col in dataframe.columns if col != "Sales"]

    sns.pairplot(dataframe, x_vars=cols, y_vars="Sales", height=4, aspect=1, kind='scatter')

    plt.show()



target_scatter(df)
# correlation graph

sns.heatmap(df.corr(), cmap="Dark2", annot = True)

plt.show()
def missing_detection(dataframe, method="boxplot"):

    if method == "boxplot":

        var_names = [col for col in df.columns if col != "Sales"]

        fig, axs = plt.subplots(len(var_names), figsize=(5, 5))

        for i, col in enumerate(var_names):

            sns.boxplot(df[col], ax=axs[i])

        plt.tight_layout()

    elif method == "lof":

        from sklearn.neighbors import LocalOutlierFactor

        clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)

        clf.fit_predict(dataframe)

        scores_df = pd.DataFrame(np.sort(clf.negative_outlier_factor_))

        scores_df.plot(stacked=True, xlim=[0,20], style='.-')

        plt.show()
missing_detection(df) # boxplot
missing_detection(df, "lof") # LOF
from sklearn.linear_model import LinearRegression # Multiple Linear Regression

from sklearn.linear_model import Ridge # Ridge Regression

from sklearn.linear_model import Lasso # Lasso Regression

from sklearn.linear_model import ElasticNet # ElasticNet Regression

from sklearn.neighbors import KNeighborsRegressor # KNN

from sklearn.tree import DecisionTreeRegressor # CART

from sklearn.ensemble import RandomForestRegressor # Random Forests

from sklearn.ensemble import GradientBoostingRegressor # GBM
# holdout method



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

X = df.drop("Sales", axis=1)

y = df.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=6106)
model_grid = {"Linear": LinearRegression(),

              "Ridge": Ridge(),

              "Lasso": Lasso(),

              "ElasticNet": ElasticNet(),

              "KNN": KNeighborsRegressor(),

              "CART": DecisionTreeRegressor(),

              "RF": RandomForestRegressor(random_state=6106),

              "GBM": GradientBoostingRegressor()}



scores_dict = {}

for name, model in model_grid.items():

    model.fit(X_train, y_train)

    rmse = np.mean(np.sqrt(-cross_val_score(model, X_test, y_test, cv=10, scoring="neg_mean_squared_error")))

    scores_dict[name] = rmse



scores_dict = {k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1])}

scores_dict
# If you want the top 4 models do the following codes



from itertools import islice

top_4_scores = dict(islice(scores_dict.items(), 4))

print("Top 4 Scores")

top_4_scores
# Random Forests Regression



# setting model

rf_model = RandomForestRegressor(random_state=6106)



# seting param grid

rf_params = {"max_depth": [5, 8, None],

             "n_estimators": [200, 500],

             "min_samples_split": [2, 5, 10]}



# search best params

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)



# tuned model

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
# Lasso Regression



# setting model

lasso_model = Lasso()



# seting param grid

lasso_param = {"alpha": 10 ** (np.linspace(10, -2, 100) * 0.5)}



# search best params

lasso_cv_model = GridSearchCV(lasso_model, lasso_param, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)



# tuned model

lasso_tuned = Lasso(**lasso_cv_model.best_params_).fit(X_train, y_train)
# ElasticNet Regression



# setting model

enet_model = ElasticNet()



# seting param grid

enet_params = {"l1_ratio": [0.1, 0.4, 0.5, 0.6, 0.8, 1],

               "alpha": [0.1, 0.01, 0.001, 0.2, 0.3, 0.5, 0.8, 0.9, 1]}



# search best params

enet_cv_model = GridSearchCV(enet_model, enet_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)



# tuned model

enet_tuned = ElasticNet(**enet_cv_model.best_params_).fit(X_train, y_train)
# Ridge Regression



# setting model

ridge_model = Ridge()



# seting param grid

ridge_param = {"alpha": 10 ** (np.linspace(10, -2, 100) * 0.5)}



# search best params

ridge_cv_model = GridSearchCV(ridge_model, ridge_param, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)



# tuned model

ridge_tuned = ElasticNet(**ridge_cv_model.best_params_).fit(X_train, y_train)
from sklearn.metrics import mean_squared_error



# set tuned models

tuned_grid = dict([("RF", rf_tuned), ("Lasso", lasso_tuned), ("ElasticNet", enet_tuned), ("Ridge", ridge_tuned)])



# get rmse values from tuned models

tuned_scores = {}

for name, model in tuned_grid.items():

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    tuned_scores[name] = rmse



# sorting rmse

tuned_scores = {k: v for k, v in sorted(tuned_scores.items(), key=lambda item: item[1])}



# choose best model

best_model = dict(islice(tuned_scores.items(), 1))

best_model
# visualization RF (first tree)

from sklearn.tree import plot_tree



plt.figure(figsize=(20,20))

plot_tree(rf_tuned.estimators_[0], filled=True)

plt.show()
rf_cv_model.best_params_
rf_new_model = RandomForestRegressor(max_depth=3, min_samples_split=2, n_estimators=200).fit(X_train, y_train)

plt.figure(figsize=(20,20))

plot_tree(rf_new_model.estimators_[0], filled=True)

plt.show()