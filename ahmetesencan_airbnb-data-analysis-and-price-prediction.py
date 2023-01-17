import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import os

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()
df.shape
df.drop(["id", "name", "host_name", "last_review"], axis = 1, inplace = True)
df.isnull().sum()
df["reviews_per_month"].fillna(df.reviews_per_month.mean(), inplace=True)
plt.figure(figsize = (10,6))

df["host_id"].value_counts().head().sort_values().plot(kind = "barh", color = "darkblue")

plt.xlabel("Count", size = 14)

plt.ylabel("Host ID", size = 14)

plt.title("Top 5 Hosts With Most Posts", size = 18)
df.loc[lambda df: df['host_id'] == 219517861]["neighbourhood"].value_counts().plot(kind = "bar", 

                                                                                   figsize=(10,6), color = "skyblue")

plt.xlabel("Neighbourhood", size = 14)

plt.ylabel("Count", size = 14)
print("There are", df["neighbourhood_group"].nunique(), "distinct values.")
plt.figure(figsize = (10,6))

df["neighbourhood_group"].value_counts().sort_values().plot(kind = "barh", color = "brown")

plt.xlabel("Count", size = 14)

plt.ylabel("Neighbourhood Group", size = 14)
plt.figure(figsize = (10,6))

ng_p_mean_df = df.groupby("neighbourhood_group")["price"].agg("mean")

ng_p_mean_df.sort_values().plot(kind = "barh", color = "pink")

plt.xlabel("Price", size = 14)

plt.ylabel("Neighbourhood Group", size = 14)

plt.title("Average Price of Neighbourhood Groups", size = 18)
print("There are", df["neighbourhood"].nunique(), "distinct values.")
plt.figure(figsize = (10,6))

df["neighbourhood"].value_counts().head(10).sort_values().plot(kind = "barh")

plt.xlabel("Count", size = 14)

plt.ylabel("Neighbourhood", size = 14)

plt.title("Top 10 Neighbourhood", size = 18)
plt.figure(figsize = (10,6))

top_20_n_p_mean_df = df.groupby("neighbourhood")["price"].agg("mean").sort_values(ascending=False).head(20)

top_20_n_p_mean_df.sort_values(ascending=False).plot(kind = "bar", color = "violet")

plt.xlabel("Neighbourhood", size = 14)

plt.ylabel("Price", size = 14)

plt.title("Top 20 Neighbourhood With Highest Mean Price", size = 18)
plt.figure(figsize = (10,6))

sns.scatterplot(df.longitude, df.latitude, hue=df.neighbourhood_group)

plt.ylabel("Latitude", fontsize=14)

plt.xlabel("Longitude", fontsize=14)

plt.title("Distribution of Neighbourhood Group with Respect to Latitude and Longitude", fontsize=18)

plt.legend(prop={"size":12})
plt.figure(figsize = (10,6))

sns.scatterplot(df.longitude, df.latitude, hue=df.room_type)

plt.ylabel("Latitude", fontsize=14)

plt.xlabel("Longitude", fontsize=14)

plt.title("Distribution of Room Type with Respect to Latitude and Longitude", fontsize=18)

plt.legend(prop={"size":12})
plt.figure(figsize = (10,6))

df["room_type"].value_counts().sort_values().plot(kind = "bar", color = "green")

plt.xlabel("Room Type", size = 14)

plt.ylabel("Count", size = 14)
plt.figure(figsize = (10,6))

r_p_mean_df = df.groupby("room_type")["price"].agg("mean")

r_p_mean_df.sort_values(ascending=True).plot(kind = "barh", color = "gray")

plt.xlabel("Price", size = 14)

plt.ylabel("Room Type", size = 14)

plt.title("Room Types With Mean Price", size = 18)
fig, ax = plt.subplots(1,2, figsize = (16,8))

sns.distplot(df.price, color = "darksalmon", ax=ax[0]).set_title("Price Distribution Before Log Transformation",

                                                                size = 16)

sns.distplot(np.log1p(df.price), color = "darksalmon", ax=ax[1]).set_title("Price Distribution After Log Transformation",

                                                                size = 16)

df["price"] = np.log1p(df["price"])
plt.figure(figsize = (10,6))

df_corr = df.corr()

sns.heatmap(df_corr, annot=True, cmap="copper")
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(12,8))

ax1.boxplot(df["price"])

ax1.set_ylabel("Price(log)", size=13)

ax2.boxplot(df["minimum_nights"])

ax2.set_ylabel("Minimum Nights", size=13)

ax3.boxplot(df["number_of_reviews"])

ax3.set_ylabel("Number of Reviews", size=13)

ax4.boxplot(df["reviews_per_month"])

ax4.set_ylabel("Reviews Per Month", size=13)

ax5.boxplot(df["calculated_host_listings_count"])

ax5.set_ylabel("Calculated Host Listings Count", size=13)

ax6.boxplot(df["availability_365"])

ax6.set_ylabel("Availability 365", size=13)

plt.tight_layout(pad=3)
df["neighbourhood"].value_counts().describe()
top_10_neig = df["neighbourhood"].value_counts().sort_values(ascending=False).head(10)

summation = top_10_neig.sum()

percentage = (100 * summation) / df.shape[0]

print(round(percentage, 2), "% of neighbourhood column consists of the 10 most common neighbourhood values.")
other_values = df["neighbourhood"].value_counts().sort_values(ascending=False).tail(df["neighbourhood"].nunique() - 10).index.tolist()

df_new = df.replace(other_values, "Other")

df_new["neighbourhood"].nunique()
df_dummy = pd.get_dummies(df_new, columns=["neighbourhood_group", "neighbourhood", "room_type"], 

                          prefix=["ng", "n", "rt"])

df_dummy.drop(["host_id"], axis=1, inplace=True)
X = df_dummy.drop("price", axis = 1)

y = df_dummy["price"]
scale = StandardScaler()

X_scaled = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.33, random_state = 42)

df_scaled = pd.DataFrame(X_scaled , columns = X.columns)

df_scaled.head()
def models(X_train, X_test, y_train, y_test):

    

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    print("Linear Regression")

    print("-----------------")

    print("Test Score:", r2_score(y_test, y_pred))

    print("Train Score:", lr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test, y_pred))

    print("\n****************************************************\n")

    

    lasso = Lasso(alpha = 0.0001)

    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)

    print("Lasso")

    print("-----------------")

    print("Test Score:", r2_score(y_test, y_pred))

    print("Train Score:", lasso.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test, y_pred))

    print("\n****************************************************\n")

    

    dtr = DecisionTreeRegressor(min_samples_leaf=25)

    dtr.fit(X_train, y_train)

    y_pred= dtr.predict(X_test)

    print("DTR")

    print("-----------------")

    print("Test Score:", r2_score(y_test, y_pred))

    print("Train Score:", dtr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test, y_pred))

    print("\n****************************************************\n")

    

    rfr = RandomForestRegressor(random_state = 42)

    rfr.fit(X_train, y_train)

    y_pred= rfr.predict(X_test)

    print("RFR")

    print("-----------------")

    print("Test Score:", r2_score(y_test, y_pred))

    print("Train Score:", rfr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test, y_pred))

    print("\n****************************************************\n")
models(X_train, X_test, y_train, y_test)
lasso = Lasso(0.0001)

lasso.fit(X_train, y_train)

importance = lasso.coef_

features = pd.DataFrame(importance, df_scaled.columns, columns=['coefficient'])

features.plot(kind = "bar", figsize = (10, 6)).set_title("Lasso Coefficients", size = 16)
dtr = DecisionTreeRegressor(min_samples_leaf=25)

dtr.fit(X_train, y_train)

f_importance = pd.DataFrame({'Importance': dtr.feature_importances_}, 

                            index=df_scaled.columns).sort_values(by='Importance', ascending=False)

f_importance.plot(kind = "bar", color="green", figsize=(10,6)).set_title("DTR Feature Importances", size = 16)
rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)

y_pred= rfr.predict(X_test)

r2_score(y_test, y_pred)

f_importance = pd.DataFrame({'Importance': rfr.feature_importances_}, 

                            index=df_scaled.columns).sort_values(by='Importance', ascending=False)

f_importance.plot(kind = "bar", figsize=(10,6)).set_title("RFR Feature Importances", size = 16)
df["price"] = np.expm1(df["price"])
q1_price = df["price"].quantile(0.25)

q3_price = df["price"].quantile(0.75)

iqr_price = q3_price - q1_price

lower_limit_price = q1_price - 1.5 * iqr_price

upper_limit_price = q3_price + 1.5 * iqr_price



df_filter_price = df[(df["price"] > lower_limit_price) & (df["price"] < upper_limit_price)]



q1_min_nights = df["minimum_nights"].quantile(0.25)

q3_min_nights = df["minimum_nights"].quantile(0.75)

iqr_min_nights = q3_min_nights - q1_min_nights

lower_limit_min_nights = q1_min_nights - 1.5 * iqr_min_nights

upper_limit_min_nights = q3_min_nights + 1.5 * iqr_min_nights



df_filter_min_nights = df_filter_price[(df_filter_price["minimum_nights"] > lower_limit_min_nights) & 

                                       (df_filter_price["minimum_nights"] < upper_limit_min_nights)]



q1_num_rew = df["number_of_reviews"].quantile(0.25)

q3_num_rew = df["number_of_reviews"].quantile(0.75)

iqr_num_rew = q3_num_rew - q1_num_rew

lower_limit_num_rew = q1_num_rew - 1.5 * iqr_num_rew

upper_limit_num_rew = q3_num_rew + 1.5 * iqr_num_rew



df_filter_num_rew = df_filter_min_nights[(df_filter_min_nights["number_of_reviews"] > lower_limit_num_rew) & 

                                       (df_filter_min_nights["number_of_reviews"] < upper_limit_num_rew)]



q1_rpm = df['reviews_per_month'].quantile(0.25)

q3_rpm = df['reviews_per_month'].quantile(0.75)

iqr_rpm = q3_rpm - q1_rpm

lower_limit_rpm = q1_rpm - 1.5 * iqr_rpm

upper_limit_rpm = q3_rpm + 1.5 * iqr_rpm



df_filter_rpm = df_filter_num_rew[(df_filter_num_rew["reviews_per_month"] > lower_limit_rpm) & 

                                       (df_filter_num_rew["reviews_per_month"] < upper_limit_rpm)]



q1_hlc = df['calculated_host_listings_count'].quantile(0.25)

q3_hlc = df['calculated_host_listings_count'].quantile(0.75)

iqr_hlc = q3_hlc - q1_hlc

lower_limit_hlc = q1_hlc - 1.5 * iqr_hlc

upper_limit_hlc = q3_hlc + 1.5 * iqr_hlc



df_filtered = df_filter_rpm[(df_filter_rpm["calculated_host_listings_count"] > lower_limit_hlc) & 

                                       (df_filter_rpm["calculated_host_listings_count"] < upper_limit_hlc)]
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(12,8))

ax1.boxplot(df_filtered["price"])

ax1.set_ylabel("Price", size=13)

ax2.boxplot(df_filtered["minimum_nights"])

ax2.set_ylabel("Minimum Nights", size=13)

ax3.boxplot(df_filtered["number_of_reviews"])

ax3.set_ylabel("Number of Reviews", size=13)

ax4.boxplot(df_filtered["reviews_per_month"])

ax4.set_ylabel("Reviews Per Month", size=13)

ax5.boxplot(df_filtered["calculated_host_listings_count"])

ax5.set_ylabel("Calculated Host Listings Count", size=13)

ax6.boxplot(df_filtered["availability_365"])

ax6.set_ylabel("Availability 365", size=13)

plt.tight_layout(pad=3)
other_values = df_filtered["neighbourhood"].value_counts().sort_values(ascending=False).tail(217 - 30).index.tolist()

df_filtered_new = df_filtered.replace(other_values, "Other")

summation = df_filtered["neighbourhood"].value_counts().sort_values(ascending=False).head(30).sum()

percentage = (100 * summation) / df.shape[0]

print(round(percentage, 2), "% of neighbourhood column consists of the 30 most common neighbourhood values.")
df_dummy_filtered = pd.get_dummies(df_filtered_new, columns=["neighbourhood_group", "neighbourhood", "room_type"], 

                          prefix=["n_g", "n", "r_t"])

df_dummy_filtered.drop(["host_id"], axis=1, inplace=True)
X_filtered = df_dummy_filtered.drop("price", axis = 1)

y_filtered = df_dummy_filtered["price"]

y_filtered = np.log1p(y_filtered)
scale = StandardScaler()

X_scaled_filtered = scale.fit_transform(X_filtered)

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_scaled_filtered, y_filtered, 

                                                                                test_size = 0.33, random_state = 42)

df_scaled_filtered = pd.DataFrame(X_scaled_filtered , columns = X_filtered.columns)

df_scaled_filtered.head()
models(X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered)
def cv_scores(X_train, y_train):

    

    print("Linear Regression")

    print("-----------------")

    lr = LinearRegression()

    print(cross_val_score(lr, X_train, y_train, scoring='r2', cv=5).mean())

    print("\n****************************************************\n")

    

    print("Lasso")

    print("-----------------")

    lasso = Lasso(alpha = 0.0001)

    print(cross_val_score(lasso, X_train, y_train, scoring='r2', cv=5).mean())

    print("\n****************************************************\n")

    

    print("DTR")

    print("-----------------")

    dtr = DecisionTreeRegressor(min_samples_leaf=25)

    print(cross_val_score(dtr, X_train, y_train, scoring='r2', cv=5).mean())

    print("\n****************************************************\n")

    

    print("RFR")

    print("-----------------")

    rfr = RandomForestRegressor(random_state = 42)

    print(cross_val_score(rfr, X_train, y_train, scoring='r2', cv=5).mean())

    print("\n****************************************************\n")
cv_scores(X_train_filtered, y_train_filtered)
def param_tuning(X_train, y_train):

    

    print("Best Hyperparameters for Linear Regression")

    print("-----------------")

    param = {'fit_intercept':[True,False], 

             'normalize':[True,False], 

             'copy_X':[True, False]}

    lr_random = RandomizedSearchCV(estimator = LinearRegression(), 

                                   param_distributions = param, n_iter = 100, cv = 3, 

                                   verbose=2, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

    lr_random.fit(X_train, y_train)

    print(lr_random.best_params_)

    print(lr_random.best_score_ * -1)

    print("\n****************************************************\n")

    

    

    print("Best Hyperparameters for Lasso")

    print("-----------------")

    alpha = {"alpha": [5, 0.5, 0.05, 0.005, 0.0005, 1, 0.1, 0.01, 0.001, 0.0001, 0]}

    lasso_random = RandomizedSearchCV(estimator = Lasso(), 

                                   param_distributions = alpha, n_iter = 100, cv = 3, 

                                   verbose=2, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

    lasso_random.fit(X_train, y_train)

    print(lasso_random.best_params_)

    print(lasso_random.best_score_ * -1)

    print("\n****************************************************\n")

    

    print("Best Hyperparameters for DTR")

    print("-----------------")

    param_dist = {"criterion": ["mse", "mae"],

              "min_samples_split": [10, 20, 40],

              "max_depth": [2, 6, 8],

              "min_samples_leaf": [20, 40, 100],

              "max_leaf_nodes": [5, 20, 100],

              }

    dtr_random = RandomizedSearchCV(estimator = DecisionTreeRegressor(), 

                                   param_distributions = param_dist, n_iter = 100, cv = 3, 

                                   verbose=2, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

    dtr_random.fit(X_train, y_train)

    print(dtr_random.best_params_)

    print(dtr_random.best_score_ * -1)

    print("\n****************************************************\n")

    

    

    print("Best Hyperparameters for RFR")

    print("-----------------")

    random_grid = {'n_estimators': [int(i) for i in range(50, 400, 50)],

                   'max_features': ['auto', 'sqrt'],

                   'max_depth': [int(i) for i in range(10, 60, 10)],

                   'min_samples_split': [2, 5, 10],

                   'min_samples_leaf': [1, 2, 4],

                   'bootstrap': [True, False]

                  }

    rfr_random = RandomizedSearchCV(estimator = RandomForestRegressor(), 

                                   param_distributions = random_grid, n_iter = 100, cv = 3, 

                                   verbose=2, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

    rfr_random.fit(X_train, y_train)

    print(rfr_random.best_params_)

    print(rfr_random.best_score_ * -1)

    print("\n****************************************************\n")
#param_tuning(X_train_filtered, y_train_filtered)
test_score_dict = {}

mae_dict = {}

mse_dict = {}



def modelTuned(X_train, y_train):

    

    lr = LinearRegression(normalize = True, 

                          fit_intercept = True, 

                          copy_X = True)

    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test_filtered)

    print("Linear Regression")

    print("-----------------")

    print("Test Score:", r2_score(y_test_filtered, y_pred))

    print("Train Score:", lr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test_filtered, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test_filtered, y_pred))

    print("\n****************************************************\n")

    test_score_dict["LR"] = r2_score(y_test_filtered, y_pred)

    mae_dict["LR"] = mean_absolute_error(y_test_filtered, y_pred)

    mse_dict["LR"] = mean_squared_error(y_test_filtered, y_pred)

    

    lasso = Lasso(alpha = 0.0001)

    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test_filtered)

    print("Lasso")

    print("-----------------")

    print("Test Score:", r2_score(y_test_filtered, y_pred))

    print("Train Score:", lasso.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test_filtered, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test_filtered, y_pred))

    print("\n****************************************************\n")

    test_score_dict["Lasso"] = r2_score(y_test_filtered, y_pred)

    mae_dict["Lasso"] = mean_absolute_error(y_test_filtered, y_pred)

    mse_dict["Lasso"] = mean_squared_error(y_test_filtered, y_pred)

    

    dtr = DecisionTreeRegressor(min_samples_split = 10, 

                                min_samples_leaf = 40,

                                max_leaf_nodes = 100, 

                                max_depth = 8, 

                                criterion = 'mse')

    dtr.fit(X_train, y_train)

    y_pred= dtr.predict(X_test_filtered)

    print("DTR")

    print("-----------------")

    print("Test Score:", r2_score(y_test_filtered, y_pred))

    print("Train Score:", dtr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test_filtered, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test_filtered, y_pred))

    print("\n****************************************************\n")

    test_score_dict["DTR"] = r2_score(y_test_filtered, y_pred)

    mae_dict["DTR"] = mean_absolute_error(y_test_filtered, y_pred)

    mse_dict["DTR"] = mean_squared_error(y_test_filtered, y_pred)

    



    rfr = RandomForestRegressor(random_state = 42, 

                                n_estimators = 150,

                                min_samples_split = 10,

                                min_samples_leaf = 4,

                                max_features = 'auto',

                                max_depth = 10,

                                bootstrap = True)

    rfr.fit(X_train, y_train)

    y_pred = rfr.predict(X_test_filtered)

    print("RFR")

    print("-----------------")

    print("Test Score:", r2_score(y_test_filtered, y_pred))

    print("Train Score:", rfr.score(X_train, y_train))

    print("Mean Absolute Error:", mean_absolute_error(y_test_filtered, y_pred))

    print("Mean Squared Error", mean_squared_error(y_test_filtered, y_pred))

    print("\n****************************************************\n")

    test_score_dict["RFR"] = r2_score(y_test_filtered, y_pred)

    mae_dict["RFR"] = mean_absolute_error(y_test_filtered, y_pred)

    mse_dict["RFR"] = mean_squared_error(y_test_filtered, y_pred)
modelTuned(X_train_filtered, y_train_filtered)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3 ,figsize = (16,6))

ax1.set_title("R2 Scores")

ax1.plot(list(test_score_dict.keys()), list(test_score_dict.values()), marker = "o", color = "steelblue")

ax2.set_title("MAE")

ax2.plot(list(mae_dict.keys()), list(mae_dict.values()), marker = "o", color = "green")

ax3.set_title("MSE")

ax3.plot(list(mse_dict.keys()), list(mse_dict.values()), marker = "o", color = "orange")
rfr = RandomForestRegressor(random_state = 42, 

                                n_estimators = 150,

                                min_samples_split = 10,

                                min_samples_leaf = 4,

                                max_features = 'auto',

                                max_depth = 10,

                                bootstrap = True)

rfr.fit(X_train_filtered, y_train_filtered)

y_pred_filtered = rfr.predict(X_test_filtered)

comp_airbnb = pd.DataFrame({

        'Actual Values': np.array(np.expm1(y_test_filtered)),

        'Predicted Values': np.expm1(y_pred_filtered)})



comp_airbnb.head(5)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (16,8))

ax1.scatter(np.array(np.expm1(y_test_filtered)),np.expm1(y_pred_filtered))

ax1.set_xlabel("True", size = 14)

ax1.set_ylabel("Prediction", size = 14)

ax2.plot(np.array(np.expm1(y_test_filtered)), label="True")

ax2.plot(np.expm1(y_pred_filtered), label = "Prediction")

ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})
price_higher_250 = df[df["price"] > 250].shape[0]

print("Number of data with price that more than 250 dollars in the data is", price_higher_250, 

      "out of", df["price"].shape[0])