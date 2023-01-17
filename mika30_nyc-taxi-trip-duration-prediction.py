from datetime import date

from lightgbm import LGBMRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import plotly.express as px

import seaborn as sns

sns.set()



seed = 42
PATH_TO_DATA = "/kaggle/input/ieor242hw4"

train = pd.read_csv(PATH_TO_DATA + "/train.csv", parse_dates=["pickup_datetime"])

test = pd.read_csv(PATH_TO_DATA + "/test.csv", parse_dates=["pickup_datetime"])

sub = pd.read_csv(PATH_TO_DATA + "/submission.csv")



print("Train sample:")

display(train.sample(5))

print("Test sample:")

display(test.sample(5))
PATH_TO_LOOKUP = "/kaggle/input/nyc-yellow-taxi-zone-lookup-table"

lookup = pd.read_csv(PATH_TO_LOOKUP + "/taxi_zone_lookup.csv")



loc_to_borough = dict(zip(lookup["LocationID"], lookup["Borough"].apply(lambda x: str(x).lower())))

loc_to_zone = dict(zip(lookup["LocationID"], lookup["Zone"].apply(lambda x: str(x).lower())))



zone_to_loc = {value: key for key, value in loc_to_zone.items()}

zone_to_loc["unknown"] = 265



borough_to_label = {'Bronx': 1, 'Brooklyn': 2, 'EWR': 3, 'Manhattan': 4, 'Queens': 5, 'Staten Island': 6, 'Unknown': 7}

borough_to_label = {key.lower(): value for key, value in borough_to_label.items()}
train.isnull().mean()
train[

    train["VendorID"].isnull() |

    train["passenger_count"].isnull()

][["VendorID", "passenger_count"]].isnull().mean()
test.isnull().mean()
out_val = 999



def preprocess_data(data):

    # VendorID and passenger count

    data["VendorID"] = data["VendorID"].replace({np.nan: 0}).apply(int)

    data["passenger_count"] = data["passenger_count"].replace({np.nan: 0}).apply(int)



    # Pickup and dropoff boroughs and zones

    data["pickup_borough"] = data["pickup_borough"].apply(lambda x: borough_to_label[x.lower()])

    data["dropoff_borough"] = data["dropoff_borough"].apply(lambda x: borough_to_label[x.lower()])

    data["pickup_zone"] = data["pickup_zone"].replace({np.nan: "Unknown"}).apply(lambda x: zone_to_loc[x.lower()])

    data["dropoff_zone"] = data["dropoff_zone"].replace({np.nan: "Unknown"}).apply(lambda x: zone_to_loc[x.lower()])



    # Pickup datetime

    data["pickup_datetime"] = data["pickup_datetime"].replace({np.nan: out_val})

    data["pickup_year"] = data["pickup_datetime"].apply(lambda x: int(x.year) if x != out_val else out_val)

    data["pickup_month"] = data["pickup_datetime"].apply(lambda x: int(x.month) if x != out_val else out_val)

    data["pickup_day"] = data["pickup_datetime"].apply(lambda x: int(x.day) if x != out_val else out_val)

    data["pickup_hour"] = data["pickup_datetime"].apply(lambda x: x.hour if x != out_val else out_val)

    data["pickup_minute"] = data["pickup_datetime"].apply(lambda x: x.minute if x != out_val else out_val)

   

    # Further extraction: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timestamp.html

    data["pickup_dayofweek"] = data["pickup_datetime"].apply(lambda x: int(x.dayofweek) if x != out_val else out_val)

    data["pickup_dayofyear"] = data["pickup_datetime"].apply(lambda x: int(x.dayofyear) if x != out_val else out_val)

    # data["pickup_weekofyear"] = data["pickup_datetime"].apply(lambda x: int(x.weekofyear) if x != out_val else out_val)

    

    # Drop useless columns

    data = data.drop(columns=["row_id"])

    

    return data.reset_index(drop=True)





train = preprocess_data(train)

test = preprocess_data(test)
categorical_columns = [

    "VendorID", "passenger_count", "pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_minute",

    "pickup_dayofweek", "pickup_borough", "dropoff_borough"

]

numerical_columns = ["trip_distance", "pickup_dayofyear"]





def display_data(data):

    nb_rows, nb_cols = max((len(categorical_columns + numerical_columns) - 3) // 4 + 1, 2), 4

    fig, ax = plt.subplots(figsize=(14.5, 4 * nb_rows), nrows=nb_rows, ncols=nb_cols)

    for ind, column in enumerate(categorical_columns + numerical_columns):

        if column in categorical_columns:

            sns.countplot(data[data[column] != out_val][column], ax=ax[ind // nb_cols, ind % nb_cols])

        elif column in numerical_columns:

            sns.distplot(data[data[column] != out_val][column], ax=ax[ind // nb_cols, ind % nb_cols])

            ax[ind // nb_cols, ind % nb_cols].set_ylim((0, 0.05))



    plt.show()





print("Train EDA:")

display_data(train)

print("Test EDA:")

display_data(test)
fig, ax = plt.subplots(figsize=(15, 4 * 2), nrows=2, ncols=2)



ax[0, 0].scatter(train["trip_distance"], train["duration"], s=1)

ax[0, 0].set_title("Trip duration vs. distance")

max_duration = 3600 * 3

# Some outlier removal performed here, be careful

train = train[(train["duration"] <= max_duration) & (train["trip_distance"] <= test["trip_distance"].max())]

ax[0, 1].scatter(train["trip_distance"], train["duration"], s=1)

ax[0, 1].set_title("Trip duration vs. distance (duration < 3 hrs)")



negative_dist = train[(train["trip_distance"] < 0)]

ax[1, 0].scatter(negative_dist["trip_distance"], negative_dist["duration"], s=1)

ax[1, 0].set_title("Trip duration vs. distance (distance < 0)")



positive_dist = train[(train["trip_distance"] >= 0)]

ax[1, 1].scatter(positive_dist["trip_distance"], positive_dist["duration"], s=1)

ax[1, 1].scatter(negative_dist["trip_distance"].apply(abs), negative_dist["duration"], s=1)

ax[1, 1].set_xlim(0, 40)

ax[1, 1].set_ylim(0, max_duration)

ax[1, 1].set_title("Trip duration vs. distance (absolute distance)")



plt.show()
fig = px.scatter(

    x=train["trip_distance"], y=train["duration"], range_x=[0, 40], range_y=[0, 3600 * 2]

)

fig.update_traces(marker=dict(size=3))

fig.show()
fig, ax = plt.subplots(figsize=(15, 4), nrows=1, ncols=2)

ax[0].scatter(train["trip_distance"], train["duration"], s=1)

slow_distance, slow_duration = 2.61, 29828

fast_distance, fast_duration = 35.7, 2124

ax[0].plot(

    [0.5, 0.5, fast_distance, 125],

    [0, 0.5 * fast_duration / fast_distance, fast_duration, 125 * fast_duration / fast_distance],

    color="green"

)

ax[0].plot(

    [0, slow_distance * 3000 / slow_duration, slow_distance * 2, slow_distance * 86400 / slow_duration],

    [3000, 3000, slow_duration * 2, 86400],

    color="green"

)

ax[0].set_xlim(0, 40)

ax[0].set_ylim(0, 3600 * 2)

ax[0].set_title("Trip duration vs. distance (with outliers)")



outliers = (

    (train["trip_distance"] == 0) |

    (train["duration"] == 0) |

    (train["trip_distance"] / train["duration"] >= fast_distance / fast_duration) & (train["trip_distance"] >= 0.5) |

    (train["trip_distance"] / train["duration"] <= slow_distance / slow_duration) & (train["duration"] >= 3000)

)

ax[1].scatter(train[~outliers]["trip_distance"], train[~outliers]["duration"], s=1)

ax[1].set_xlim(0, 40)

ax[1].set_ylim(0, 3600 * 2)

ax[1].set_title("Trip duration vs. distance (without outliers)")



plt.show()
def correct_outliers(data):

    data["trip_distance"] = data["trip_distance"].apply(abs)

    

    return data.reset_index(drop=True)





train = correct_outliers(train)

test = correct_outliers(test)
def remove_outliers(data):

    data = data[(data["duration"] <= max_duration)]  # Already removed before, be careful

    outliers = (

        (data["trip_distance"] == 0) |

        (data["duration"] == 0) |

        (data["trip_distance"] / data["duration"] >= fast_distance / fast_duration) & (data["trip_distance"] >= 0.5) |

        (data["trip_distance"] / data["duration"] <= slow_distance / slow_duration) & (data["duration"] >= 3000)

    )

    data = data[~outliers]



    return data.reset_index(drop=True)





train = remove_outliers(train)
plt.figure(figsize=(15, 4))

train["duration"].apply(np.log).hist(bins=160)

plt.xlim(4, 9)

plt.show()
def feature_engineering(data):

    data["pickup_quarterhour"] = data["pickup_datetime"].apply(

        lambda x: (x - pd.Timestamp(int(x.year), 1, 1)).seconds // (60 * 15) if x != out_val else out_val

    )

    data = data.drop(columns=[

        "pickup_datetime", "pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_minute"

    ])

    

    return data.reset_index(drop=True)





train = feature_engineering(train)

test = feature_engineering(test)
def one_hot_encoding(data):

    for i in range(1, 8):

        data["pickup_borough_{}".format(i)] = data["pickup_zone"].apply(

            lambda x: x if borough_to_label[loc_to_borough[x]] == i else 0

        )

        data["dropoff_borough_{}".format(i)] = data["dropoff_zone"].apply(

            lambda x: x if borough_to_label[loc_to_borough[x]] == i else 0

        )



    data = data.drop(columns=["pickup_zone", "dropoff_zone", "pickup_borough", "dropoff_borough"])

    

    return data





train = one_hot_encoding(train)

test = one_hot_encoding(test)



train.shape
(train >= 0).mean()
def downcast_data(data):

    data["trip_distance"] = (100 * data["trip_distance"]).astype(int)



    for column in data.columns:

        data[column] = pd.to_numeric(data[column], downcast='unsigned')

    

    return data.reset_index(drop=True)





train = downcast_data(train)

test = downcast_data(test)



train.dtypes
plt.figure(figsize=(15, 15))

sns.heatmap(np.abs(np.round(train.corr(), 2)), square=True, annot=True, cmap=plt.cm.Blues)

plt.show()
X, y = train.drop(columns=["duration"]), train["duration"]

X_test = test



validate = True



if validate:

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
commargs = {"learning_rate": 0.03, "colsample_bytree": 0.9, "reg_lambda": 0.2, "random_state": seed, "n_jobs": -1}



lgbm_63 = LGBMRegressor(n_estimators=12000, num_leaves=63, **commargs)

lgbm_127 = LGBMRegressor(n_estimators=6000, num_leaves=127, **commargs)

lgbm_255 = LGBMRegressor(n_estimators=4000, num_leaves=255, **commargs)

lgbm_511 = LGBMRegressor(n_estimators=2000, num_leaves=511, **commargs)

lgbm_1023 = LGBMRegressor(n_estimators=1000, num_leaves=1023, **commargs)



lgbm_estimators = [

    ("LGBM_63", lgbm_63), ("LGBM_127", lgbm_127), ("LGBM_255", lgbm_255),

    ("LGBM_511", lgbm_511), ("LGBM_1023", lgbm_1023),

]



lgbm_voting = VotingRegressor(lgbm_estimators)
def plot_classifiers_validation(regs, X_train, y_train, X_val, y_val):

    fitted_regs = []

    nb_rows, nb_cols = 2, 3

    fig, ax = plt.subplots(figsize=(15, 10), nrows=nb_rows, ncols=nb_cols)

    for ind, reg in enumerate(regs):

        reg.fit(X_train, np.log(y_train))

        fitted_regs.append(reg)

        y_pred = reg.predict(X_val)

        ax[ind // nb_cols, ind % nb_cols].scatter(y_val, np.exp(y_pred), s=1)

        max_plot_value = max(y_val.max(), np.exp(y_pred).max())

        ax[ind // nb_cols, ind % nb_cols].plot([0, max_plot_value], [0, max_plot_value], color="orange")

        ax[ind // nb_cols, ind % nb_cols].set_title("IN-RMSE = {0:.2f}, IN-RMSLE = {1:.5f}".format(

            np.sqrt(mean_squared_error(y_val[y_val <= 7000], np.exp(y_pred)[y_val <= 7000])),

            np.sqrt(mean_squared_log_error(y_val[y_val <= 7000], np.exp(y_pred)[y_val <= 7000]))

        ))

        ax[ind // nb_cols, ind % nb_cols].set_xlim(0, max_plot_value)

        ax[ind // nb_cols, ind % nb_cols].set_ylim(0, max_plot_value)

    plt.show()

    return fitted_regs





if validate:

    lgbm_regs = plot_classifiers_validation(

        [lgbm_63, lgbm_127, lgbm_255, lgbm_511, lgbm_1023, lgbm_voting], X_train, y_train, X_val, y_val

    )
categorical_columns = ["VendorID", "passenger_count", "pickup_dayofweek", "pickup_borough", "dropoff_borough"]

numerical_columns = ["trip_distance", "pickup_dayofyear", "pickup_quarterhour"]





def plot_lgbm_feature_importance(clf, ax):

    ft_imp_dummies = dict(zip(X_train.columns, clf.feature_importances_))

    ft_imp = {

        column: sum([value for key, value in ft_imp_dummies.items() if column in key])

        if column in categorical_columns else ft_imp_dummies[column]

        for column in categorical_columns + numerical_columns

    }

    ft_imp = {key: value for key, value in sorted(ft_imp.items(), key=lambda item: item[1])}



    labels, values = list(ft_imp.keys()), list(ft_imp.values())

    ylocs = np.arange(len(values))

    ax.barh(ylocs, values, align='center', height=0.2)

    for x, y in zip(values, ylocs):

        ax.text(x + 1, y, x, va='center')

    ax.set_yticks(ylocs)

    ax.set_yticklabels(labels)

    ax.set_title("Feature importance for LGBM")





if validate:

    fig, ax = plt.subplots(figsize=(13, 8), nrows=3, ncols=2)

    for ind, reg in enumerate(lgbm_regs[:-1]):

        plot_lgbm_feature_importance(reg, ax=ax[ind // 2, ind % 2])

    plt.subplots_adjust(wspace=0.5, hspace=0.4)

    plt.show()
fit_predict = True

predict_on_train = True



if fit_predict:

    plt.figure(figsize=(15, 4))

    lgbm_voting.fit(X, np.log(y))



    if predict_on_train:

        y_pred = np.exp(lgbm_voting.predict(X))

        print("Train RMSE: ", np.sqrt(mean_squared_error(y, y_pred)))

        plt.hist(y, density=True, bins=[50 * i for i in range(160)])



    y_sub = np.exp(lgbm_voting.predict(X_test))

    plt.hist(y_sub, density=True, bins=[50 * i for i in range(160)], alpha=0.5)

    plt.xlim((0, 3500))

    plt.show()



    sub["duration"] = y_sub

    display(sub)

    sub.to_csv("lgbm-voting-final.csv", index=False)