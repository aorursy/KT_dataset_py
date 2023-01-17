# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.options.display.max_columns = 30
#importing data
train = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/train.csv.zip", parse_dates = ["Date"])

def snake_case(dataframe):
    #convert column names to snake_case
    dataframe.columns = dataframe.columns.str.lower()
    dataframe = dataframe.rename(columns = {"isholiday" : "is_holiday"})
    return dataframe

#converting column names to snake_case
train = snake_case(train)

#exploring the data
print(train.shape)
train.head()
#importing data
test = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/test.csv.zip", parse_dates = ["Date"])

#converting column names to snake_case
test = snake_case(test)

#exploring the data
print(test.shape)
print(test.shape[0] / train.shape[0]) #how long is the test dataframe in comparison to train the dataframe
test.head()
#importing data
features = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/features.csv.zip", parse_dates = ["Date"])

#converting column names to snake_case
features = snake_case(features)

#exploring the data
print(features.shape)
features.head()
#importing data
stores = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/stores.csv")

#converting column names to snake_case
stores = snake_case(stores)

#exploring the data
print(stores.shape)
stores.head()
#importing data
sample = pd.read_csv("../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")

#exploring the data
print(sample.shape)
sample.head()
#checking for duplicated. If sum is > than 0, it has duplicated rows.
print("duplicated in train:", train.duplicated(subset = ["store", "dept", "date"]).sum())
print("duplicated in test:", test.duplicated(subset = ["store", "dept", "date"]).sum())
print("duplicated in features:", features.duplicated(subset = ["store", "date"]).sum())
print("duplicated in stores:", stores.duplicated(subset = ["store"]).sum())
print("duplicated in sample:", sample.duplicated(subset = ["Id"]).sum())
train.info()
test.info()
stores.info()
features.info()
#importing data visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

#plotting missing data
fig_missing_data, ax_missing_data = plt.subplots(figsize = (18,6))
sns.heatmap(features.isnull(), ax = ax_missing_data)

ax_missing_data.set_title("Missing data by column")
ax_missing_data.set_ylabel("Row index")

plt.show()
#registering converters to avoid warning
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_feature(axes_object, dataframe, column_name, store = None):
    #plot single store
    if store is not None:
        dataframe_mask = dataframe["store"] == store
        
        hue = None #only one line will be plotted
        
        title = "Store {} - {} data".format(store, column_name.title())
    
    #plot all stores
    else:
        # creating mask with only true values
        dataframe_mask = pd.Series(np.ones(len(dataframe), dtype=bool))
        
        hue = dataframe[dataframe_mask]["store"] #create one line for each store
        
        title = "All Stores - {} data".format(column_name.title())
    
    #plotting
    sns.lineplot(dataframe[dataframe_mask]["date"], dataframe[dataframe_mask][column_name], hue = hue, ax = axes_object)
    
    #setup
    axes_object.set_title(title)
    axes_object.set_xlabel("Date")
    axes_object.set_ylabel(column_name.title())
    
    plt.setp(axes_object.xaxis.get_majorticklabels(), rotation=45)

    
#creating figure with four graphs
fig_time, axs_time = plt.subplots(2, 2, figsize = (15, 15))
    
plot_feature(axs_time[0, 0], features, "cpi", store = 1)
plot_feature(axs_time[1, 0], features, "unemployment", store = 1)
plot_feature(axs_time[0, 1], features, "cpi")
plot_feature(axs_time[1, 1], features, "unemployment")
#Checking if "cpi" and "unemployment" data are missing in the same rows
(features["cpi"].isnull() != features["unemployment"].isnull()).sum() # When summing boolean values, True equals 1 
                                                                      # and False equals 0
#creating mask
missing_mask = features["cpi"].isnull()

#dates with missing cpi and unemployment data
missing_dates = features[missing_mask]["date"].unique()
sorted(missing_dates)
#all dates in the dataframe
all_dates = features["date"].unique()
sorted(all_dates)
#column that identify if cpi is null
features["cpi_isnull"] = missing_mask

# pivot table to check if "cpi" and "unemployment" data are missing in the same dates for all stores
pivot_missing_cpi = features.pivot_table("cpi_isnull", "date", "store", aggfunc = np.sum)
pivot_missing_cpi
#counting how many stores have missing "cpi" data for each date
count_missing_stores = pivot_missing_cpi.sum(axis = 1)

# printing the value counts to be sure that all values will be either 0 (no missing data for all stores) 
# or 45 (all stores missing cpi data)
print(count_missing_stores.value_counts())
count_missing_stores.tail(15)
# checking how many times "cpi" and "unemployment" will be missing in each dataframe, after merging with features
missing_cpi_train = train["date"].isin(missing_dates).sum()
missing_cpi_test = test["date"].isin(missing_dates).sum()

print(missing_cpi_train, missing_cpi_test)
#dropping "CPI_isnull" column that was used specifically for our previous sanity check
features = features.drop("cpi_isnull", axis = 1)

#slicing the dataframe with the date of 2013-04-26
features_2013_04_26 = features[features["date"] == "2013-04-26"]

#filling the CPI and Unemployment missing values with the data from the day 2013-04-26 for each store
for store in range(1, 46):
    
    #values to be used to fill
    cpi_value =  features_2013_04_26[features_2013_04_26["store"] == store]["cpi"].iloc[0]
    unemployment_value =  features_2013_04_26[features_2013_04_26["store"] == store]["unemployment"].iloc[0]
    
    #filling the missing values
    indexes = features[(features["store"] == store) & features["cpi"].isnull()].index
    features.loc[indexes, "cpi"] = cpi_value
    features.loc[indexes, "unemployment"] = unemployment_value
cpifig_time, axs_time = plt.subplots(2, 2, figsize = (15, 15))
    
plot_feature(axs_time[0, 0], features, "cpi", store = 1)
plot_feature(axs_time[1, 0], features, "unemployment", store = 1)
plot_feature(axs_time[0, 1], features, "cpi")
plot_feature(axs_time[1, 1], features, "unemployment")
#markdown column names
mark_cols = ["markdown{}".format(count) for count in range(1,6)]

#percentage of missing values
features[mark_cols].isnull().sum() / len(features[mark_cols])
features_zero = features.copy()
features_zero[mark_cols] = features[mark_cols].fillna(0).copy()
features_zero
features_zero.info()
#calculating the mean of "markdown" columns
markdown_holiday = features.groupby("is_holiday")[mark_cols].mean()
markdown_holiday
#identifying the index of holiday rows and non holiday rows
features_holiday_index = features[features["is_holiday"]].index
features_not_holiday_index = features[~features["is_holiday"]].index

features_mean = features.copy()

#filling with the appropriate mean
features_mean.loc[features_holiday_index, mark_cols] = (features_mean.loc[features_holiday_index, 
                                                                          mark_cols
                                                                         ].fillna(markdown_holiday.iloc[1])) # Holiday
features_mean.loc[features_not_holiday_index, mark_cols] = (features_mean.loc[features_not_holiday_index, 
                                                                              mark_cols
                                                                             ].fillna(markdown_holiday.iloc[0])) # Non Holiday
features_mean
features_mean.info()
#merging features with stores dataframes
features_zero = pd.merge(features_zero, stores, how = "left", on = ["store"])
features_mean = pd.merge(features_mean, stores, how = "left", on = ["store"])

#function to group train and test dataframes
def group_dataframe(dataframe, is_train = True):
    #grouping dataframe
    grouped = dataframe.groupby(["store", "date"]).agg(["sum","mean"])
    
    if is_train:
        #selecting the columns with the sum of sales and the mean of is_holiday
        grouped = grouped.iloc[:,[2,5]]

        #renaming the columns
        grouped.columns = ["weekly_sales", "is_holiday"]
    
    else:
        #selecting the column with the mean of is_holiday
        grouped = grouped.iloc[:,3]

        #drop column level
        grouped.name = "is_holiday"
        
    #return dataframe with indexes reset
    return grouped.reset_index()

# merging features with train and test dataframes. We are leaving "is_holiday" column from features
# out to avoid duplicating columns
feature_df_cols = ['store', 'date', 'temperature', 'fuel_price', 'markdown1', 'markdown2',
                   'markdown3', 'markdown4', 'markdown5', 'cpi', 'unemployment', 'type', 'size']

grouped_train = group_dataframe(train, True)
grouped_test = group_dataframe(test, False)

train_zero = pd.merge(grouped_train, features_zero[feature_df_cols], how = "left", on = ["store", "date"])
train_mean = pd.merge(grouped_train, features_mean[feature_df_cols], how = "left", on = ["store", "date"])

#Converting boolean to zero or one
train_zero["is_holiday"] = train_zero["is_holiday"].astype(int)
train_mean["is_holiday"] = train_mean["is_holiday"].astype(int)
#Creating exploratory dataframe
exploratory = train_zero.copy()

#function to create time related columns
def add_time_columns(dataframe):
    #add time related columns
    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["year_month"] = dataframe["date"].dt.to_period("M")
    dataframe["week"] = dataframe["date"].dt.week
    
    return dataframe

#creating dummy function to explore categorical columns
def add_dummy(column_name, dataframe):
    # add dummy columns
    dummy_df = pd.get_dummies(dataframe[column_name], prefix = column_name)
    dataframe = pd.concat([dataframe, dummy_df], axis = 1)
    return dataframe

#adding time related columns and dummy columns to dataframes
exploratory = add_time_columns(exploratory)
exploratory = add_dummy('type', exploratory)

exploratory.head()
#monthly sales
exploratory.groupby("year_month")["weekly_sales"].sum().plot()
plt.title("Monthly Sales")
plt.ylabel("Revenue")
plt.show()
#weekly sales
exploratory.groupby("date")["weekly_sales"].sum().plot()
plt.title("Weekly Sales")
plt.ylabel("Revenue")
plt.show()
#list of holidays from the exploratory set
holiday_weeks = exploratory.query("is_holiday == 1")["date"].unique()

#weekly sales
exploratory.groupby("date")["weekly_sales"].sum().plot()
plt.title("Weekly Sales")
plt.ylabel("Revenue")

#plot holidays
for holiday in holiday_weeks: 
    plt.axvline(holiday, c = "red", lw = 0.5 )

plt.show()
#correlation
correlation_table = np.abs(exploratory.corr()["weekly_sales"]).sort_values(ascending = False)

#removing weekly_sales
correlation_table = correlation_table.drop("weekly_sales")

sns.barplot(correlation_table.values, correlation_table.index, orient = "h")
plt.show()
pd.options.display.max_rows = 150
#table to see in what week of the year sales are the strongest
exploratory.groupby("date")["weekly_sales","is_holiday", "week"].agg(["sum", "mean"]).iloc[:, [0,3,5]]
#creating new column
exploratory["is_strong_sales"] = 0

# it is 1 if weeks are 47, 49, 50 or 51
strong_weeks = [47, 49, 50, 51]

exploratory["is_strong_sales"] = exploratory["is_strong_sales"].mask(exploratory["week"].isin(strong_weeks), 1)
#correlation
correlation_table = np.abs(exploratory.corr()["weekly_sales"]).sort_values(ascending = False)

#removing weekly_sales
correlation_table = correlation_table.drop("weekly_sales")

sns.barplot(correlation_table.values, correlation_table.index, orient = "h")
plt.show()
#changing the name of exploratory dataframe for standardization
exploratory_zero = exploratory

#Creating exploratory dataframe using train_mean instead of train_zero
exploratory_mean = train_mean.copy()

#adding time related columns and dummy columns to dataframes
exploratory_mean = add_time_columns(exploratory_mean)
exploratory_mean = add_dummy('type', exploratory_mean)

#creating new column
exploratory_mean["is_strong_sales"] = 0

# it is 1 if weeks are 47, 49, 50 or 51
exploratory_mean["is_strong_sales"] = exploratory_mean["is_strong_sales"].mask(exploratory_mean["week"].isin(strong_weeks), 1)

exploratory_mean.head()
#correlation
correlation_table_mean = np.abs(exploratory_mean.corr()["weekly_sales"]).sort_values(ascending = False)

#removing weekly_sales
correlation_table_mean = correlation_table_mean.drop("weekly_sales")

sns.barplot(correlation_table_mean.values, correlation_table_mean.index, orient = "h")
plt.show()
def normalize(list_of_columns, dataframe):
    #normalize dataframe making it range from 0 to 1
    normal_df = ((dataframe[list_of_columns] - dataframe[list_of_columns].min()) / 
                 (dataframe[list_of_columns].max() - dataframe[list_of_columns].min()))

    return normal_df

#features
numerical_cols = ['is_holiday', 'temperature', 'fuel_price', 'markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5', 
                  'cpi', 'unemployment', 'size', 'type_A', 'type_B', 'type_C', 'is_strong_sales', 'week', 'month']

normal_train_zero = normalize(numerical_cols, exploratory_zero)
normal_train_mean = normalize(numerical_cols, exploratory_mean)

target = exploratory_zero["weekly_sales"]
# importing models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score

#top 6 features
top_features = ["size", "type_A", "type_B", "markdown1", "markdown5", "is_strong_sales"]

#models
knn = KNeighborsRegressor()
kf = KFold(10, shuffle = True)

#root_mean_squared_error
knn_zero_all_features_rmse = (-np.mean(cross_val_score(knn, normal_train_zero, target, cv = kf, 
                                                 scoring = "neg_mean_squared_error")))**(1/2)
knn_zero_six_features_rmse = (-np.mean(cross_val_score(knn, normal_train_zero[top_features], target, cv = kf, 
                                                 scoring = "neg_mean_squared_error")))**(1/2)

knn_mean_all_features_rmse = (-np.mean(cross_val_score(knn, normal_train_mean, target, cv = kf, 
                                                 scoring = "neg_mean_squared_error")))**(1/2)
knn_mean_six_features_rmse = (-np.mean(cross_val_score(knn, normal_train_mean[top_features], target, cv = kf, 
                                                 scoring = "neg_mean_squared_error")))**(1/2)

#zero dataframe
print("zero version error:", knn_zero_all_features_rmse, knn_zero_six_features_rmse)

#mean dataframe
print("mean version error:", knn_mean_all_features_rmse, knn_mean_six_features_rmse)
#Importing dummy regressor and cross_val_predict
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict

#model
dr = DummyRegressor()
kf = KFold(10, shuffle = True)

#predictions
dummy_predictions = cross_val_predict(dr, normal_train_zero, target, cv = kf)

#function to calculate the Weighted Mean Absolute Error
def wmae(predictions, correct_value, is_holiday_column):
    #size of the series/vector
    size = len(correct_value)

    #creating series object with weights set to 1
    weights = pd.Series(np.ones(size), index = correct_value.index).astype(int)

    #changing weights to 5 when it is holiday 
    weights = weights.mask(is_holiday_column == 1, 5)

    #error metric
    wmae_value = (np.abs(correct_value - predictions) * weights).sum() / weights.sum()
    
    return wmae_value

#calculating the error
dummy_wmae = wmae(dummy_predictions, target, normal_train_zero["is_holiday"])

print("dummy error:", dummy_wmae)
#importing libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

#function to select the best features by recursive feature elimination
def select_features(X_train, y_train):
    #recursive feature elimination in random forests, 10 fold, shuffling rows
    rfr = RandomForestRegressor(n_estimators = 10)
    kf = KFold(10, shuffle = True)
    
    #fitting recursive feature elimination model
    selector = RFECV(rfr, cv = kf)
    selector.fit(X_train,y_train)
    
    #best features
    best_columns = list(X_train.columns[selector.support_])
    
    return best_columns

#assigning features and target
X_train = normal_train_zero
y_train = target

#dictionary to hold 10 features selection by recursive feature elimination
dic_features = {}

for count in range(10):
    features = select_features(X_train, y_train)
    dic_features[count] = features
#resulting dictionary
dic_features = {
                    0: ['temperature', 'fuel_price', 'markdown3', 'cpi', 'unemployment', 'size', 'type_A', 'type_B', 
                        'is_strong_sales', 'week'],

                    1: ['cpi', 'unemployment', 'size', 'is_strong_sales', 'week'],

                    2: ['temperature', 'fuel_price', 'markdown3', 'markdown4', 'cpi', 'unemployment', 'size', 'type_A', 
                        'type_B', 'is_strong_sales', 'week', 'month'],

                    3: ['temperature', 'fuel_price', 'markdown3', 'markdown4', 'cpi', 'unemployment', 'size', 'type_A', 
                        'type_B', 'is_strong_sales', 'week', 'month'],

                    4: ['cpi', 'unemployment', 'size', 'is_strong_sales', 'week'],

                    5: ['cpi', 'unemployment', 'size', 'type_A', 'is_strong_sales', 'week'],

                    6: ['temperature', 'fuel_price', 'markdown3', 'markdown4', 'cpi', 'unemployment', 'size', 'type_A', 
                        'type_B', 'type_C', 'is_strong_sales', 'week', 'month'],

                    7: ['cpi', 'unemployment', 'size', 'is_strong_sales', 'week'],

                    8: ['cpi', 'unemployment', 'size', 'is_strong_sales', 'week'],

                    9: ['temperature', 'fuel_price', 'markdown1', 'markdown3', 'markdown4', 'markdown5', 'cpi', 
                        'unemployment', 'size', 'type_A', 'type_B', 'type_C', 'is_strong_sales', 'week', 'month']
                }

# one big list to hold features off all runs
features_list = []

for run in dic_features:
    #adding each run to the list
    features_list += dic_features[run]

#counting how often each feature occured
pd.Series(features_list).value_counts()
#selected features
features_list = ['temperature', 'fuel_price', 'markdown3', 'markdown4', 'cpi', 'unemployment', 'size', 'type_A', 
                 'type_B', 'is_strong_sales', 'week', 'month']

#importing grid search for model tuning
from sklearn.model_selection import GridSearchCV

#importing again to be able to run this cell even if feature selection cell was not run
from sklearn.ensemble import RandomForestRegressor

#assigning again to be able to run this cell even if feature selection cell was not run
X_train = normal_train_zero
y_train = target

#function to select the best parameters of Random Forests
def select_hyperparams(features_columns, X_train, y_train):
    
    hyperparameters = {
                        "n_estimators": [10],
                        "max_depth": [None, 8, 13, 18],
                        "min_samples_leaf": [1, 4],
                        "min_samples_split": [2, 4, 5, 6, 7]
                      }
    
    rfr = RandomForestRegressor(n_jobs = 4)
    kf = KFold(10, shuffle = True)

    grid = GridSearchCV(rfr, param_grid = hyperparameters, cv = kf)
    grid.fit(X_train[features_columns], y_train)
    best_params = grid.best_params_
    best_score = grid.best_score_
    return best_params, best_score

#dictionaries to hold 10 hyperparameters selection and scores
dic_params = {}
dic_score = {}

for count in range(10):
    best_params, best_score = select_hyperparams(features_list, X_train, y_train)
    dic_params[count] = best_params
    dic_score[count] = best_score
dic_params = {
                0: {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10},
                1: {'max_depth': 18, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10},
                2: {'max_depth': 18, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10},
                3: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 10},
                4: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10},
                5: {'max_depth': 18, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 10},
                6: {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 10},
                7: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 10},
                8: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 10},
                9: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 10}
             }
#importing again to be able to run this cell even if previous cells were not run
from sklearn.ensemble import RandomForestRegressor

#assigning again to be able to run this cell even if previous cells were not run
X_train = normal_train_zero
y_train = target

rfr = RandomForestRegressor(n_estimators = 100, max_depth = None, min_samples_leaf = 1, min_samples_split = 4)
kf = KFold(10, shuffle = True)

rfr_predictions = cross_val_predict(rfr, X_train[features_list], y_train, cv = kf)
wmae(rfr_predictions, y_train, X_train["is_holiday"])
# creating train dataframe with week column
reference_train = train.copy()
reference_train["week"] = reference_train["date"].dt.week

#pivot_table with weekly sales by store and by department
dept_reference_train = reference_train.pivot_table("weekly_sales", ["store", "week"], "dept", aggfunc = np.sum, fill_value = 0)

#weekly sales by store
store_weekly_sales = dept_reference_train.sum(axis = 1)

#finding how representative each department is within a store at a certain week of the year
proportion_sales = dept_reference_train.div(store_weekly_sales, axis = 0)

proportion_sales.head()
#function that returns the proportion of each department of each store
def proportion_by_dept(row):
    
    return proportion_sales.loc[(row["store"], row["week"]), row["dept"]]

#copying the train dataframe 
train_predictions = train.copy()

#creating the proportion column
train_predictions["proportion"] = reference_train.apply(proportion_by_dept, axis = 1)
train_predictions.head()
# adding predictions column to the grouped train dataframe
grouped_train_predictions = pd.concat([grouped_train, pd.Series(rfr_predictions)], axis = 1)
grouped_train_predictions = grouped_train_predictions.rename(columns = {0:"store_predictions"})
grouped_train_predictions.head()
# adding store predictions column to the train dataframe
train_predictions = pd.merge(train_predictions, grouped_train_predictions[["store", "date", "store_predictions"]],
                             on = ["store", "date"], how = "left")

# predict department sales based on the relevance of the deparment in each store
train_predictions["predicted_department_sales"] = train_predictions["proportion"] * train_predictions["store_predictions"]
train_predictions.head()
#weighted error of the department sales predictions
train_wmae = wmae(train_predictions["predicted_department_sales"], train_predictions["weekly_sales"], 
                  train_predictions["is_holiday"])

train_wmae
#Dummy model
dr = DummyRegressor()
kf = KFold(10, shuffle = True)

#predictions
dummy_predictions = cross_val_predict(dr, train["is_holiday"], train["weekly_sales"], cv = kf)

#weighted error of the department sales using the dummy model
dummy_wmae = wmae(dummy_predictions, train["weekly_sales"], train["is_holiday"])
dummy_wmae
def plot_results(axes_object, grouped_dataframe, store_id):
    
    #plot single store
    dataframe_mask = grouped_dataframe["store"] == store_id

    hue = None #only one line will be plotted

    title = "Store {} - prediction vs real data".format(store_id)

    #plotting target data
    sns.lineplot(grouped_dataframe[dataframe_mask]["date"], grouped_dataframe[dataframe_mask]["weekly_sales"], hue = hue,
                 ax = axes_object, color = "blue")
    
    #plotting random forests predictions
    sns.lineplot(grouped_dataframe[dataframe_mask]["date"], grouped_dataframe[dataframe_mask]["store_predictions"], hue = hue,
                 ax = axes_object, color = "green")

    #setup
    axes_object.set_title(title)
    axes_object.set_xlabel("Date")
    axes_object.set_ylabel("Sales value")
    axes_object.legend(["Weekly Sales", "Predicted Value"])

    #rotating the ticks
    plt.setp(axes_object.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()

def create_dashboard(rows, columns, store_list, grouped_dataframe):
    
    #creating dashboard with rows x columns graphs
    fig, axs = plt.subplots(rows, columns, figsize = (18, 18))

    #plotting one graph for each store
    for count, store_id in enumerate(store_list):
        row = int(count / columns)
        column = count % columns
        plot_results(axs[row, column], grouped_dataframe, store_id)

    plt.show()

#creating a 3x3 dashboard with stores from 1 to 9
create_dashboard(3, 3, range(1,10), grouped_train_predictions)
#creating dataframe with all features
test_zero = pd.merge(grouped_test, features_zero[feature_df_cols], how = "left", on = ["store", "date"])

#Converting boolean to zero or one
test_zero["is_holiday"] = test_zero["is_holiday"].astype(int)

#adding time related columns and dummy columns to dataframes
test_zero = add_time_columns(test_zero)
test_zero = add_dummy('type', test_zero)

#creating new column "is_strong_sales"
test_zero["is_strong_sales"] = 0
test_zero["is_strong_sales"] = test_zero["is_strong_sales"].mask(test_zero["week"].isin(strong_weeks), 1)

test_zero.head()
#normalizing numerical columns
normal_test = normalize(numerical_cols, test_zero)
normal_test
#assigning again to be able to run this cell even if previous cells were not run
X_train = normal_train_zero
y_train = target
X_test = normal_test

#model with the optmized hyperparameters 
rfr_test = RandomForestRegressor(n_estimators = 100, max_depth = None, min_samples_leaf = 1, min_samples_split = 4)

#fitting the model with our selected features 
rfr_test.fit(X_train[features_list], y_train)

#predicting the test file with our selected features
rfr_test_predictions = rfr_test.predict(X_test[features_list])
rfr_test_predictions
#test file with predictions
test_predictions = test.copy()

#adding week column
test_predictions["week"] = test_predictions["date"].dt.week

#creating the proportion column
test_predictions["proportion"] = test_predictions.apply(proportion_by_dept, axis = 1)

# adding predictions column to the grouped test dataframe
grouped_test_predictions = pd.concat([grouped_test, pd.Series(rfr_test_predictions)], axis = 1)
grouped_test_predictions = grouped_test_predictions.rename(columns = {0:"store_predictions"})

# adding store predictions column to the test dataframe
test_predictions = pd.merge(test_predictions, grouped_test_predictions[["store", "date", "store_predictions"]],
                             on = ["store", "date"], how = "left")

# predict department sales based on the relevance of the deparment in each store
test_predictions["predicted_department_sales"] = test_predictions["proportion"] * test_predictions["store_predictions"]
test_predictions.head()
#creating Id column
test_predictions[['store', 'dept', 'date']] = test_predictions[['store', 'dept', 'date']].astype(str)
test_predictions['Id'] = test_predictions[['store', 'dept', 'date']].agg('_'.join, axis=1)

#creating submission file
my_sample = test_predictions[["Id", "predicted_department_sales"]].copy()
my_sample = my_sample.rename(columns = {"predicted_department_sales" : "Weekly_Sales"})
#saving csv file
my_sample.to_csv("submission.csv", index = False)
my_sample.head()
