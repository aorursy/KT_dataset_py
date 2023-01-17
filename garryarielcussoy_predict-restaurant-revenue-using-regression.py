# Import some packages needed

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import linear_model

from datetime import datetime



# Read the train and test dataset

train_df = pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip")

test_df = pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip")



# Remove Id field from train dataset

train_df = train_df.drop(["Id"], axis = 1)



# Look at some train data examples

train_df.head(10)
# Turn Open Date field into datetime type

train_df["Open Date"] = train_df["Open Date"].astype('datetime64[ns]')



# Get competition date

competition_date = datetime.strptime('2015-09-14', '%Y-%m-%d')



# Convert Open Date field into integer

train_df["Dummy"] = competition_date

train_df["Open Date"] = train_df["Open Date"] - train_df["Dummy"]

train_df["Open Date"] = train_df["Open Date"].dt.days

train_df = train_df.drop(["Dummy"], axis = 1)

train_df["Open Date"] = train_df["Open Date"].abs()



# Look the Open Date converted field

train_df[["Open Date"]].head(10)
# Draw scatter plot of Open Date vs Revenue

plt.scatter(train_df["Open Date"], train_df["revenue"])

plt.xlabel("Open Date")

plt.ylabel("Revenue")

plt.show()
# Get index of those outliars

index_out = train_df[train_df['revenue'] > 12500000].index



# Remove outliar data from train dataset

train_df = train_df.drop(index_out)



# Copy train_df for later observations

for_corr_df = train_df



# To confirm, we now only have 134 rows

train_df
# Get the frequency of each unique value on City field

train_df[["City"]].value_counts()
# Turn City field into one-hot-encoding form

city_dummy_df = pd.get_dummies(train_df[["City"]], prefix = ['City'])



# Create new column titled City_Other

city_dummy_df["City_Other"] = 0

for index, rows in city_dummy_df.iterrows():

    if (

        rows["City_İstanbul"] == 0 and

        rows["City_Ankara"] == 0 and

        rows["City_İzmir"] == 0 and

        rows["City_Bursa"] == 0 and

        rows["City_Samsun"] == 0 and

        rows["City_Antalya"] == 0 and

        rows["City_Sakarya"] == 0

    ):

        city_dummy_df["City_Other"][index] = 1



# Choose only 8 features

city_dummy_df = city_dummy_df[["City_İstanbul", "City_Ankara", "City_İzmir", "City_Bursa", "City_Samsun", "City_Antalya", "City_Sakarya", "City_Other"]]



# Merge that one-hot-encoding dataframe to train_df

train_df = pd.merge(train_df, city_dummy_df, left_index = True, right_index = True)



# Look at the result

train_df.head(10)
# Draw a chart about the frequency of each value of City Group field

ax = sns.countplot(x = "City Group", data = train_df)
# Turn City Group field into one-hot-encoding form

group_dummy_df = pd.get_dummies(train_df[["City Group"]], prefix = ['Group'])



# Merge that one-hot-encoding dataframe to train_df

train_df = pd.merge(train_df, group_dummy_df, left_index = True, right_index = True)



# Look at the result

train_df.head(10)
# Draw a chart about the frequency of each value of Type field (from train dataset and test dataset)

plt.figure(1)

ax = sns.countplot(x = "Type", data = train_df)

plt.figure(2)

ax = sns.countplot(x = "Type", data = test_df)
# Get information about revenue's average in each type

rev_avg_df = train_df[["Type", "revenue"]].groupby("Type").mean()

type_freq_df = train_df[["Type", "revenue"]].groupby("Type").count()

rev_info_by_type_df = pd.merge(type_freq_df, rev_avg_df, on = "Type").sort_values(by = ['revenue_x'], ascending = False)

rev_info_by_type_df = rev_info_by_type_df.rename(columns = {"revenue_y": "Average Rev", "revenue_x": "Frequency"})

rev_info_by_type_df
# Replace all DT with IL

for index, rows in train_df.iterrows():

    if rows["Type"] == "DT":

        train_df["Type"][index] = "IL"



# Turn into one-hot-encoding form

type_dummy_df = pd.get_dummies(train_df[["Type"]], prefix = ['Type'])



# Merge that one-hot-encoding dataframe to train_df

train_df = pd.merge(train_df, type_dummy_df, left_index = True, right_index = True)



# Look at the result

train_df.head(10)
# Get sub dataframe of all numerical type fields

numerical_df = for_corr_df.drop(["City", "City Group", "Type"], axis = 1)



# Normalize the dataset

numerical_df_val = numerical_df.values # Returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

numerical_scaled = min_max_scaler.fit_transform(numerical_df_val)

normal_numerical_df = pd.DataFrame(numerical_scaled)



# Get correlation matrix

corr_df = normal_numerical_df.corr()

corr_df = corr_df.abs()



# Draw correlation heatmap for all numerical type fields

plt.figure(figsize = (39, 39))

ax = sns.heatmap(corr_df)
# Draw scatter plot for each feature vs revenue

features_index = [2, 28]

counter = 1

for index in features_index:

    plt.figure(counter)

    x_field_name = "P" + str(index)

    plt.scatter(train_df[x_field_name], train_df["revenue"])

    plt.xlabel(x_field_name)

    plt.ylabel("Revenue")

    counter += 1



# Draw the plot in a frame

plt.show()
# Feature selection

to_drop = []

for index in list(range(1, 38)):

    if index not in features_index:

        to_drop.append("P" + str(index))

train_df = train_df.drop(to_drop, axis = 1)



# Look at the result

train_df
# Create regresson object and prepare the dataset

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train_df.drop(['revenue', 'City', 'City Group', 'Type'], axis = 1))

train_y = np.asanyarray(train_df[['revenue']])



# Feed the data into the model

regr.fit(train_x, train_y)
# --------------- Convert Open Date ---------------

# Turn Open Date field into datetime type

test_df["Open Date"] = test_df["Open Date"].astype('datetime64[ns]')



# Convert Open Date field into integer

test_df["Dummy"] = competition_date

test_df["Open Date"] = test_df["Open Date"] - test_df["Dummy"]

test_df["Open Date"] = test_df["Open Date"].dt.days

test_df = test_df.drop(["Dummy"], axis = 1)

test_df["Open Date"] = test_df["Open Date"].abs()



# Look the Open Date converted field

test_df[["Open Date"]].head(10)
# --------------- Convert City ---------------

# Turn City field into one-hot-encoding form

city_dummy_df = pd.get_dummies(test_df[["City"]], prefix = ['City'])



# Create new column titled City_Other

city_dummy_df["City_Other"] = 0

for index, rows in city_dummy_df.iterrows():

    if (

        rows["City_İstanbul"] == 0 and

        rows["City_Ankara"] == 0 and

        rows["City_İzmir"] == 0 and

        rows["City_Bursa"] == 0 and

        rows["City_Samsun"] == 0 and

        rows["City_Antalya"] == 0 and

        rows["City_Sakarya"] == 0

    ):

        city_dummy_df["City_Other"][index] = 1



# Choose only 8 features

city_dummy_df = city_dummy_df[["City_İstanbul", "City_Ankara", "City_İzmir", "City_Bursa", "City_Samsun", "City_Antalya", "City_Sakarya", "City_Other"]]



# Merge that one-hot-encoding dataframe to train_df

test_df = pd.merge(test_df, city_dummy_df, left_index = True, right_index = True)



# Look at the result

test_df.head(10)
# --------------- Convert City Group ---------------

# Turn City Group field into one-hot-encoding form

group_dummy_df = pd.get_dummies(test_df[["City Group"]], prefix = ['Group'])



# Merge that one-hot-encoding dataframe to train_df

test_df = pd.merge(test_df, group_dummy_df, left_index = True, right_index = True)



# Look at the result

test_df.head(10)
# --------------- Convert Type ---------------

# Replace all DT with IL

for index, rows in test_df.iterrows():

    if rows["Type"] == "DT":

        test_df["Type"][index] = "IL"

    elif rows["Type"] == "MB":

        test_df["Type"][index] = "FC"



# Turn into one-hot-encoding form

type_dummy_df = pd.get_dummies(test_df[["Type"]], prefix = ['Type'])



# Merge that one-hot-encoding dataframe to train_df

test_df = pd.merge(test_df, type_dummy_df, left_index = True, right_index = True)



# Look at the result

test_df.head(10)
# --------------- Feature Selection ---------------

to_drop = []

for index in list(range(1, 38)):

    if index not in features_index:

        to_drop.append("P" + str(index))

test_df = test_df.drop(to_drop, axis = 1)



# Look at the result

test_df.head(10)
# Prepare the dataset

test_x = np.asanyarray(test_df.drop(['Id', 'City', 'City Group', 'Type'], axis = 1))
# Predict restaurant revenue

y_predict = regr.predict(test_x)



# Save into csv format

test_df["Prediction"] = y_predict

submit_df = test_df[["Id", "Prediction"]]

submit_df.to_csv("submission.csv", index = False)