import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 10)
PATH = "../input/open-shopee-code-league-marketing-analytics/"
    
os.listdir(PATH)
train_df = pd.read_csv(f"{PATH}train.csv")
user_df = pd.read_csv(f"{PATH}users.csv")

test_df = pd.read_csv(f"{PATH}test.csv")
available_dfs = {
    "train.csv": train_df, 
    "test.csv": test_df, 
    "users.csv": user_df
}

for available_df in available_dfs.keys():
    columns = available_dfs[available_df].columns.tolist()
    
    print(f"{available_df} contains {len(columns)} columns")
    print(f"They are : {columns}")
    print("")
train_df = pd.merge(train_df, user_df, how="left", left_on="user_id", right_on="user_id")
test_df = pd.merge(test_df, user_df, how="left", left_on="user_id", right_on="user_id")
train_df.describe()
test_df.describe()
def get_summary_df(df):
    
    columns = df.columns.tolist()
    
    dtypes = []
    unique_counts = []
    missing_counts = []
    missing_percentages = []
    total_counts = [df.shape[0]] * len(columns)

    for column in columns:
        dtype = str(df[column].dtype)
        dtypes.append(dtype)
        
        unique_count = df[column].nunique()
        unique_counts.append(unique_count)

        missing_count = df[column].isnull().sum()
        missing_counts.append(missing_count)
        
        missing_percentage = round((missing_count/df.shape[0]) * 100, 2)
        missing_percentages.append(missing_percentage)
        


    summary_df = pd.DataFrame({
        "column": columns,
        "dtype": dtypes,
        "unique_count": unique_counts,
        "missing_count": missing_counts,
        "missing_percentage": missing_percentages,
        "total_count": total_counts,
    })
    
    summary_df = summary_df.sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)
    
    return summary_df
get_summary_df(train_df)
train_df["last_open_day"].unique()
test_df["last_open_day"].unique()
activities = ["login", "checkout", "open"]
for activity in activities:
    train_df[f"last_{activity}_day"] = train_df[f"last_{activity}_day"].apply(lambda x: np.nan if x == f"Never {activity}" else int(x))
    test_df[f"last_{activity}_day"] = test_df[f"last_{activity}_day"].apply(lambda x: np.nan if x == f"Never {activity}" else int(x))
last_activity_column_names = [f"last_{activity}_day" for activity in activities]
get_summary_df(train_df[last_activity_column_names])
get_summary_df(test_df[last_activity_column_names])
train_df[last_activity_column_names].describe()
test_df[last_activity_column_names].describe()
def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style("whitegrid")
    plt.figure
    
    fig, ax = plt.subplots(5, 4, figsize=(20, 20))
    
    for feature in features:
        i += 1
        plt.subplot(5, 4, i)
        plt.scatter(df1[feature], df2[feature], marker="+", color='#2B3A67', alpha=0.2)
        plt.xlabel(feature, fontsize=9)
        
    plt.show()
features = ["subject_line_length", "login_count_last_10_days", "login_count_last_30_days", "login_count_last_60_days", "open_count_last_10_days", "open_count_last_30_days", "open_count_last_60_days", "checkout_count_last_10_days", "checkout_count_last_30_days", "checkout_count_last_60_days", "age", "attr_1", "attr_2", "attr_3", "last_open_day", "last_checkout_day", "last_login_day"]

plot_feature_scatter(train_df.sample(50000), test_df.sample(50000), features)
def plot_feature_distribution(df1, df2, label1, label2, features, size=[4,5]):
    i = 0
    sns.set_style("whitegrid")
    
    plt.figure()
    fig, ax = plt.subplots(size[0], size[1], figsize=(22, 20))
    
    for feature in features:
        i += 1
        plt.subplot(size[0], size[1], i)
        
        sns.distplot(df1[feature], hist=False, label=label1, kde_kws={'bw':1.5}, color="#5982C5")
        sns.distplot(df2[feature], hist=False, label=label2, kde_kws={'bw':1.5}, color="#FB3523")
        
        plt.xlabel(feature, fontsize=9)
        
        locs, labels = plt.xticks()
        plt.tick_params(axis="x", which="major", labelsize=6, pad=6)
        plt.tick_params(axis="y", which="major", labelsize=6)
        
    plt.show()
target_0_df = train_df.loc[train_df["open_flag"] == 0]
target_1_df = train_df.loc[train_df["open_flag"] == 1]

plot_feature_distribution(target_0_df, target_1_df, '0', '1', features, size=[4,5])
plot_feature_distribution(train_df, test_df, 'train', 'test', features, size=[4,5])
train_df[train_df["last_login_day"] < 18000]["last_login_day"].max()
test_df[test_df["last_login_day"] < 18000]["last_login_day"].max()
train_df[train_df["last_login_day"] > 18000].sort_values(by="last_login_day").head(10)[["grass_date", "last_login_day"]]
test_df[test_df["last_login_day"] > 18000].sort_values(by="last_login_day").head(10)
train_df[train_df["age"] < 0]
test_df[test_df["age"] < 0]
train_df[train_df["age"] < 18].shape
test_df[test_df["age"] < 18].shape
train_df[train_df["age"] > 100]["age"].value_counts()
test_df[test_df["age"] > 100]["age"].value_counts()
plot_feature_distribution(train_df[(train_df["age"] == 50) & (train_df["open_flag"] == 0)], train_df[(train_df["age"] == 50) & (train_df["open_flag"] == 1)], "0", "1", features, size=[4,5])
plot_feature_distribution(train_df[(train_df["age"] != 50) & (train_df["open_flag"] == 0)], train_df[(train_df["age"] != 50) & (train_df["open_flag"] == 1)], "0", "1", features, size=[4,5])
def plot_feature_distribution_by_elems(df1, df2, label1, label2, elems, size=[2,4]):
    i = 0
    sns.set_style("whitegrid")
    
    ig, ax = plt.subplots(size[0], size[1], figsize=(20, 10))
    
    for i, elem in enumerate(elems):
        plt.subplot(size[0], size[1], i+1)

        sns.kdeplot(df1[df1["country_code"] == elem]["subject_line_length"], bw=0.5, label=label1, color="#5982C5")
        sns.kdeplot(df2[df2["country_code"] == elem]["subject_line_length"], bw=0.5, label=label2, color="#FB3523")

        plt.xlabel(elem, fontsize=11)
        locs, labels = plt.xticks()

        plt.tick_params(axis="x", which="major", labelsize=8)
        plt.tick_params(axis="y", which="major", labelsize=8)

    plt.show()
country_codes = sorted(train_df["country_code"].unique().tolist())

plot_feature_distribution_by_elems(target_0_df, target_1_df, "0", "1", country_codes, size=[2,4])
plot_feature_distribution_by_elems(train_df, test_df, "train", "test", country_codes, size=[2,4])