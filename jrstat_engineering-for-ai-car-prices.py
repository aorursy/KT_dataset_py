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
def count_values(df, colname, n_values=20, ascending=False):
    counts = df\
        [colname]\
        .astype(str)\
        .replace("nan", "unknown")\
        .value_counts()\
        .sort_values(ascending=ascending)
    
    plot = counts\
        .iloc[:n_values]\
        .plot\
        .bar(title=f"{colname} ({counts.shape[0]} unique values)")
    
    return plot

def hist_values(df, colname, bins=10):
    if df[colname].dtype != "object":
        plot = df.hist(column=colname, bins=bins)
        return plot
    

def plot_values(df, colname, n_values=20, top_n=True, bins=10):
    if df[colname].dtype == "float":
        return hist_values(df=df, colname=colname, bins=bins)
    
    else:
        ascending = not top_n
        return count_values(df=df, colname=colname, n_values=n_values, ascending=ascending)
    
# IMPORT DATASET

filename = "/kaggle/input/personal-cars-classifieds/all_anonymized_2015_11_2017_03.csv"
raw_df = pd.read_csv(filename)
raw_shape = raw_df.shape

print(f"Raw data has {raw_shape[0]} rows, and {raw_shape[1]} columns")
# PASS IN DATA TYPES WHEN READING
# - SPEED UP READ
# - AVOID READ ERRORS

raw_dtypes = {
    "maker": str,
    "model": str,
    "mileage": float,
    "manufacture_year": float, # np.NaN doesn't work with int
    "engine_displacement": float,
    "engine_power": float,
    "body_type": str,
    "color_slug": str,
    "stk_year": str, # None's cannot be converted by pandas here
    "transmission": str,
    "door_count": str,
    "seat_count": str,
    "fuel_type": str,
    "date_created": str,
    "date_last_seen": str,
    "price_eur": float}

raw_df = pd.read_csv(filename, dtype=raw_dtypes)
raw_shape = raw_df.shape


# AND CONVERT TYPES TO NUMERIC WHERE REQUIRED
# THIS WILL ALSO GET RID OF ANY UNEXPECTED TEXT IN THESE FIELDS (e.g. "None")
to_num = ["stk_year", "door_count", "seat_count"]
for col in to_num:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
    
print(f"Raw data has {raw_shape[0]} rows, and {raw_shape[1]} columns")
# SOME USEFUL METHODS TO START EXPLORING

print(raw_df.info(), "\n\n\n")
print(raw_df.describe(), "\n\n\n")
print(raw_df.isna().sum())
raw_df.head()

# EXPLORE EACH COLUMN
colname = raw_df.columns[0]
plot_values(raw_df, colname, n_values=20, top_n=True, bins=8)
# DEALING WITH NA's (CAN'T MODEL WITH THEM)

# CREATE EXISTS COLUMNS
raw_df["stk_bool"] = raw_df["stk_year"].notnull().astype(int)


# IMPUTE SOME NAs - manufacture_year
# - USE MAKE/MODEL TO ESTIMATE manufacture_year (IS THIS SENSIBLE)
year_of_scepticism = 1970
average_years = raw_df\
    .loc[raw_df["manufacture_year"] >= year_of_scepticism]\
    .groupby(["maker", "model"])\
    ["manufacture_year"]\
    .mean()\
    .round()\
    .rename("avg_manufacture_year")\
    .reset_index()


# REPLACE SCEPTICAL VALUES AND NAs
clean_df = raw_df.merge(average_years, how="left", on=["maker", "model"])
clean_df.loc[raw_df["manufacture_year"] < year_of_scepticism, "manufacture_year"] = np.nan
clean_df["manufacture_year"] = clean_df["manufacture_year"].fillna(clean_df["avg_manufacture_year"])


# DROP COLUMNS AND CREATE NEW DF
drop_cols = ["model", "body_type", "color_slug", "stk_year", "avg_manufacture_year"]
clean_df = clean_df.drop(drop_cols, axis="columns")


# DROP ALL ADVERTS WITH AT LEAST ONE NA
clean_df = clean_df.dropna()
clean_df
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# FIT BENCHMARK MODEL (JUST NUMERIC FEATURES)
features_initial = ["mileage", "manufacture_year", "engine_displacement", "engine_power", "door_count", "seat_count", "stk_bool"]
target = "price_eur"

X_initial = clean_df[features_initial]
y_initial = clean_df[target]


def fit_and_score(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    rmse = mean_squared_error(y_test, lr.predict(X_test))
    actual_predicted = pd.DataFrame({"predicted": lr.predict(X_test), "actual": y_test})
    
    return (score, actual_predicted)



score_initial = fit_and_score(X_initial, y_initial)
print("Initial model R Squared: %0.3f (%i observations)" % (score_initial[0], X_initial.shape[0]))
score_initial[1].plot.scatter(x="actual", y="predicted")
# PRICE
hist_values(clean_df, target, bins=100)
# hist_values(clean_df.loc[clean_df[target] < 100000], target, bins=100)
# hist_values(clean_df.assign(log=np.log1p(clean_df[target])), "log", bins=100)
# clean_df[clean_df[target] == 1295.34]

# MODEL ON LOG(PRICE)
X_logged = clean_df[features_initial]
y_logged = np.log1p(clean_df[target])


score_logged = fit_and_score(X_logged, y_logged)
print("Initial model R Squared: %0.3f (%i observations)" % (score_initial[0], X_initial.shape[0]))
print("Logged model R Squared: %0.3f (%i observations)" % (score_logged[0], X_logged.shape[0]))
# FILTER OUT UNUSUAL DATA
filtered_df = clean_df.copy()
filtered_df = filtered_df[filtered_df[target] != 1295.34]

X_filtered = filtered_df[features_initial]
y_filtered = np.log1p(filtered_df[target])

score_filtered = fit_and_score(X_filtered, y_filtered)
print("Initial model R Squared: %0.3f (%i observations)" % (score_initial[0], X_initial.shape[0]))
print("Logged model R Squared: %0.3f (%i observations)" % (score_logged[0], X_logged.shape[0]))
print("Filtered model R Squared: %0.3f (%i observations)" % (score_filtered[0], X_filtered.shape[0]))
# count_values(filtered_df, "maker", ascending=False)
count_values(filtered_df, "seat_count", ascending=False)
# CREATE MANUFACTURER CATEGORIES
luxurious_makers = [
    'bentley', 'bmw', 'chevrolet', 'dodge',
    'hummer', 'mercedes-benz', 'rolls-royce']

sports_makers = [
    'alfa-romeo', 'aston-martin', 'audi', 'jaguar',
    'lamborghini', 'lotus', 'maserati', 'porsche', 'tesla']

filtered_df["maker_type"] = np.select(
    condlist=[
        filtered_df["maker"].isin(luxurious_makers),
        filtered_df["maker"].isin(sports_makers)],
    choicelist=[
        "luxurious",
        "sports"],
    default="normal")

# CREATE CAR SIZE CATEGORIES
filtered_df["seat_str"] = np.select(
    condlist=[
        (filtered_df["seat_count"] >= 0) & (filtered_df["seat_count"] < 4),
        (filtered_df["seat_count"] >= 4) & (filtered_df["seat_count"] < 6),
        (filtered_df["seat_count"] >= 6) & (filtered_df["seat_count"] < 10),
        (filtered_df["seat_count"] >= 10)],
    choicelist=[
        "small",
        "medium",
        "large",
        "very large"],
    default="unknown")

filtered_df.head()
# CALCULATE TIME THE AD HAS BEEN POSTED FOR
filtered_df["date_created"] = pd.to_datetime(filtered_df["date_created"])
filtered_df["date_last_seen"] = pd.to_datetime(filtered_df["date_last_seen"])
filtered_df["post_duration"] = (filtered_df["date_last_seen"] - filtered_df["date_created"]).dt.total_seconds() / 86400
# ENCODING CATEGORICAL VARIABLES
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

str_cols = ["transmission", "fuel_type", "seat_str", "maker_type"]
model_df = pd.DataFrame(index=filtered_df.index)
for col in str_cols:
    ohe = OneHotEncoder()
    ohe.fit(filtered_df[[col]])
    col_df = pd.DataFrame(
        ohe.transform(filtered_df[[col]]).toarray(), 
        columns=ohe.get_feature_names([col]), 
        index=filtered_df.index)
    
    model_df = pd.concat([model_df, col_df], axis=1)
# ADD NUMERIC COLUMNS TO ENCODED COLUMNS
num_cols = ["manufacture_year", "mileage", "engine_displacement", "engine_power", "door_count", "post_duration", "stk_bool", target]
model_df[num_cols] = filtered_df[num_cols]
model_df.head()
X_engineered = model_df.drop(target, axis="columns")
y_engineered = np.log1p(model_df[target])

score_engineered = fit_and_score(X_engineered, y_engineered)
print("Initial model R Squared: %0.3f (%i observations)" % (score_initial[0], X_initial.shape[0]))
print("Logged model R Squared: %0.3f (%i observations)" % (score_logged[0], X_logged.shape[0]))
print("Filtered model R Squared: %0.3f (%i observations)" % (score_filtered[0], X_filtered.shape[0]))
print("Engineered model R Squared: %0.3f (%i observations)" % (score_engineered[0], X_engineered.shape[0]))
score_engineered[1].plot.scatter(x="actual", y="predicted")
# STILL A FEW OUTLIERS
# REMOVE ADVERTS THAT ARE OUTLIERS IN ANY COLUMN
from scipy import stats

final_df = model_df\
    .loc[:, lambda df: df.std() > 0.05]\
    .loc[lambda df: (np.abs(stats.zscore(df)) < 3).all(axis=1)]

X_final = final_df.drop(target, axis="columns")
y_final = np.log1p(final_df[target])

score_final = fit_and_score(X_final, y_final)
print("Initial model R Squared: %0.3f (%i observations)" % (score_initial[0], X_initial.shape[0]))
print("Logged model R Squared: %0.3f (%i observations)" % (score_logged[0], X_logged.shape[0]))
print("Filtered model R Squared: %0.3f (%i observations)" % (score_filtered[0], X_filtered.shape[0]))
print("Engineered model R Squared: %0.3f (%i observations)" % (score_engineered[0], X_engineered.shape[0]))
print("Final model R Squared: %0.3f (%i observations)" % (score_final[0], X_final.shape[0]))

score_final[1].plot.scatter(x="actual", y="predicted")
# COMPARE THE PERFORMANCE BY USING
# SAME DATA FOR EACH MODEL
index_final = final_df.index

X_initial1 = clean_df.loc[index_final, features_initial]
y_initial1 = clean_df.loc[index_final, target]

X_logged1 = clean_df.loc[index_final, features_initial]
y_logged1 = np.log1p(clean_df.loc[index_final, target])

X_filtered1 = filtered_df.loc[index_final, features_initial]
y_filtered1 = np.log1p(filtered_df.loc[index_final, target])

X_engineered1 = model_df.loc[index_final].drop(target, axis="columns")
y_engineered1 = np.log1p(model_df.loc[index_final, target])

score_initial1 = fit_and_score(X_initial1, y_initial1)
score_logged1 = fit_and_score(X_logged1, y_logged1)
score_filtered1 = fit_and_score(X_filtered1, y_filtered1)
score_engineered1 = fit_and_score(X_engineered1, y_engineered1)

print("All models with the same %i rows:\n" % X_final.shape[0])
print("Initial model R Squared: %0.3f (old: %0.3f)" % (score_initial1[0], score_initial[0]))
print("Logged model R Squared: %0.3f (old: %0.3f)" % (score_logged1[0], score_logged[0]))
print("Filtered model R Squared: %0.3f (old: %0.3f)" % (score_filtered1[0], score_filtered[0]))
print("Engineered model R Squared: %0.3f (old: %0.3f)" % (score_engineered1[0], score_engineered[0]))
print("Final model R Squared: %0.3f" % (score_final[0]))
score_engineered1[1].plot.scatter(x="actual", y="predicted")
