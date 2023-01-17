import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
train_df.set_index("Id", inplace=True)

train_df['Date'] =  pd.to_datetime(train_df['Date'])
train_df =  train_df.loc[train_df["Date"] <= "2020-04-02", :]
last_date = train_df["Date"].max()
print("Data from {} to {}.\n".format(train_df["Date"].min(), last_date))

categorical_cols = ["Province_State", "Country_Region"]
train_df.loc[:, categorical_cols] = train_df.loc[:, categorical_cols].astype("category")

train_df["CaseFatalityRate"] = train_df["Fatalities"] / train_df["ConfirmedCases"]

int_cols = ["ConfirmedCases", "Fatalities"]
train_df.loc[:, int_cols] = train_df.loc[:, int_cols].astype("int")

print(train_df.info())
train_df
train_df.loc[(train_df["Country_Region"] == "China") & (train_df["Date"] == last_date), :]
train_df.loc[(train_df["Country_Region"] == "France") & (train_df["Date"] == last_date), :]
train_df.loc[(train_df["Country_Region"] == "Italy") & (train_df["Date"] == last_date), :]
train_df.loc[(train_df["Country_Region"] == "Spain") & (train_df["Date"] == last_date), :]
countries = ["Italy", "China", "Spain", "France", "Korea, South", "Japan", "Germany", "US", 
             "Switzerland", "Iran", "United Kingdom", "Netherlands", "Austria", "Belgium", 
             "Norway", "Portugal", "Canada", "Brazil", "Israel"]
features = ["Province_State", "Country_Region", "Date", "ConfirmedCases", "Fatalities"]

df = train_df.loc[train_df["Country_Region"].isin(countries), [xx for xx in features]]
df["Country_Region"] = df["Country_Region"].cat.remove_unused_categories()
df = df.groupby(["Country_Region", "Date"])["ConfirmedCases", "Fatalities"].sum()
df["CaseFatalityRate"] = df["Fatalities"] / df["ConfirmedCases"]

print(df.info())
df
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

df["NewConfirmedCases"] = df["ConfirmedCases"].diff()
df["NewConfirmedCases"].fillna(0, inplace=True)
df.loc[df["NewConfirmedCases"] < 0, "NewConfirmedCases"] = 0
df["NewFatalities"] = df["Fatalities"].diff()
df["NewFatalities"].fillna(0, inplace=True)
df.loc[df["NewFatalities"] < 0, "NewFatalities"] = 0

fig, axes = plt.subplots(nrows=len(countries)*2, ncols=2, figsize=(18, len(countries) * 6 * 2))
for i, c in enumerate(countries):
    df.loc[c, ["ConfirmedCases", "Fatalities"]].plot(ax=axes[2*i, 0], grid=True)
    axes[2*i, 0].set_title(c)
    df.loc[c, ["CaseFatalityRate"]].plot(ax=axes[2*i, 1], grid=True, color="red")
    axes[2*i, 1].set_title(c)
    axes[2*i, 1].set_yticklabels(['{:,.2%}'.format(x) for x in axes[2*i, 1].get_yticks()]) #y axis with percentage
    df.loc[c, ["NewConfirmedCases"]].plot(ax=axes[2*i+1, 0], grid=True, color="purple")
    axes[2*i+1, 0].set_title(c)
    df.loc[c, ["NewFatalities"]].plot(ax=axes[2*i+1, 1], grid=True, color="green")
    axes[2*i+1, 1].set_title(c)
plt.tight_layout()
plt.show()

df.drop(["NewConfirmedCases", "NewFatalities"], axis=1, inplace=True)
print("Done in {:.4f} s.".format(time.time() - t0))
t0 = time.time()
for feature in ["ConfirmedCases", "Fatalities", "CaseFatalityRate"]:
    plt.figure(figsize=(14, 6))
    for c in ["Italy", "China", "Spain", "France"]:
        df.loc[c, feature].plot(label=c, grid=True)
    plt.title(feature)
    if feature == "CaseFatalityRate":
        plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [3.4, 4.2, 3.0, 6.5, 11.5, 13.4, 8.3, 2.9, 4.7, 
          1.5, 2.8, 4.7, 7.6, 6.2, 3.9, 3.4, 2.7, 2.2, 3.1]
hospital_beds = pd.DataFrame(values, index=countries, columns=["hospital_beds_per_1k"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
hospital_beds.sort_values("hospital_beds_per_1k", ascending=True).plot(kind="barh", rot=0, 
                                                                       grid=True, ax=axes[0])
hospital_beds.plot(y="hospital_beds_per_1k", kind="pie", legend=False, 
                   fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [45.1, 37.1, 42.3, 41.2, 41.2, 46.9, 46.8, 37.9, 42.2, 
          29.4, 40.5, 42.5, 43.8, 41.4, 39.1, 41.8, 42.0, 31.6, 29.7]
median_age = pd.DataFrame(values, index=countries, columns=["MedianAge"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
median_age.sort_values("MedianAge", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
median_age.plot(y="MedianAge", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [83.5, 76.9, 83.6, 82.7, 83.0, 84.6, 81.3, 78.9, 83.8, 76.7, 
          81.3, 82.3, 81.5, 81.6, 82.4, 82.0, 82.4, 75.9, 83.0]
life_expectancy = pd.DataFrame(values, index=countries, columns=["LifeExpectancy"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
life_expectancy.sort_values("LifeExpectancy", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
life_expectancy.plot(y="LifeExpectancy", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [66.56, 64.18, 78.42, 78.34, 83.59, 80.48, 73.58, 69.23, 73.23, 51.86, 
          74.88, 75.63, 79.46, 78.3, 74.36, 71.64, 71.27, 55.59, 72.97]
health_care_index = pd.DataFrame(values, index=countries, columns=["HealthCareIndex"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
health_care_index.sort_values("HealthCareIndex", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
health_care_index.plot(y="HealthCareIndex", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [54.97, 81.24, 39.16, 42.91, 54.8, 36.78, 28.42, 35.74, 21.31, 
          78.03, 40.63, 27.34, 21.78, 50.48, 20.29, 30.65, 27.66, 56.1, 56.93]
pollution_index = pd.DataFrame(values, index=countries, columns=["PollutionIndex"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
pollution_index.sort_values("PollutionIndex", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
pollution_index.plot(y="PollutionIndex", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [1493.3, 2043.0, 1499.0, 1089.9, 1667.4, 1583.2, 1599.5, 1016.6, 1489.8, 936.5, 827.7, 
          1459.9, 1927.0, 2440.9, 552.8, 1133.4, 1021.3, 333.5, 1280.7]
cigarette_consumption = pd.DataFrame(values, index=countries, columns=["CigaretteConsumption"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
cigarette_consumption.sort_values("CigaretteConsumption", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
cigarette_consumption.plot(y="CigaretteConsumption", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [24.98, 78.92, 21.99, 22.21, 24.94, 29.36, 14.60, 25.69, 10.52, 
          64.41, 22.23, 8.63, 15.33, 12.07, 7.82, 12.63, 15.69, 32.79, 30.80]
press_freedom_index = pd.DataFrame(values, index=countries, columns=["PressFreedomIndex"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
press_freedom_index.sort_values("PressFreedomIndex", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
press_freedom_index.plot(y="PressFreedomIndex", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [7.52, 2.26, 8.29, 8.12, 8.00, 7.99, 8.68, 7.96, 9.03, 2.38, 
          8.52, 9.01, 8.29, 7.64, 9.87, 8.03, 9.22, 6.86, 7.86]
democracy_index = pd.DataFrame(values, index=countries, columns=["DemocracyIndex"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
democracy_index.sort_values("DemocracyIndex", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
democracy_index.plot(y="DemocracyIndex", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [205, 148, 94, 122, 530, 347, 237, 36, 216, 50, 
          275, 511, 107, 377, 15, 112, 4, 25, 411]
population_density = pd.DataFrame(values, index=countries, columns=["PopulationDensity"])

t0 = time.time()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7.5))
population_density.sort_values("PopulationDensity", ascending=True).plot(kind="barh", grid=True, ax=axes[0])
population_density.plot(y="PopulationDensity", kind="pie", legend=False, fontsize=12, ax=axes[1])
plt.show()
print("Done in {:.4f} s.".format(time.time() - t0))
values = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]
wearing_masks = pd.DataFrame(values, index=countries, columns=["WearingMasks"], dtype=float)

wearing_masks
df1 = df.reset_index(level=[0]).loc[last_date, :].set_index("Country_Region")

df1 = df1.merge(hospital_beds, left_index=True, right_index=True)
df1 = df1.merge(median_age, left_index=True, right_index=True)
df1 = df1.merge(life_expectancy, left_index=True, right_index=True)
df1 = df1.merge(health_care_index, left_index=True, right_index=True)
df1 = df1.merge(pollution_index, left_index=True, right_index=True)
df1 = df1.merge(cigarette_consumption, left_index=True, right_index=True)
df1 = df1.merge(press_freedom_index, left_index=True, right_index=True)
df1 = df1.merge(democracy_index, left_index=True, right_index=True)
df1 = df1.merge(population_density, left_index=True, right_index=True)
df1 = df1.merge(wearing_masks, left_index=True, right_index=True)

df1
# the first date when at least 100 confirmed cases are announced 
date_exceed_100_confirmed_cases = df.loc[df.loc[:, "ConfirmedCases"] >= 100, :]\
                                  .reset_index("Date").groupby("Country_Region")["Date"].min()\
                                  .rename("DateExceed100ConfirmedCases")
# the first date when at least 5 fatalities are announced 
date_exceed_5_fatalities = df.loc[df.loc[:, "Fatalities"] >= 5, :]\
                             .reset_index("Date").groupby("Country_Region")["Date"].min()\
                             .rename("DateExceed5Fatalities") 

tmp = pd.merge(date_exceed_100_confirmed_cases, date_exceed_5_fatalities, 
               left_index=True, right_index=True, how="outer")
tmp["ConfirmedCasesGrowthRate"] = df1["ConfirmedCases"] / \
((last_date - tmp["DateExceed100ConfirmedCases"]) / np.timedelta64(1, 'D'))
tmp["FatalitiesGrowthRate"] = df1["Fatalities"] / \
((last_date - tmp["DateExceed5Fatalities"]) / np.timedelta64(1, 'D'))

tmp["FatalitiesGrowthRate"].fillna(0, inplace=True)
tmp
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 7.5))
tmp[["ConfirmedCasesGrowthRate"]].sort_values("ConfirmedCasesGrowthRate", ascending=True).plot(kind="barh", rot=0, 
                                                                       grid=True, ax=axes)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 7.5))
tmp[["FatalitiesGrowthRate"]].sort_values("FatalitiesGrowthRate", ascending=True).plot(kind="barh", rot=0, 
                                                                       grid=True, ax=axes)
plt.show()
df1 = df1.merge(tmp.loc[:, ["ConfirmedCasesGrowthRate", "FatalitiesGrowthRate"]] , left_index=True, right_index=True)
df1
cases = df1["ConfirmedCases"]
fatalities = df1["Fatalities"]
case_fatality_rate = df1["CaseFatalityRate"]
cases_growth_rate = df1["ConfirmedCasesGrowthRate"]
fatalities_growth_rate = df1["FatalitiesGrowthRate"]
df1.drop(["ConfirmedCases", "Fatalities", "CaseFatalityRate", 
          "ConfirmedCasesGrowthRate", "FatalitiesGrowthRate"], axis=1, inplace=True)
df1
scaler = MinMaxScaler() 
df1 = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns, index=df1.index)
df1
def build_vif(df):
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["vif"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif.set_index("feature", inplace=True)
    return vif

def build_leverage(df, visualize=True, xticks_labels=None, rotation=85, 
                   title=None, augment_const=True):
    """
    If constant columns has not already been augmented, use augment_const=True, 
    otherwise use augment_const=False. 
    """
    if augment_const:
        X = df.assign(const=1)
    else:
        X = df
    regression_results = sm.OLS(pd.DataFrame([[1]], index=X.index), X).fit()
    leverage = regression_results.get_influence().hat_matrix_diag
    if visualize:
        plt.plot(leverage)
        plt.grid()
        if title is not None:
            plt.title("leverage - " + title)
        if xticks_labels is not None:
            plt.xticks(range(len(countries)), countries, rotation=rotation)
        plt.show()
    return leverage
build_vif(df1)
title = "df1"
leverage = build_leverage(df1, xticks_labels=countries, title=title)

leverage = pd.DataFrame(leverage, columns=["leverage"], index=countries)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7.5))
leverage.sort_values("leverage", ascending=True).plot(kind="barh", grid=True, ax=axes)
plt.title("leverage - " + title)
plt.show()
build_vif(df1.loc[:, ["MedianAge", "LifeExpectancy"]])
df2 = df1.drop(["HealthCareIndex", "LifeExpectancy", "DemocracyIndex", "PollutionIndex"], axis=1)
build_vif(df2)
title = "df2"
leverage = build_leverage(df2, xticks_labels=countries, title=title)

leverage = pd.DataFrame(leverage, columns=["leverage"], index=countries)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7.5))
leverage.sort_values("leverage", ascending=True).plot(kind="barh", grid=True, ax=axes)
plt.title("leverage - " + title)
plt.show()
df3 = df1.loc[:, ["hospital_beds_per_1k", "MedianAge", "WearingMasks"]]
build_vif(df3)
title = "df3"
leverage = build_leverage(df3, xticks_labels=countries, title=title)

leverage = pd.DataFrame(leverage, columns=["leverage"], index=countries)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7.5))
leverage.sort_values("leverage", ascending=True).plot(kind="barh", grid=True, ax=axes)
plt.title("leverage - " + title)
plt.show()
# control group 
np.random.seed(42)
df4 = pd.DataFrame({"random_col_1": np.random.rand(df1.shape[0]), 
                    "random_col_2": np.random.rand(df1.shape[0]),
                    "random_col_3": np.random.rand(df1.shape[0])}, index=df2.index) 
scaler = MinMaxScaler() 
df4 = pd.DataFrame(scaler.fit_transform(df4), columns=df4.columns, index=df4.index)

build_vif(df4)
title = "df4"
leverage = build_leverage(df4, xticks_labels=countries, title=title)

leverage = pd.DataFrame(leverage, columns=["leverage"], index=countries)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 7.5))
leverage.sort_values("leverage", ascending=True).plot(kind="barh", grid=True, ax=axes)
plt.title("leverage - " + title)
plt.show()
def do_regression(df, y):
    title_ = "leverage - " + y
    if y == "ConfirmedCases":
        y = cases
    elif y == "Fatalities":
        y = fatalities
    elif y == "CaseFatalityRate":
        y = case_fatality_rate
    elif y == "ConfirmedCasesGrowthRate":
        y = cases_growth_rate
    elif y == "FatalitiesGrowthRate":
        y = fatalities_growth_rate
    else:
        raise NotImplementedError
    regression_results = sm.OLS(y, df.assign(const=1)).fit()
    print(regression_results.summary())
do_regression(df1, "ConfirmedCases")
do_regression(df2, "ConfirmedCases")
do_regression(df3, "ConfirmedCases")
do_regression(df4, "ConfirmedCases")
do_regression(df1, "Fatalities")
do_regression(df2, "Fatalities")
do_regression(df3, "Fatalities")
do_regression(df4, "Fatalities")
do_regression(df1, "CaseFatalityRate")
do_regression(df2, "CaseFatalityRate")
do_regression(df3, "CaseFatalityRate")
do_regression(df4, "CaseFatalityRate")
do_regression(df1, "ConfirmedCasesGrowthRate")
do_regression(df2, "ConfirmedCasesGrowthRate")
do_regression(df3, "ConfirmedCasesGrowthRate")
do_regression(df4, "ConfirmedCasesGrowthRate")
do_regression(df1, "FatalitiesGrowthRate")
do_regression(df2, "FatalitiesGrowthRate")
do_regression(df3, "FatalitiesGrowthRate")
do_regression(df4, "FatalitiesGrowthRate")