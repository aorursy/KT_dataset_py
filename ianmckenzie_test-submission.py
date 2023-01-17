import pandas as pd

import numpy as np

import sklearn as sk

import matplotlib.pyplot as plt

import datetime as dt
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
df = train

df.info()
countries = df["Country_Region"].unique()

print(f"There are {len(countries)} countries in the dataset")

df[pd.notna(df["Province_State"])]["Country_Region"].unique()
df[df["Country_Region"] == "United Kingdom"]["Province_State"].unique()
df[df["Country_Region"] == "US"]["Province_State"].unique()
test = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

sub = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
# function to add an underscore to non-na values and to replace nas with empty trying

def process_province(x):

    if not pd.isna(x):

        x = "_" + x

    else:

        x = ''

    return x



df["Province_State"] = df["Province_State"].apply(process_province)

df["Region"] = df["Country_Region"] + df["Province_State"]

# saving unique regions for later

regions = df["Region"].unique()

df = df.drop(["Province_State", "Country_Region"], axis=1)
df.head()
lockdown = pd.read_csv("../input/covid19-lockdown-dates-by-country/countryLockdowndates.csv")

lockdown.head()
# dropping reference and type columns

lockdown = lockdown.drop(["Type", "Reference"], axis=1)
lockdown["Province"] = lockdown["Province"].apply(process_province)

lockdown["Region"] = lockdown["Country/Region"] + lockdown["Province"]

lockdown = lockdown.drop(["Province", "Country/Region"], axis=1)
# preparing the date column to be readable

# lockdownlockdown["Date"].fillna()

lockdown["Date"] = pd.to_datetime(lockdown["Date"], dayfirst=True)

lockdown = lockdown.rename({"Region": "Region","Date": "Lockdown_Date"}, axis=1)

lockdown = lockdown.set_index("Region")
df = df.assign(Lockdown_Date = '')
df = df.set_index("Region")
df.update(lockdown)
df["Lockdown_Date"] = pd.to_datetime(df["Lockdown_Date"])

df["Date"] = pd.to_datetime(df["Date"])
df = df.assign(Two_Weeks_Lockdown = (df["Date"] - df["Lockdown_Date"] >= dt.timedelta(days=14)).astype(int))

df = df.drop("Lockdown_Date", axis=1)



# after all this mess we reset the index

df = df.reset_index()

df = df.set_index("Id")
df["Two_Weeks_Lockdown"].value_counts()

# seems to be working as italy entered lockdown 2020-3-11

df[df["Region"] == "Italy"][50:]
# Establishing the number of 'lags' to keep track of 

P = 10
def lag_dataset(df_orig, y, p=P):

    """

    Function to lag this specific dataset, doesn't work in general.

    INPUT

    df_orig : the pandas dataframe of cases etc. (pd.DataFrame)

    y : which column we are interested in (str)

    p : how many shifted columns to make (int)

    OUTPUT

    lagged_df : the lagged dataframe (pd.DataFrame)

    """

    df = df_orig.copy()

    df = df.reset_index().set_index(["Id","Date"])

    # we have to treat each region individually

    types = df["Region"].unique()

    lagged_df = pd.DataFrame()

    

    for t in types:

        values = pd.DataFrame(df[df["Region"]==t][y])

        lagged_y = pd.concat([values.shift(s) for s in range(p+1)], axis=1)

        lagged_y.columns = ['t']+['t-'+str(s) for s in range(1,p+1)]

        lagged_y = lagged_y[p:]

        lagged_y = lagged_y.assign(Region=t)

        lagged_df = pd.concat([lagged_df, lagged_y])

    # merging in stages

    lagged_df = lagged_df.reset_index()

    lagged_df = lagged_df.merge(df.drop(["ConfirmedCases", "Fatalities"], axis=1), left_on=["Region","Date"], right_on=["Region", "Date"],suffixes=(None, None))

    lagged_df = lagged_df.set_index(["Id"])

    # done merging

    return lagged_df

fatalities_lagged = lag_dataset(df, "Fatalities", p=P)

cases_lagged = lag_dataset(df, "ConfirmedCases", p=P)

fatalities_lagged[fatalities_lagged["Region"] == "Italy"]
# now we one-hot encode the "Region" column, which will give us 293 new columns. Yay.

fatalities_df = pd.get_dummies(fatalities_lagged, "Region")

cases_df = pd.get_dummies(cases_lagged, "Region")

fatalities_df.info()

fatalities_df.head()

# rmsle where p is predicted and a is actual value

# INPUT

# p : 1d list or array of predictions

# a : 1d list or array of actual values

def rmsle(p, a):

    n = len(p)

    sigma = [(np.log(p[i] + 1) - np.log(a[i] + 1))**2 for i in range(n)]

    return np.sqrt(1/n * np.sum(sigma))
# Predicting the future for a certain region

def predict_region(reg_df, target, clf, p=P):

    # adding rows onto the end until we get to the target

    while reg_df.iloc[-1]["Date"] < target:

        last_row = reg_df.iloc[-1].copy()

        last_row["Date"] += pd.Timedelta('1 day')

        # moving the ts back and predicting the next one

        for i in range(p, 0, -1):

            if i > 1:

                first = "t-" + str(i)

                second = "t-" + str(i-1)

            else:

                first = "t-" + str(i)

                second = "t"

            last_row[first] = last_row[second]

        last_row["t"] = clf.predict(last_row.drop(["t", "Date"]).to_numpy().reshape(1,-1))[0]

        # we can work inplace because this is a copy

        reg_df = reg_df.append(last_row)

    return reg_df



# using PRE TRAINED clf, predict the fatalities or cases between start and stop

def predict_region_between(df, region, start, stop, clf, p=P):

    # evaluating only on one region between start and stop

    reg_df = df[df[f"Region_{region}"] == 1]

    reg_df = reg_df[reg_df["Date"] < start].copy()

    return predict_region(reg_df, stop, clf, p)
from sklearn.neural_network import MLPRegressor



# We train two separate mlps:

# one on data up to 19/3 for predicting to the end of March and

# one on ALL the data for predicting April

start_march = np.datetime64('2020-03-19')

end_march = np.datetime64('2020-03-31')

end_april = np.datetime64('2020-04-30')



# FATALITIES MLPS AND TRAINING SETS

mlp_some_f = MLPRegressor(verbose=True)

mlp_all_f = MLPRegressor(verbose=True)



# training data is ALL regions with dates before start

X_some_f = fatalities_df[fatalities_df["Date"] < start_march].drop(["t", "Date"], axis=1).to_numpy()

y_some_f = fatalities_df[fatalities_df["Date"] < start_march]["t"]



X_all_f = fatalities_df.drop(["t", "Date"], axis=1).to_numpy()

y_all_f = fatalities_df["t"]



mlp_some_f.fit(X_some_f, y_some_f)

mlp_all_f.fit(X_all_f, y_all_f)
# CASES MLPS AND TRAINING SETS

mlp_some_c = MLPRegressor(verbose=True)

mlp_all_c = MLPRegressor(verbose=True)



# training data is ALL regions with dates before start

X_some_c = cases_df[cases_df["Date"] < start_march].drop(["t", "Date"], axis=1).to_numpy()

y_some_c = cases_df[cases_df["Date"] < start_march]["t"]



X_all_c = cases_df.drop(["t", "Date"], axis=1).to_numpy()

y_all_c = cases_df["t"]



mlp_some_c.fit(X_some_c, y_some_c)

mlp_all_c.fit(X_all_c, y_all_c)
def graph_region_between(df, region, start, stop, clf, p=P, title=None):

    pred_df = predict_region_between(df, region, start, stop, clf)

    reg_preds = pred_df[pred_df["Date"]>=start]

    reg_true = df[df[f"Region_{region}"]==1]

    reg_true = reg_true[reg_true["Date"] >= start]

    

    plt.scatter(reg_true["t"], reg_preds["t"])

    plt.xlabel(f"True # cases in {region}")

    plt.ylabel(f"Predicted # cases in {region}")

    if title is None:

        plt.title(f"{region} from {start} to {stop}")

    else:

        plt.title(title)

    plt.plot(range(int(max(max(reg_true["t"]), max(reg_preds["t"])))))

    plt.show()



graph_region_between(cases_df, "Italy" ,start_march, end_march, mlp_some_c, p=P, title="Cases 19-31/3")
# evaluating on the whole test data set

y_hat_f = pd.DataFrame()

for region in regions:

    print(region)

    # predicting to end of march and to end of april separately

    f_some_df = predict_region_between(fatalities_df, region, start_march, end_march, mlp_some_f)

    f_some_preds = f_some_df[f_some_df["Date"]>=start_march]["t"]

    

    # predicting to end of march and to end of april separately

    f_all_df = predict_region_between(fatalities_df, region, end_march, end_april, mlp_all_f)

    f_all_preds = f_all_df[f_all_df["Date"]>end_march]["t"]



    y_hat_f = pd.concat([y_hat_f, f_some_preds, f_all_preds])

# evaluating on the whole test data set

y_hat_c = pd.DataFrame()

for region in regions:

    print(region)

    # predicting to end of march and to end of april separately

    c_some_df = predict_region_between(cases_df, region, start_march, end_march, mlp_some_c)

    c_some_preds = c_some_df[c_some_df["Date"]>=start_march]["t"]

    

    # predicting to end of march and to end of april separately

    c_all_df = predict_region_between(cases_df, region, end_march, end_april, mlp_all_c)

    c_all_preds = c_all_df[c_all_df["Date"]>end_march]["t"]



    y_hat_c = pd.concat([y_hat_c, c_some_preds, c_all_preds])
def scatter_y_yhat(y, yhat):

    plt.scatter(y, yhat)

    plt.xlabel(f"Actual values")

    plt.ylabel(f"Predicted values")

    plt.title(f"Scatter of predictions vs true values, centre line is perfect prediction")

    plt.plot(range(int(max(max(y), max(yhat)))))

    plt.show()
y_hat_f.columns = ["Fatalities"]

y_hat_c.columns = ["ConfirmedCases"]
y_h_f  = y_hat_f.copy().reset_index()

y_h_c  = y_hat_c.copy().reset_index()



print(len(y_h_f), len(sub))

sub.update(y_h_f)

sub.update(y_h_c)



sub.to_csv("submission.csv", index=False)
