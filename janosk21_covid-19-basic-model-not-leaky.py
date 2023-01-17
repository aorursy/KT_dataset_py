import numpy as np

import pandas as pd



df = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

df.shape
loc_group = ["Province_State", "Country_Region"]





def preprocess(df):

    df["Date"] = df["Date"].astype("datetime64[ms]")

    for col in loc_group:

        df[col].fillna("none", inplace=True)

    return df



df = preprocess(df)

sub_df = preprocess(pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv"))

df.head()
df["Date"].min(), df["Date"].max()
TARGETS = ["ConfirmedCases", "Fatalities"]



for col in TARGETS:

    df[col] = np.log1p(df[col])
for col in TARGETS:

    df["prev_{}".format(col)] = df.groupby(loc_group)[col].shift()
df = df[df["Date"] > df["Date"].min()].copy()

df.head()
from datetime import timedelta



TEST_DAYS = 7



TRAIN_LAST =  - timedelta(days=TEST_DAYS)



TEST_FIRST = sub_df["Date"].min()

TEST_DAYS = (df["Date"].max() - TEST_FIRST).days + 1



dev_df, test_df = df[df["Date"] < TEST_FIRST].copy(), df[df["Date"] >= TEST_FIRST].copy()

dev_df.shape, test_df.shape
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline



model = Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),

                  ('linear', LinearRegression())])



features = ["prev_{}".format(col) for col in TARGETS]



model.fit(dev_df[features], dev_df[TARGETS])



[mean_squared_error(dev_df[TARGETS[i]], model.predict(dev_df[features])[:, i]) for i in range(len(TARGETS))]
def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))



def evaluate(df):

    error = 0

    for col in TARGETS:

        error += rmse(df[col].values, df["pred_{}".format(col)].values)

    return np.round(error/len(TARGETS), 5)





def predict(test_df, first_day, num_days, val=False):



    y_pred = np.clip(model.predict(test_df.loc[test_df["Date"] == first_day][features]), None, 16)



    for i, col in enumerate(TARGETS):

        test_df["pred_{}".format(col)] = 0

        test_df.loc[test_df["Date"] == first_day, "pred_{}".format(col)] = y_pred[:, i]



    if val:

        print(first_day, evaluate(test_df[test_df["Date"] == first_day]))



    for d in range(1, num_days):

        y_pred = np.clip(model.predict(y_pred), None, 16)

        date = first_day + timedelta(days=d)



        for i, col in enumerate(TARGETS):

            test_df.loc[test_df["Date"] == date, "pred_{}".format(col)] = y_pred[:, i]



        if val:

            print(date, evaluate(test_df[test_df["Date"] == date]))

        

    return test_df



test_df = predict(test_df, TEST_FIRST, TEST_DAYS, val=True)

evaluate(test_df)
for col in TARGETS:

    test_df[col] = np.expm1(test_df[col])

    test_df["pred_{}".format(col)] = np.expm1(test_df["pred_{}".format(col)])
SUB_FIRST = sub_df["Date"].min()

SUB_DAYS = (sub_df["Date"].max() - sub_df["Date"].min()).days + 1



sub_df = dev_df.append(sub_df, sort=False)



for col in TARGETS:

    sub_df["prev_{}".format(col)] = sub_df.groupby(loc_group)[col].shift()

    

sub_df = sub_df[sub_df["Date"] >= SUB_FIRST].copy()

sub_df["ForecastId"] = sub_df["ForecastId"].astype(np.int16)

sub_df = predict(sub_df, SUB_FIRST, SUB_DAYS)



for col in TARGETS:

    sub_df[col] = np.expm1(sub_df["pred_{}".format(col)])

    

sub_df.head()
sub_df.to_csv("submission.csv", index=False, columns=["ForecastId"] + TARGETS)