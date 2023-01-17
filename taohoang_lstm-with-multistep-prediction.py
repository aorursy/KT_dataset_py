# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from datetime import timedelta

        

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
# check if any col is null

train_df.apply(lambda col: col.isnull().value_counts(), axis=0)
test_df.apply(lambda col: col.isna().value_counts(), axis=0)
# fill na

train_df["Province_State"] = train_df["Province_State"].fillna("")

test_df["Province_State"] = test_df["Province_State"].fillna("")
train_df["Date"] = pd.to_datetime(train_df["Date"])

test_df["Date"] = pd.to_datetime(test_df["Date"])
train_df["NewCases"] = train_df.groupby(["Country_Region", "Province_State"])["ConfirmedCases"].diff(periods=1)

train_df["NewCases"] = train_df["NewCases"].fillna(0)

train_df["NewCases"] = np.where(train_df["NewCases"] < 0, 0, train_df["NewCases"])

train_df["NewFatalities"] = train_df.groupby(["Country_Region", "Province_State"])["Fatalities"].diff(periods=1)

train_df["NewFatalities"] = train_df["NewFatalities"].fillna(0)

train_df["NewFatalities"] = np.where(train_df["NewFatalities"] < 0, 0, train_df["NewFatalities"])
train_df["NewCases"] = np.log(train_df["NewCases"] + 1)

train_df["NewFatalities"] = np.log(train_df["NewFatalities"] + 1)
def preprocess_train(n_prev, n_next):

    df = train_df.copy()

    input_feats, output_feats = [], []

    for i in range(1, n_prev+1):

        for feat in ["NewCases", "NewFatalities"]:

            df["{}_prev_{}".format(feat, i)] = df.groupby(["Country_Region", "Province_State"])[feat].shift(i)

            input_feats.append("{}_prev_{}".format(feat, i))

    

    output_feats.extend(["NewCases", "NewFatalities"])

    for i in range(1, n_next):

        for feat in ["NewCases", "NewFatalities"]:

            df["{}_next_{}".format(feat, i)] = df.groupby(["Country_Region", "Province_State"])[feat].shift(-i)

            output_feats.append("{}_next_{}".format(feat, i))

    df.dropna(inplace=True)       

            

    const_df = pd.get_dummies(df[["Province_State", "Country_Region"]], drop_first=True)

    time_df = df[input_feats]

    time_df = time_df.values.reshape((df.shape[0],-1,2))

    output_df = df[output_feats]

    return const_df, time_df, output_df
def preprocess_test(n_prev):

    input_feats = []

    append_df = pd.concat([train_df, test_df[test_df["Date"] == train_df["Date"].max() + timedelta(days=1)]])

    append_df.sort_values(["Country_Region", "Province_State", "Date"], ascending=[True, True, True], inplace=True)

    for i in range(1, n_prev+1):

        for feat in ["NewCases", "NewFatalities"]:

            append_df["{}_prev_{}".format(feat, i)] = append_df.groupby(["Country_Region", "Province_State"])[feat].shift(i)

            input_feats.append("{}_prev_{}".format(feat, i))

    append_df = append_df[append_df["ForecastId"].notnull()]

            

    const_df = pd.get_dummies(append_df[["Province_State", "Country_Region"]], drop_first=True)

    time_df = append_df[input_feats]

    time_df = time_df.values.reshape((append_df.shape[0],-1,2))

    return const_df, time_df
n_next = (test_df["Date"].max() - train_df["Date"].max()).days

n_next
const_df, time_df, output_df = preprocess_train(n_next, n_next)
const_test_df, time_test_df = preprocess_test(n_next)
time_test_df.shape
from keras.models import Model

from keras import layers

from keras import Input
time_input = Input(shape=(time_df.shape[1], time_df.shape[2]))

lstm = layers.LSTM(32)(time_input)



const_input = Input(shape=(const_df.shape[1],))



combine = layers.concatenate([lstm, const_input], axis=-1)

output = layers.Dense(output_df.shape[1], activation='relu')(combine)

model = Model([time_input, const_input], output)

model.compile(optimizer='adam',

              loss='mean_squared_error')

model.summary()
model.fit([time_df, const_df], output_df, epochs=300, batch_size=128)
output = model.predict([time_test_df, const_test_df])

output.shape
sub_test_df = test_df[test_df["Date"] > train_df["Date"].max()]

sub_test_df = pd.concat([sub_test_df,

                         pd.DataFrame(output.reshape((-1, 2)), columns=["NewCases", "NewFatalities"], index=sub_test_df.index)],

                         axis=1)

sub_test_df["NewCases"] = np.exp(sub_test_df["NewCases"]) - 1

sub_test_df["NewFatalities"] = np.exp(sub_test_df["NewFatalities"]) - 1

sub_test_df
fixed_test_df = test_df[test_df["Date"] <= train_df["Date"].max()].merge(train_df[train_df["Date"] >= test_df["Date"].min()][["Province_State","Country_Region", "Date", "ConfirmedCases", "Fatalities"]],

                                                                         how="left", on=["Province_State","Country_Region", "Date"])

fixed_test_df
predict_df = pd.concat([sub_test_df, fixed_test_df]).sort_values(["Country_Region", "Province_State", "Date"],

                                                                 ascending=[True, True, True])

predict_df
predict_df = predict_df.reset_index()

for i in range(len(predict_df)):

    if pd.isnull(predict_df.iloc[i]["ConfirmedCases"]):

        predict_df.loc[i, "ConfirmedCases"] = predict_df.iloc[i - 1]["ConfirmedCases"] + predict_df.iloc[i]["NewCases"]

    if pd.isnull(predict_df.iloc[i]["Fatalities"]):

        predict_df.loc[i, "Fatalities"] = predict_df.iloc[i - 1]["Fatalities"] + predict_df.iloc[i]["NewFatalities"]

predict_df
assert predict_df.shape[0] == test_df.shape[0]
predict_df[["ForecastId", "ConfirmedCases", "Fatalities"]].to_csv("submission.csv", index=False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
country = "Australia"

state = "South Australia"

target = "ConfirmedCases"

region_train_df = train_df[(train_df["Country_Region"]==country) & (train_df["Province_State"]==state)]

region_predict_df = predict_df[(predict_df["Country_Region"]==country) & (predict_df["Province_State"]==state)]



fig = plt.figure(figsize=(10, 6))

ax1 = fig.add_axes([0, 0, 1, 1])

ax1.plot(region_train_df["Date"],

         region_train_df[target],

         color="green")



ax1.plot(region_predict_df["Date"],

         region_predict_df[target],

         color="red")

plt.show()