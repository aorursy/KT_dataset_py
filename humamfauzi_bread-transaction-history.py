# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings  
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/BreadBasket_DMS.csv")
df_train["Datetime"] = df_train["Date"] + " " + df_train["Time"]
df_train["Datetime"] = pd.to_datetime(df_train["Datetime"])
df_train.drop(["Date", "Time"], axis=1, inplace=True)
df_train.info()
df_train["Item"].nunique()
buffer = []
for i in df_train["Transaction"].unique():
    arr_buffer = np.array(df_train[df_train["Transaction"] == i])
    list_transaction = {"Date": arr_buffer[0,2] , "Item": arr_buffer[:,1]}
    buffer.append(list_transaction)
new_df_train = pd.DataFrame(buffer)
new_df_train.head()
for i in df_train["Item"].unique():
    new_df_train[i] = 0
new_df_train.head()
for i in range(len(new_df_train)):
    for k in new_df_train.loc[i,"Item"]:
        new_df_train.loc[i, k] += 1

new_df_train.drop("Item", axis=1, inplace=True)
new_df_train.head()
a = new_df_train["Date"]

def plot_transaction(Series, title):
    sorted_val = Series.value_counts().sort_index()
    plt.figure(figsize=(14,8))
    plt.title(title)
    
    plt.xlabel("Unique Value")
    plt.ylabel("Counts")
    plt.grid()
    
    sns.barplot(y=np.array(sorted_val), x=sorted_val.index)
hour = a.apply(lambda x: x.hour)
plot_transaction(hour, "Hour")
month = a.apply(lambda x: x.month)
plot_transaction(month, "Month")
day = a.apply(lambda x: x.day)
plot_transaction(day, "Day")
df_bread = new_df_train[["Date", "Bread"]]
df_bread["Date__"] = df_bread["Date"].apply(lambda x:x.date())
arr = []
for i in df_bread["Date__"].unique():
    arr.append(df_bread["Bread"][df_bread["Date__"]==i].sum())
bread_Series = pd.Series(arr, index=df_bread["Date__"].unique())
plt.figure(figsize=(14,8))
plt.title("Bread Transaction")
plt.plot(bread_Series)
plt.grid()

plt.xlabel("Time")
plt.ylabel("Transaction Count")
def sequencer(Series, pad):
    seq = Series
    X = np.zeros([len(seq)-pad, pad])
    Y = np.zeros(len(seq)-pad)
    for i in range(len(seq)-pad):
        X[i, :] = seq[i:i+pad]
        Y[i] = seq[i+pad]
    return X, Y

X, Y = sequencer(bread_Series, 10)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

X_tr, X_te, y_tr, y_te = train_test_split(X,Y, test_size=0.2, random_state=77)

estimators = []
MAE = []

estimators.append(KNeighborsRegressor())
estimators.append(LinearRegression())
estimators.append(DecisionTreeRegressor())
estimators.append(AdaBoostRegressor())
estimators.append(GradientBoostingRegressor())

for i in estimators:
    i.fit(X_tr, y_tr)
    MAE.append(mean_absolute_error(y_te, i.predict(X_te)))

plt.figure(figsize=(14,5))
sns.barplot(x=MAE, y=["KNR", "LR", "DTR", "ABR", "GBR"], )
plt.grid()
plt.title("Mean Absolute Error For Bread Transaction")
plt.xlabel("Error")
plt.ylabel("Estimator")
def predict_item(original, col):
    df = new_df_train[["Date", col]]
    df["Date__"] = df["Date"].apply(lambda x:x.date())
    
    arr = []
    for i in df["Date__"].unique():
        arr.append(df[col][df["Date__"]==i].sum())
    new_series = pd.Series(arr, index=df["Date__"].unique())
    
    X, Y = sequencer(new_series, 10)
    X_tr, X_te, y_tr, y_te = train_test_split(X,Y, test_size=0.2, random_state=77)

    estimators = []
    MAE = []

    estimators.append(KNeighborsRegressor())
    estimators.append(LinearRegression())
    estimators.append(DecisionTreeRegressor())
    estimators.append(AdaBoostRegressor())
    estimators.append(GradientBoostingRegressor())

    for i in estimators:
        i.fit(X_tr, y_tr)
        MAE.append(mean_absolute_error(y_te, i.predict(X_te)))

    plt.figure(figsize=(14,5))
    sns.barplot(x=MAE, y=["KNR", "LR", "DTR", "ABR", "GBR"], )
    plt.grid()
    plt.title("Mean Absolute Error For " + col + " Transaction")
    plt.xlabel("Error")
    plt.ylabel("Estimator")

predict_item(new_df_train, "Coffee")
predict_item(new_df_train, "Scandinavian")
predict_item(new_df_train, "Muffin")