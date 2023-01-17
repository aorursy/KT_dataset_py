# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv', index_col="id")

data_test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv', index_col="id")
data.shape, data_test.shape
data.head()
data.info()
data_test.info()
#is it cheating or not?

full_data = pd.concat([data, data_test], sort=False)
full_data.shape
full_data.info()
for col in full_data.columns.drop("target"):

    full_data[col].fillna(full_data[col].mode()[0], inplace=True)
full_data.info()
full_data["target"].groupby(full_data["ord_2"]).mean().sort_values()
full_data["bin_0"].unique()
def get_ord_convert_dict(ord_col, target_col):

    sorted_target_mean = target_col.groupby(ord_col).mean().sort_values()

    map_dict = {sorted_target_mean.index[x]: x for x in range(len(ord_col.unique()))}

    return map_dict
for column_num in range(5):

    column_name = "bin_" + str(column_num)

    map_dict = get_ord_convert_dict(full_data[column_name], full_data["target"]) 

    print(map_dict)

    full_data[column_name+"_converted"] = full_data[column_name].map(map_dict)
print(full_data["bin_3"])

print(full_data["bin_3_converted"])
full_data["nom_0"]
full_data["nom_3"]
for column_num in range(10):

    column_name = "nom_" + str(column_num)

    all_cats = full_data[column_name].unique()

    if len(all_cats) < 10: 

        for cat in all_cats:

            new_col_name = "Is" + "".join(cat.split(" "))

            full_data[new_col_name] = full_data[column_name] == cat 

            print(full_data[new_col_name])

    else:

        map_dict = get_ord_convert_dict(full_data[column_name], full_data["target"]) 

        full_data[column_name+"_converted"] = full_data[column_name].map(map_dict)
full_data.info()
for column_num in range(6):

    column_name = "ord_" + str(column_num) 

    map_dict = get_ord_convert_dict(full_data[column_name], full_data["target"]) 

    print(map_dict)

    full_data[column_name+"_converted"] = full_data[column_name].map(map_dict)
full_data["target"].groupby(full_data["ord_4"]).mean().sort_values()
full_data["day"].unique()
full_data["day"].unique()
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

for daynum, weekday in enumerate(weekdays, start=1): 

    full_data["Is"+weekday] = full_data["day"] == daynum
full_data.loc[:, "day":]
full_data["month"].unique()
full_data["month"].value_counts()
full_data.groupby("month")["target"].mean().sort_values()
#I will choose later 1hot or label but for now I will do both



#1hot

months = ["January", "February", "March", "April", "May", "June",

          "July", "August", "September", "October", "November", "December"]

for monthnum, month in enumerate(months, start=1):

    full_data["Is"+month] = full_data["month"] == monthnum



#label 

map_dict = get_ord_convert_dict(full_data["month"], full_data["target"]) 

print(map_dict)

full_data["month_converted"] = full_data["month"].map(map_dict) + 1
full_data.loc[:, "month":]
full_data.info()
to_drop = [full_data.columns[:23]]

print(to_drop)

full_data.drop(*to_drop, axis=1, inplace=True)
full_data.info()
X_train = full_data[:600000].copy(deep=True)

y_train = X_train["target"]; X_train.drop("target", axis=1, inplace=True)
X_train.info()
y_train
X_train1hot = X_train.drop("month_converted", axis=1) 

X_train1hot
X_trainlabel = X_train.drop(["Is"+x for x in months], axis=1) 

X_trainlabel
from sklearn.preprocessing import StandardScaler 



X_train_scaled = StandardScaler().fit_transform(X_train)

X_train1hot_scaled = StandardScaler().fit_transform(X_train1hot)

X_trainlabel_scaled = StandardScaler().fit_transform(X_trainlabel)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



linear1hot = LogisticRegression()

linear1hot.fit(X_train1hot_scaled, y_train)

print(linear1hot.score(X_train1hot_scaled, y_train))

print(cross_val_score(linear1hot, X_train1hot_scaled, y_train, scoring="roc_auc"))
linearlabel = LogisticRegression()

linearlabel.fit(X_trainlabel_scaled, y_train)

print(linearlabel.score(X_trainlabel_scaled, y_train))

print(cross_val_score(linearlabel, X_trainlabel_scaled, y_train, scoring="roc_auc"))
#looks like label and 1hot are absolutely identical in this case, then lets delete the one that uses more memory

print(X_train1hot_scaled.size)

print(X_trainlabel_scaled.size)
del X_train1hot, X_train1hot_scaled, linear1hot 
X_test = full_data[600000:].drop(["target"] + ["Is"+x for x in months], axis=1)

X_test_scaled = StandardScaler().fit_transform(X_test)
predictions = linearlabel.predict(X_test_scaled)

predictions
result = pd.DataFrame({

    "id": X_test.index,

    "target": predictions

})
result.to_csv('cat_submission.csv', index=False)