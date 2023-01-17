# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.v
train = pd.read_csv("../input/train_2016_v2.csv")

prop = pd.read_csv("../input/properties_2016.csv")

sub = pd.read_csv("../input/sample_submission.csv")
train.head()
prop.head()
merged_df = train.merge(prop, how="left", on="parcelid")

print(train.shape)

print(prop.shape)

print(merged_df.shape)

merged_df.head()
merged_df.info()
cat_features = ["hashottuborspa", "propertycountylandusecode", 

                "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]



for col in cat_features:

    print("category", col, ":")

    print(merged_df[col].unique())
merged_df['Quarter'] = pd.PeriodIndex(merged_df['transactiondate'], freq='Q').strftime('Q%q')

merged_df = pd.concat([merged_df, pd.get_dummies(merged_df["Quarter"], prefix_sep='_')], axis=1)

merged_df['first15daysmonth'] = np.where(pd.to_datetime(merged_df['transactiondate']).dt.day <= 15, 1, 0)



merged_df.drop(["Quarter"], axis=1, inplace=True)
merged_df.tail()
merged_df["logerror"].plot(figsize=(16,9))
merged_df["logerror"].plot.density(figsize=(16,9))
merged_df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2,

                c="logerror", cmap=plt.get_cmap("jet"), colorbar=True,

                figsize=(16,9))

plt.legend()
corr_matrix = merged_df.corr()
corr_matrix["logerror"].sort_values(ascending=False)
merged_df["bath_per_sqft"] = merged_df["bathroomcnt"] / merged_df["calculatedfinishedsquarefeet"]

merged_df["bath3-4_per_sqft"] = merged_df["threequarterbathnbr"] / merged_df["calculatedfinishedsquarefeet"]

merged_df["sqft_per_floor"] = merged_df["calculatedfinishedsquarefeet"] / merged_df["numberofstories"]

merged_df["basement_perc_totalarea"] = merged_df["basementsqft"] / merged_df["finishedsquarefeet15"]
corr_matrix = merged_df.corr()

corr_matrix["logerror"].sort_values(ascending=False)
merged_df.head()
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, r2_score



X = [merged_df.iloc[:, 3:][col] for col in merged_df.iloc[:, 3:].columns if merged_df.iloc[:, 3:][col].dtype!="object"]

X = pd.DataFrame(X).transpose().fillna(0)

y = merged_df["logerror"].fillna(0)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

dt_y_pred = dt.predict(X_test)



r2_score(y_true=y_test, y_pred=dt_y_pred)