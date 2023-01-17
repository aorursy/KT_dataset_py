# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn import linear_model

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train_df = pd.read_csv('../input/train_V2.csv')

test_df = pd.read_csv('../input/test_V2.csv')

train_df.info()
corr_df = train_df.corr()

corr_df = corr_df[abs(corr_df)>0.3].fillna(0)



plt.figure(figsize=(20, 20))

sns.heatmap(corr_df, annot=True, linewidth=0.5)

plt.show()
train_df = train_df.dropna(subset=["walkDistance", "swimDistance", "rideDistance", "winPlacePerc"])

train_df["Distance"] = train_df["walkDistance"] + train_df["swimDistance"] + train_df["rideDistance"]

plt.figure(figsize=(20, 20))

sns.pairplot(train_df.loc[:,["Distance","winPlacePerc"]].sample(n=10000))

plt.show()
train_df = train_df.dropna(subset=["boosts", "heals", "weaponsAcquired"])

item_df = train_df.loc[:,["boosts", "heals", "weaponsAcquired", "winPlacePerc"]]

plt.figure(figsize=(20, 20))

sns.pairplot(item_df.sample(n=10000))

plt.show()

#weaponsAcquired is not related with the target.
check_df = train_df.loc[:,["boosts", "heals", "Distance", "winPlacePerc"]]

plt.figure(figsize=(20, 20))

sns.pairplot(check_df.sample(n=10000))

plt.show()
# regression.



para = train_df[["boosts","heals","Distance","winPlacePerc"]]

para = para.dropna()

x = para[["boosts","heals","Distance"]]

y = para["winPlacePerc"]

reg = linear_model.LinearRegression().fit(x.values,y.values)

print(reg.coef_)
test_df = test_df.fillna(test_df.median())

test_df["Distance"] = test_df["walkDistance"] + test_df["swimDistance"] + test_df["rideDistance"]

tpara = test_df.loc[:,["boosts","heals","Distance"]]

y = reg.predict(tpara.values)

predict_s = pd.Series(y)

id_s = test_df["Id"]

submission_df = pd.DataFrame({"Id":id_s, "winPlacePerc":predict_s})

submission_df.to_csv("submission.csv",index=False)

#print(submission_df)