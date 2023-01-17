# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sbn

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df.head()
# remove Iris part

df["Species"] = df['Species'].apply(lambda x: x.split('-')[1])
ax = sbn.swarmplot(x="Species", y="SepalWidthCm", data=df, color='white')

ax = sbn.violinplot(x="Species", y="SepalWidthCm", data=df, inner=None)
#g = sbn.FacetGrid(df, col="Species")

#g = g.map(plt.scatter, "SepalWidthCm", "SepalLengthCm")

#plt.show()
cols = df.columns[1:-1]


df.pivot_table(index="Species", values=cols, aggfunc='mean')
df["feat1"] = df["PetalLengthCm"] + df["PetalWidthCm"]

df["feat2"] = df["SepalLengthCm"] + df["SepalWidthCm"]



g = sbn.pairplot(df.drop('Id', axis=1), hue="Species")

plt.show()

#df.drop('Id', axis=1).groupby("Species").corr()

from sklearn.ensemble import RandomForestClassifier 

forest = RandomForestClassifier(n_estimators = 200, oob_score=True)



df = df.sample(frac=1).reset_index(drop=True)

train = df.drop('Id', axis=1)[:75]

test = df.drop('Id' , axis=1)[75:]

test_cols = df.columns.tolist()

test_cols = test_cols[1:5] + test_cols[-2:] 

test = test[test_cols]

train_cols = df.columns.tolist()

train_cols = train_cols[1:5] + train_cols[-2:] + train_cols[5:6]

train = train[train_cols]

forest.fit(train.ix[:,:6],train.ix[:,6])
prediction = pd.DataFrame([forest.predict(test), df["Species"][75:]]).T
prediction = pd.DataFrame([forest.predict(test), df["Species"][75:]]).T



prediction.columns = ['Prediction', 'Real_value'] 

prediction["Prediction"][prediction["Prediction"] == 'setosa'] = 0

prediction["Prediction"][prediction["Prediction"] == 'versicolor'] = 1

prediction["Prediction"][prediction["Prediction"] == 'virginica'] = 2



prediction["Real_value"][prediction["Real_value"] == 'setosa'] = 0

prediction["Real_value"][prediction["Real_value"] == 'versicolor'] = 1

prediction["Real_value"][prediction["Real_value"] == 'virginica'] = 2

prediction["Score"] = 0

prediction["Score"][prediction["Prediction"] == prediction["Real_value"]] = 1
forest.feature_importances_

prediction["Score"].value_counts(normalize=True)

forest.oob_score_
# Scoring is not robust, this is preliminary. 