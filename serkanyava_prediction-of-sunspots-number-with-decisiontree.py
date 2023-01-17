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
sspot=pd.read_csv('/kaggle/input/daily-sun-spot-data-1818-to-2019/sunspot_data.csv')
sspot
sspot.drop("Unnamed: 0", axis=1, inplace=True)
sspot.sample(5)
sspot.dtypes
sspot[(sspot["Number of Sunspots"]== -1 )]#3247 missing values in 73718 rows
df=sspot.copy()
nan_value = float("NaN")



df.replace(-1,nan_value, inplace=True)



df
df = df.dropna()

df = df.reset_index(drop=True)
df
df.isnull().sum()
df.groupby("Year").agg(["min", "max", "std", "mean"])
a=df.groupby(['Year']).sum()

t=a.sort_values(by='Number of Sunspots', ascending = False)

t['Number of Sunspots']
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
cmap = sns.cubehelix_palette(dark=.3, light=.9, as_cmap=True)

plt.figure(figsize=(20,20))

sns.scatterplot(data=df, x="Number of Sunspots", y="Year", hue="Observations", palette=cmap)

plt.show()
t['Number of Sunspots'].head(5).plot(kind='barh', figsize=(10,10))

yaxes="Number of Sunpots"
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(15,15))

sns.heatmap(dfcorr, cmap='coolwarm')
df
x = df.drop(["Number of Sunspots","Month","Day"],axis=1)
x
y= df["Number of Sunspots"]

y=y.to_frame()

y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)

x_train.shape
y_train.shape
from sklearn.neural_network import MLPRegressor

neu = MLPRegressor(random_state=1, max_iter=450).fit(x_train, y_train)
df_new=pd.DataFrame(neu.predict(x_test))

pred_val= df_new.rename(columns={0: 'Number of Sunspots'})

pred_val
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

print("MLPRegressor's R2 Score:",r2_score(y_test, pred_val))
from sklearn import tree

clf = tree.DecisionTreeRegressor()

clf = clf.fit(x_train, y_train)
df_new=pd.DataFrame(clf.predict(x_test))

pred_val= df_new.rename(columns={0: 'Number of Sunspots'})

pred_val
print("Decision Tree Regressor's R2 Score:",r2_score(y_test, pred_val))
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score
mean_absolute_error(y_test, pred_val)
pred_val.head(5)
y_test.head(5)
pred_val.to_csv('sunspotprediction.csv', index=False)