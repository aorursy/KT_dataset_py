# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/year_prediction.csv")

data = data.rename(index=str, columns={"label":"year"})
nsongs = {}

for y in range(1922,2012):

    nsongs[y] = len(data[data.year==y])

yrs = range(1922,2011)

values = [nsongs[y] for y in yrs]

plt.bar(yrs, values, align='center')

plt.xlabel("Year")

plt.ylabel("Number of songs")
# separate input attributes and output into different dataframes

X = data.iloc[:,1:]

Y = data.iloc[:,0]







# Train set

X_train = X.iloc[0:463715,:]

y_train = Y.iloc[0:463715]



# Validation set

X_test = X.iloc[463715:,:]

y_test = Y.iloc[463715:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the train set and the test set.

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



X_train = pd.DataFrame(X_train_scaled,columns=X_train.columns)

X_test = pd.DataFrame(X_test_scaled,columns=X_train.columns)
X_train.describe()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training set only.

scaler.fit(X_train)

# Apply transform to both the train set and the test set.

X_train_std = scaler.transform(X_train)

X_test_std = scaler.transform(X_test)
X_train_std = pd.DataFrame(X_train_std,columns=X_train.columns)

X_train_std.describe()
from sklearn.decomposition import PCA

# Make an instance of the Model

pca = PCA(.90)



# We fit to only our training set

pca.fit(X_train_std)

# Print number of components generated

pca.n_components_
X_train_proc = pca.transform(X_train_std)

X_test_proc = pca.transform(X_test_std)
y_train_proc = y_train - min(y_train)

y_test_proc = y_test - min(y_test)

# y_train_proc
from sklearn.linear_model import LogisticRegression



# logisticRegr = LogisticRegression(verbose=10)

# logisticRegr.fit(X_train_proc, y_train_proc)
# predictions = logisticRegr.predict(X_test_proc)
# score = logisticRegr.score(X_test_proc, y_test_proc)

# score
# predictions = logisticRegr.predict(X_test_proc)

# np.mean(np.absolute((predictions-np.array(y_test_proc))))
from sklearn.linear_model import LinearRegression



linearRegr = LinearRegression()
linearRegr.fit(X_train_proc, y_train_proc)
score_linearReg = linearRegr.score(X_test_proc, y_test_proc)

score_linearReg
predictions_linearRegr = linearRegr.predict(X_test_proc)

np.mean(np.absolute((predictions_linearRegr-np.array(y_test_proc))))
# from sklearn.metrics import mean_squared_error, r2_score

# mean_squared_error(predictions_linearRegr, np.array(y_test_proc))
# from sklearn.metrics import accuracy_score



# print(accuracy_score(predictions, y_test_proc))
# predictions
df_plot = pd.DataFrame([np.array(predictions_linearRegr).T, np.array(y_test_proc).T])

df_plot = df_plot.transpose()

df_plot = df_plot.sort_values([1]).reset_index(drop=True)

plt.scatter(df_plot[1], df_plot[0], s=1)

# plt.line()

# plt.plot(df_plot)google
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(np.round_(np.array(predictions_linearRegr), decimals=-1), np.round_(np.array(y_test_proc), decimals=-1))

# np.round_(np.array(predictions_linearRegr), decimals=-1)
import seaborn as sns

ind = list(range(1920,2040,10))

df_heat = pd.DataFrame(cm, index=ind, columns=ind)

len(ind)

# lab = pd.unique(df_heat[0])

sns.heatmap(df_heat)

df_heat

# df_plot.transpose().corr()