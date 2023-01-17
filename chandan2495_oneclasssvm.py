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
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/air-passengers/AirPassengers.csv")

df.head()
months = df["Month"].str.split("-", expand=True)

df["month_only"] = months[1]

df["year_only"] = months[0]
df["month_only"] = df["month_only"].astype(int)

df["year_only"] = df["year_only"].astype(int)
df["day"] = 1

df["month"] = df["month_only"]

df["year"] = df["year_only"]

df = df.set_index(pd.to_datetime(df[["month", "year", "day"]]))

df.describe()
df.set_index("year_only")["#Passengers"].plot(figsize=(15,10), fontsize=18)

plt.show()
data = df[["#Passengers"]]

from statsmodels.tsa.seasonal import seasonal_decompose

series = data

result = seasonal_decompose(series)
plt.subplot(411)

plt.plot(data, label='Original')

plt.legend(loc='best')

plt.subplot(412)

plt.plot(result.trend, label='Trend')

plt.legend(loc='best')

plt.subplot(413)

plt.plot(result.seasonal,label='Seasonality')

plt.legend(loc='best')

plt.subplot(414)

plt.plot(result.resid, label='Residuals')

plt.legend(loc='best')

plt.tight_layout()
# X = df.drop(columns=["year_only", "Month"], axis="columns")

X = df[["#Passengers", "year_only"]]

passengers_mean = X["#Passengers"].mean()

X["#Passengers"] = X["#Passengers"] #/ passengers_mean

X1 = X.copy()

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
from sklearn.svm import OneClassSVM
res_mean = result.resid.mean()

train_data = result.resid.to_frame().fillna(value=res_mean)

ocs = OneClassSVM(nu=0.001, kernel="sigmoid", gamma=0.1)

ocs.fit(train_data)
output = ocs.predict(train_data)

print(output)

print(f"Positive {sum(output == 1)}")

print(f"Negative {sum(output == -1)}")
X1 = df.copy()

X1.iloc[90, df.columns.get_loc("#Passengers")] = 800 #/ passengers_mean

X1.iloc[133, df.columns.get_loc("#Passengers")] = 600 #/ passengers_mean

# X1 = X1.iloc[[90,133],:]

X2 = X1.copy()

# X1 = scaler.transform(X1)



test_series = X1[["#Passengers"]]

test_result = seasonal_decompose(test_series)

test_X = test_result.resid.to_frame().fillna(value=result.resid.mean())
# passengers_mean

# test_X["resid"]

# X1.iloc[133, df.columns.get_loc("#Passengers")] = 100

# X1
X2.reset_index().iloc[ocs.predict(test_X) == -1, :]
# X.iloc[90]
plt.style.use("ggplot")

fig, ax = plt.subplots(figsize=(15,10))

# create a mesh to plot in

h = 0.02

x_min, x_max = train_data.iloc[:,0].min() - 1, train_data.iloc[:,0].max() + 1

y_min, y_max = train_data.iloc[:,0].min() - 1, train_data.iloc[:,0].max() + 1



x_outlier_min, x_outlier_max = train_data.iloc[:,0].min() - 1, train_data.iloc[:,0].max() + 1

y_outlier_min, y_outlier_max = test_X.iloc[:,0].min() - 1, test_X.iloc[:,0].max() + 1





xx, yy = np.meshgrid(np.arange(min(x_min, x_outlier_min), max(x_max, x_outlier_max), h),

                     np.arange(min(y_min, y_outlier_min), max(y_max, y_outlier_max), h))

Z = ocs.predict(xx.ravel().reshape(-1,1))

Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

normal = ax.scatter(X[:, 0], X[:, 1], c="gold", cmap=plt.cm.coolwarm)



outlier = ax.scatter(X1[:, 0], X1[:, 1], c="white", cmap=plt.cm.coolwarm)



ax.legend([normal, outlier], ["Normal passengers count", "Outlier passenger count"], fontsize="18")



plt.show()
train_data.iloc[:,0]