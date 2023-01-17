import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import plotly

%matplotlib inline

from pandas import concat, DataFrame

from math import sqrt

from numpy import concatenate

from matplotlib import pyplot

# Fix random seed for reproducibility

np.random.seed(6)
plt.rcParams["figure.figsize"]=(10,5)

# To avoid conflict between pandas and prophet

pd.plotting.register_matplotlib_converters()
#Train data 

train1=pd.read_csv("../input/ashrae-energy-prediction/train.csv",index_col="timestamp",parse_dates=True,squeeze=True)

#test data

test1=pd.read_csv("../input/ashrae-energy-prediction/test.csv",index_col="timestamp",parse_dates=True,squeeze=True)

train1.head(2)
test1.head(2)
train1.info()
test1.info()
train1.isnull().sum()
test1.isnull().sum()
train1.describe()
# Correlation of train attributes

corr_train=train1.corr(method="spearman")
#correlation heatmap plot between features in train data

corr_train.style.background_gradient(cmap='viridis').set_precision(2)
# Count per meter type in train data

sns.countplot("meter",data=train1)

plt.title("NUMBER OF METERS PER METER TYPE N TRAIN DATA")

plt.xlabel("METER TYPE")

plt.ylabel("COUNT PER METER TYPE")

plt.show()
# Count per meter type in test data

sns.countplot("meter",data=test1)

plt.title("NUMBER OF METERS PER METER TYPE N TEST DATA")

plt.xlabel("METER TYPE")

plt.ylabel("COUNT PER METER TYPE")

plt.show()
meter_t0,meter_t1,meter_t2,meter_t3=train1[train1["meter"]==0],train1[train1["meter"]==1],train1[train1["meter"]==2],train1[train1["meter"]==3]
meter_e0,meter_e1,meter_e2,meter_e3=test1[test1["meter"]==0],test1[test1["meter"]==1],test1[test1["meter"]==2],test1[test1["meter"]==3]
# Meter type to meter reading plots

meter_types=(meter_t0,meter_t1,meter_t2,meter_t3)

for meter_type in meter_types:

    plt.plot(meter_type.index,meter_type.meter_reading)

    plt.show()
# Plotting time series data after grouping

types=(0,1,2,3)

meters=train1.groupby("meter")

meters=meters["meter_reading"].resample("MS").mean()

for meter in types:

    plt.title(meter)

    meters[meter].plot()

    plt.show()
x_train=pd.DataFrame(train1[["meter","building_id"]])

x_test=pd.DataFrame(test1[["meter","building_id"]])
y_train=pd.DataFrame(train1["meter_reading"])
# Import and use random forest

from sklearn.ensemble import RandomForestRegressor as RF
# Model Fit and Predict

regressor = RF(n_estimators=10,

             criterion='mse',

             max_features= None, 

             max_depth = 14,bootstrap=True)

regressor=regressor.fit(x_train, y_train.values.ravel())

regressor.fit(x_train, y_train.values.ravel())

y_pred = regressor.predict(x_test)
y_pred1=pd.DataFrame(y_pred)

y_pred1.tail()
y_pred1=y_pred1.reset_index()
# Top rows after resetting index

y_pred1.head()
y_pred1.rename(columns={"index":"row_id",

                       0:"meter_reading"},inplace=True)
y_pred1.shape
#Final Submmission

y_pred1.to_csv("SubmissionA.csv",index=False)