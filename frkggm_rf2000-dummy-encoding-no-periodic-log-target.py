
import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

import category_encoders as ce
import sklearn
from xgboost import XGBRegressor
import xgboost as xgb

import datetime
%matplotlib inline

import numpy as np
import pandas as pd
#%qtconsole
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)
trainData = pd.read_csv("../input/train.csv")
testData = pd.read_csv("../input/test.csv")
print(trainData.shape)
print(testData.shape)
trainData["testflag"]=0
testData["testflag"]=1
fullData = trainData.append(testData)

print(fullData.shape)
print(fullData.columns)
data = fullData.copy()
data.reset_index(inplace=True)
data["date"] = data.datetime.apply(lambda x : x.split()[0])
data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")
data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])
data["weekday"] = data.date.apply(lambda dateString : datetime.datetime.strptime(dateString,"%Y-%m-%d").weekday())
data["month"] = data.date.apply(lambda dateString : datetime.datetime.strptime(dateString,"%Y-%m-%d").month)
featcols = testData.columns.values
featcols = list(set(featcols))
featcols.append("date")
featcols.append("hour")
featcols.append("year") 
featcols.append("weekday")
featcols.append("month")
featcols= list(set(featcols))
featcols.remove("date")
featcols.remove("datetime")
featcols.remove("testflag")
featcols
data[data.testflag==1].datetime.describe()
data[data.testflag==0].datetime.describe()
from matplotlib import pyplot as pp 
A = data[data.testflag==0].sample(100)
B = data[data.testflag==1].sample(100)
#pp.plot(data[data.testflag==0].datetime, np.ones(data[data.testflag==0].shape[0]))
pp.plot(A.datetime, np.ones(A.shape[0]), ". r")
pp.plot(B.datetime, np.ones(B.shape[0]), ". b")
f=pp.figure(figsize=(16,3))
data[data.testflag==0].plot("datetime", "count", marker=".")
f.tight_layout()
f=pp.figure(figsize=(16,3))
a1=f.add_subplot(131)
data[data.testflag==0].loc[:500].plot("datetime", "count", ax=a1)
a2=f.add_subplot(132)
data[data.testflag==0].loc[2000:2500].plot("datetime", "count", ax=a2)
a3=f.add_subplot(133)
data[data.testflag==0].tail(500).plot("datetime", "count", ax=a3)
import seaborn as sns
f = pp.figure(figsize=(16,4))
a1 = f.add_subplot(131)
sns.boxplot(data=data,y="count",x="month")
a2 = f.add_subplot(132)
sns.boxplot(data=data,y="count",x="weekday")
a3 = f.add_subplot(133)
sns.boxplot(data=data,y="count",x="hour")
# Thanks to Flavia

data["weekday2"] = np.cos(data["weekday"]/6. *(np.pi) ) #runs originally from 0..6
data["month2"] = np.cos((data["month"]-1.)/11. *(np.pi) ) # runs or. from 1..12
data["hour2"] = np.cos((data["hour"])/23. *(np.pi) ) # runs or. from 1..12

featcols.append("weekday2")
featcols.append("month2")
featcols.append("hour2")

desc = data.describe(include="all")
for c in featcols+["count",]:
    desc.loc["nnan", c] = data[pd.isnull(data[c])][c].shape[0]
    desc.loc["unique", c] = data[c].unique().shape[0]
    desc.loc["is_feature", c] = 1
    desc.loc["is_numeric", c] = 1
    #print(c, data[pd.isnull(data[c])][c].shape[0])
c="count"
desc.loc["is_feature", c] = 0
# cat features:
for c in( "holiday", "season","weather", "workingday", "year" ):
    desc.loc["is_numeric", c] = 0
    data[c] = data[c].astype("category")
#desc.loc["is_numeric", ["date", "datetime"]] = 0
#desc.loc["is_feature", ["date", "datetime"]] = 0
# special: sine "weekday", "month", "year" has only 2 values, so cat.

desc

X_cat=[]
feat_num = desc.T[(desc.loc["is_numeric"]==1) & (desc.loc["is_feature"]==1)].T.columns.values

feat_cat = desc.T[desc.loc["is_numeric"]==0 & (desc.loc["is_feature"]==1) ].T.columns.values

import category_encoders as ce

for i,c in enumerate(feat_cat):
    #ci = ce.OneHotEncoder()
    ci = ce.OneHotEncoder(cols=[c], impute_missing=False)
    if i==0:
        X_cat = ci.fit_transform(data[[c]])
    else:
        X_cat = pd.concat([X_cat , ci.fit_transform(data[[c]])], axis=1)
X_cat
X_num = data[feat_num]
X_num
X = pd.concat([X_num, X_cat], axis=1)
X.shape
y = np.log(data["count"])
#y = data["count"]
y
X_train  = X[data.testflag==0].values
y_train = y[data.testflag==0]
print(X_train.shape)
print(y_train.shape)
X_test  = X[data.testflag==1].values
y_test = y[data.testflag==1]
print(X_test.shape)
print(y_test.shape)
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 2000 decision trees
rf = RandomForestRegressor(n_estimators = 2000, random_state = 42)

# Train the model on training data
rf.fit(X_train, y_train);
# # We only have labeled train data (so far)

predictions = np.exp(rf.predict(X_train))

# Calculate the absolute errors
errors = np.abs(predictions - np.exp(y_train))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', np.mean(errors) )
submission_y = np.exp(rf.predict(X_test))
submission_y
submission= data[data["testflag"]==1][["datetime"]].copy()
submission["count"] = submission_y

#submission.head(10)
submission.to_csv("submission_04.csv.gz", index=False, sep=",", compression="gzip") # 0.43
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41
gbm.fit(X_train,y_train)

# We only have labeled train data (so far)
predictions = np.exp(gbm.predict(X_train))

# Calculate the absolute errors
errors = np.abs(predictions - np.exp(y_train))

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', np.mean(errors) )


# need to create CV set
