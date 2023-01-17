%matplotlib inline
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
# get homesite & test csv files as a DataFrame
homesite_df = pd.read_csv("../input/train.csv")
test_df     = pd.read_csv("../input/test.csv")

# preview the data
homesite_df.head()
homesite_df.info()
print("----------------------------")
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction
homesite_df = homesite_df.drop(['QuoteNumber'], axis=1)
# date

# Convert Date to Year, Month, and Week
homesite_df['Year']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
homesite_df['Month'] = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
homesite_df['Week']  = homesite_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

test_df['Year']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[:4]))
test_df['Month'] = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[5:7]))
test_df['Week']  = test_df['Original_Quote_Date'].apply(lambda x: int(str(x)[8:10]))

homesite_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
test_df.drop(['Original_Quote_Date'], axis=1,inplace=True)
# customers purchased insurance plan

# Plot
sns.countplot(x="QuoteConversion_Flag", data=homesite_df)
# year
# Which year has higher number of customers purchased insurance plan

# Plot
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(x="QuoteConversion_Flag",hue="Year", data=homesite_df, ax=axis1)
sns.countplot(x=homesite_df["Year"].loc[homesite_df["QuoteConversion_Flag"] == 1], 
              order=[2013,2014,2015], ax=axis2)
# month
# Which month has higher number of customers purchased insurance plan

# Plot
sns.countplot(x=homesite_df["Month"].loc[homesite_df["QuoteConversion_Flag"] == 1], 
              order=[1,2,3,4,5,6,7,8,9,10,11,12])
# fill NaN values

homesite_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)
# There are some columns with non-numerical values(i.e. dtype='object'),
# So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.

from sklearn import preprocessing

for f in homesite_df.columns:
    if homesite_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(homesite_df[f].values) + list(test_df[f].values)))
        homesite_df[f] = lbl.transform(list(homesite_df[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))
# define training and testing sets

X_train = homesite_df.drop("QuoteConversion_Flag",axis=1)
Y_train = homesite_df["QuoteConversion_Flag"]
X_test  = test_df.drop("QuoteNumber",axis=1).copy()

def classify_RF(X_train, Y_train, X_test):
    print("\nFitting Training Data...", flush=True)
    rfc = RandomForestClassifier(n_estimators=200, max_features=5, min_samples_leaf=50, n_jobs=-1, class_weight='balanced', verbose=2)
    calibrated_rfc = CalibratedClassifierCV(rfc, method="isotonic", cv=5)
    calibrated_rfc.fit(X_train, Y_train)
    print("Final Prediction", flush=True)
    return calibrated_rfc.predict(X_test)
# Xgboost 

#params = {"objective": "binary:logistic"}
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import maxabs_scale
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

#scale
X_train = maxabs_scale(X_train)
X_test = maxabs_scale(X_test)

#T_train_xgb = xgb.DMatrix(X_train, Y_train)
#X_test_xgb  = xgb.DMatrix(X_test)

#gbm = xgb.train(params, T_train_xgb, 20)
#Y_pred = gbm.predict(X_test_xgb)


Y_test = classify_RF(X_train, Y_train, X_test)
print("Generating Submission")
submission = pd.read_csv("../input/sample_submission.csv")
submission.QuoteConversion_Flag = Y_test
submission.to_csv("RF_grid.csv", index=False)
