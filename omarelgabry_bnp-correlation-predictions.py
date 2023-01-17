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
from sklearn import cross_validation
import xgboost as xgb
# get bnp & test csv files as a DataFrame
bnp_df   = pd.read_csv('../input/train.csv')
test_df  = pd.read_csv('../input/test.csv')

# preview the data
bnp_df.head()
bnp_df.info()
print("----------------------------")
test_df.info()
# target

# most of the claims were suitable for an accelerated approval
sns.countplot(x="target", data=bnp_df)
# float & string columns

for f in bnp_df.columns:
    # fill NaN values with mean
    if bnp_df[f].dtype == 'float64':
        bnp_df[f][np.isnan(bnp_df[f])] = bnp_df[f].mean()
        test_df[f][np.isnan(test_df[f])] = test_df[f].mean()
        
    # fill NaN values with most occured value
    elif bnp_df[f].dtype == 'object':
        bnp_df[f][bnp_df[f] != bnp_df[f]] = bnp_df[f].value_counts().index[0]
        test_df[f][test_df[f] != test_df[f]] = test_df[f].value_counts().index[0]
# There are some columns with non-numerical values(i.e. dtype='object'),
# So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.

from sklearn import preprocessing

for f in bnp_df.columns:
    if bnp_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(bnp_df[f].values) + list(test_df[f].values)))
        bnp_df[f]   = lbl.transform(list(bnp_df[f].values))
        test_df[f]  = lbl.transform(list(test_df[f].values))
# define training and testing sets

X_train = bnp_df.drop(["ID","target"],axis=1)
Y_train = bnp_df["target"]
X_test  = test_df.drop("ID",axis=1).copy()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)
# Random Forests

# random_forest = RandomForestClassifier(n_estimators=100)

# random_forest.fit(X_train, Y_train)

# Y_pred = random_forest.predict_proba(X_test)[:,1]

# random_forest.score(X_train, Y_train)
# Xgboost 

params = {"objective": "binary:logistic"}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# get Coefficient of Determination(R^2) for each feature using Logistic Regression
coeff_df = DataFrame(bnp_df.columns.delete([0,1]))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = (pd.Series(logreg.coef_[0])) ** 2

# preview
coeff_df.head()
# Plot coefficient of determination in order

coeff_ser = Series(list(coeff_df["Coefficient Estimate"]), index=coeff_df["Features"]).sort_values()
fig = coeff_ser.plot(kind='barh', figsize=(15,5))
fig.set(ylim=(116, 131))
# Create submission

submission = pd.DataFrame()
submission["ID"]            = test_df["ID"]
submission["PredictedProb"] = Y_pred

submission.to_csv('bnp.csv', index=False)