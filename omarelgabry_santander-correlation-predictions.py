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
# get santander & test csv files as a DataFrame
santander_df = pd.read_csv("../input/train.csv")
test_df      = pd.read_csv("../input/test.csv")

# preview the data
santander_df.head()
santander_df.info()
print("----------------------------")
test_df.info()
# drop unnecessary columns, these columns won't be useful in analysis and prediction

# 1. constant columns
columns_to_be_removed = []
cunt_constant = 0

for col in santander_df.columns:
    if santander_df[col].std() == 0:
        columns_to_be_removed.append(col)
        cunt_constant =  cunt_constant + 1

santander_df.drop(columns_to_be_removed, axis=1, inplace=True)
test_df.drop(columns_to_be_removed, axis=1, inplace=True)

# 2. duplicated columns
columns_to_be_removed = []
cunt_duplicates = 0

for col_i in santander_df.columns:
    for col_j in santander_df.columns:
        if(col_i == col_j): continue
        if((santander_df[col_i] == santander_df[col_j]).all()):
            columns_to_be_removed.append(col_i)
            cunt_duplicates = cunt_duplicates + 1
            break
            
santander_df.drop(columns_to_be_removed, axis=1, inplace=True)
test_df.drop(columns_to_be_removed, axis=1, inplace=True)
# plot count of constant & duplicate columns

axis = Series([cunt_constant, cunt_duplicates]).plot(kind='bar')
labels = axis.set_xticklabels(["Constants", "Duplicates"], rotation=0)
# get unique values for each column
cunt_nunique = santander_df.apply(lambda col:col.nunique())
cunt_nunique.drop("ID", inplace=True)

# plot frequency of unique values only for columns with <= 100 unique elements
cunt_nunique[cunt_nunique <= 100].hist(bins=100, figsize=(15, 5))
# Replace -999999 in var3 column with most common value 

most_common_num = santander_df['var3'].value_counts().index[0]
santander_df["var3"][santander_df["var3"] == -999999] = most_common_num
# define training and testing sets

X_train = santander_df.drop(["ID","TARGET"],axis=1)
Y_train = santander_df["TARGET"]
X_test  = test_df.drop("ID",axis=1).copy()
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict_proba(X_test)[:,1]

logreg.score(X_train, Y_train)
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict_proba(X_test)[:,1]

random_forest.score(X_train, Y_train)
# Xgboost 

params = {"objective": "binary:logistic"}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# get Coefficient of Determination(R^2) for each feature using Logistic Regression

coeff_df = DataFrame(X_train.columns)
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = (pd.Series(logreg.coef_[0])) ** 2

# preview
coeff_df.head()
# Plot coefficient of determination in order

coeff_ser = Series(list(coeff_df["Coefficient Estimate"]), index=coeff_df["Features"]).sort_values()
coeff_ser.tail(15).plot(kind='barh', figsize=(15,5), title="Coefficient of Determination(R^2)")
# Plot feature importance in order using Random Forest Classifier

imp_ser = Series(random_forest.feature_importances_, index=X_train.columns).sort_values()
imp_ser.tail(15).plot(kind='barh', figsize=(15,5), title="Feature importance")
# Create submission

submission = pd.DataFrame()
submission["ID"]     = test_df["ID"]
submission["TARGET"] = Y_pred

submission.to_csv('santander.csv', index=False)