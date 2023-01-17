import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_data.sample(5)
train_data.shape
train_data.columns
sns.distplot(train_data['SalePrice'])
plt.subplots(figsize=(15, 15))
sns.heatmap(train_data.corr());
train_data.corr()['SalePrice'].sort_values(ascending=False)
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_data.sample(5)
test_data.columns
test_data.shape
pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv").sample(5)
temp = train_data.isna().sum()
temp.where(temp>0).dropna()
temp = test_data.isna().sum()
temp.where(temp>0).dropna()
train_data = train_data.drop(columns=['Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
test_data = test_data.drop(columns=['Alley','FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])
train_data.describe()
test_data.describe()
train_data.corr()['SalePrice'].sort_values(ascending=False)
correlation = train_data.corr()['SalePrice']
correlation.where(correlation>0.1)
train_data = train_data.drop(columns=['MSSubClass','OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])
test_data = test_data.drop(columns=['MSSubClass','OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'])
train_data.corr()['SalePrice'].sort_values(ascending=False)
encoded_train = pd.get_dummies(train_data.iloc[:-1,1:-1])
encoded_test = pd.get_dummies(test_data.iloc[:,1:])
final_train, final_test = encoded_train.align(encoded_test, join='right', axis=1)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

final_train = imputer.fit_transform(final_train)
final_test  = imputer.fit_transform(final_test)
y_train = train_data.iloc[:-1,-1]
X_train = final_train
X_test = final_test
# Function for saving our data into csv file.
def saveResult(filename, id_code, y_preds):
    result = pd.concat([id_code, pd.Series(y_preds)], axis = 1)
    result.rename(columns={result.columns[1]: "SalePrice"}, inplace = True)
    result.to_csv(filename, index=False)
from sklearn.linear_model import LogisticRegression

log_reg_cls = LogisticRegression()
log_reg_cls.fit(X_train, y_train)
y_preds_log_reg = log_reg_cls.predict(X_test)
y_preds_log_reg
saveResult("logical_regression.csv", test_data.Id, y_preds_log_reg)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Normalizer
gnb = GaussianNB()
y_preds_gnb = gnb.fit(X_train, y_train).predict(X_test)
y_preds_gnb
saveResult("gauss_nb.csv", test_data.Id, y_preds_gnb)
bnb = BernoulliNB(binarize=0)
y_preds_bnb = bnb.fit(X_train, y_train).predict(X_test)
y_preds_bnb
saveResult("bernulli_nb_binarize0.csv", test_data.Id, y_preds_bnb)
bnb = BernoulliNB(binarize=1)
y_preds_bnb1 = bnb.fit(X_train, y_train).predict(X_test)
y_preds_bnb1
saveResult("bernulli_nb_binarize1.csv", test_data.Id, y_preds_bnb1)
mnb = MultinomialNB()
y_preds_mnb = mnb.fit(X_train, y_train).predict(X_test)
y_preds_mnb
saveResult("mnb_correct.csv", test_data.Id, y_preds_mnb)
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
y_preds_dt = dt_clf.fit(X_train, y_train).predict(X_test)
y_preds_dt
saveResult("decision_tree.csv", test_data.Id, y_preds_dt)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rf_clf = RandomForestClassifier(max_depth=2, random_state=0)
rf_clf = rf_clf.fit(X_train, y_train)
y_preds_rf = rf_clf.predict(X_test)
y_preds_rf
saveResult("random_forest.csv", test_data.Id, y_preds_rf)