# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.distplot(train.SalePrice)
sns.lmplot(x="1stFlrSF", y="SalePrice", data=train)
num_missing = train.isnull().sum()
percent = num_missing / train.isnull().count()

df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])
df_missing = df_missing.sort_values('Fraction', ascending=False)
df_missing[df_missing['MissingValues'] > 0]

drop = df_missing[df_missing["MissingValues"] == 0].index
train = train[drop]
heatmap = train.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(heatmap, square=True)
interesting = heatmap['SalePrice'].sort_values(ascending=False)
interesting = interesting[abs(interesting) >= 0.4]
interesting = interesting[interesting.index != 'SalePrice']
interesting
sns.lmplot(x="OverallQual", y="SalePrice", data=train)
cols = interesting.index.values.tolist() + ['SalePrice']
sns.pairplot(train[cols], size=2.5)
plt.show()
heatmap = train[cols].corr()
sns.heatmap(heatmap)
from sklearn.model_selection import train_test_split

X_all = train[cols].drop(["SalePrice"],axis=1)
y_all = train["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.10, random_state=20)

model = RandomForestRegressor(n_estimators=300, n_jobs=2,oob_score=True, min_samples_leaf=5,random_state=42)
model.fit(X_train, y_train)
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

y_pred=model.predict(X_test)
print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred))
print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred))
feature_imp = pd.Series(model.feature_importances_,index=list(X_train)).sort_values(ascending=False)
feature_imp
test = pd.read_csv("../input/test.csv")
final_Id = test[cols[:-1] + ["Id"]]
final_test = test[cols[:-1]]
num_missing = final_test.isnull().sum()
percent = num_missing / final_test.isnull().count()

df_missing = pd.concat([num_missing, percent], axis=1, keys=['MissingValues', 'Fraction'])
df_missing = df_missing.sort_values('Fraction', ascending=False)
df_missing[df_missing['MissingValues'] > 0]
final_test.head()
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
final_test = my_imputer.fit_transform(final_test)
final_pred = model.predict(final_test)
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'Id':final_Id['Id'],'SalePrice':final_pred})

#Visualize the first 5 rows
submission.head()
filename = 'SalePricePredictions1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
