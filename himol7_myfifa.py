import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style = "whitegrid")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
data = pd.read_csv("../input/FIFA 2018 Statistics.csv")
data.shape
data.head(2)
gc = np.array([], dtype = 'int')
for i in range(0, 128, 2):
    gc = np.append(gc, data.loc[i+1, "Goal Scored"])
    gc = np.append(gc, data.loc[i, "Goal Scored"])
    
data.insert(4, "Goal Conceded", pd.Series(gc))
data.head(2)
conditions = [(data["Goal Scored"] > data["Goal Conceded"]), (data["Goal Scored"] == data["Goal Conceded"]), (data["Goal Scored"] < data["Goal Conceded"])]
result = np.array([0, 1, 2], dtype = 'int')
data.insert(5, "Result", pd.Series(np.select(conditions, result, default = -1)))
data.head(2)
data.insert(15, "Total Set Pieces", pd.Series(data["Corners"] + data["Free Kicks"], dtype = 'int'))

data.head(2)
sns.lmplot(x = 'Ball Possession %', y = 'Attempts', data = data)
data["Ball Possession %"].corr(data["Attempts"], method = 'pearson')
sns.lmplot(x = 'Passes', y = 'Distance Covered (Kms)', data = data)
data["Passes"].corr(data["Distance Covered (Kms)"], method = 'pearson')
sns.set_context("paper")
sns.swarmplot( x = 'Result', y = 'Total Set Pieces', data = data)
data.isna().sum()
data[['Own goals', 'Own goal Time']].head()
data[['Own goals', 'Own goal Time']] = data[['Own goals', 'Own goal Time']].fillna(0)
data.isna().sum()
data["1st Goal"].head()
data["1st Goal"] = data["1st Goal"].fillna(90)
data.isna().sum()
data.dtypes.unique()
cat = data.columns.values[data.dtypes == object]
cat
data.drop(["Date"], axis = 1, inplace = True)

temp = np.array(['Man of the Match', 'Round', 'PSO'])
for i in range(0,len(temp)):
    x = temp[i]
    data[x] = data[x].astype('category').cat.codes
    
data.head(2)
# I am currently working on the above mentioned approach.
# This is a temporary solution.

data["Team"] = data["Team"].astype('category').cat.codes
data["Opponent"] = data["Opponent"].astype('category').cat.codes
data.head()
features = data.drop(["Man of the Match"], axis = 1)
target = data["Man of the Match"]
modelxgb = XGBClassifier()
modelxgb.fit(features, target)

print(modelxgb.feature_importances_)
from xgboost import plot_importance
plot_importance(modelxgb)
f_xgb = pd.DataFrame(data = {'feature' : features.columns, 'value' : modelxgb.feature_importances_})
f_xgb = f_xgb.sort_values(['value'], ascending = False)
top10xgb = f_xgb.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10xgb["feature"], y = top10xgb["value"])
modellgbm = LGBMClassifier()
modellgbm.fit(features, target)

print(modellgbm.feature_importances_)
f_lgbm = pd.DataFrame(data = {'feature' : features.columns, 'value' : modellgbm.feature_importances_})
f_lgbm = f_lgbm.sort_values(['value'], ascending = False)
top10lgbm = f_lgbm.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10lgbm["feature"], y = top10lgbm["value"])
modeletc = ExtraTreesClassifier()
modeletc.fit(features, target)

print(modeletc.feature_importances_)
f_etc = pd.DataFrame(data = {'feature' : features.columns, 'value' : modeletc.feature_importances_})
f_etc = f_etc.sort_values(['value'], ascending = False)
top10etc = f_etc.head(10)
plt.figure(figsize=(15,8))
sns.barplot(x = top10etc["feature"], y = top10etc["value"])
ft = pd.merge(f_xgb, f_lgbm, how = 'inner', on = ["feature"])
ft = pd.merge(ft, f_etc, how = 'inner', on = ["feature"])
ft.head(5)
features = ft["feature"].head(5).values
X = data[features]
Y = target.values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 7)
modelsvc = svm.SVC(kernel = 'rbf', gamma='auto')
modelsvc.fit(X_train, y_train)
y_svc = modelsvc.predict(X_test)
accuracy_score(y_test, y_svc)
modelreg = linear_model.LogisticRegression()
modelreg.fit(X_train, y_train)
y_reg = modelreg.predict(X_test)
accuracy_score(y_test, y_reg.round(), normalize = False)
modelrf = RandomForestClassifier(max_depth=2, random_state=0)
modelrf.fit(X_train, y_train)
y_rf = modelrf.predict(X_test)
accuracy_score(y_test, y_rf)
