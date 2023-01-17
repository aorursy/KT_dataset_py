import pandas as pd



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()
print(set(train["wheezy-copper-turtle-magic"]))
train_1 = train.loc[train["wheezy-copper-turtle-magic"]==1,].copy()

test_1 = test.loc[test["wheezy-copper-turtle-magic"]==1,].copy()

train_1.shape
test_1.shape
train_1.head()
Y = train_1["target"]

X = train_1.drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)

X.shape
import numpy as np

from sklearn.model_selection import cross_val_score



from sklearn.ensemble import RandomForestClassifier





model = RandomForestClassifier(n_estimators=10)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
import seaborn as sns



sns.boxplot(x=scores_rf)
model = RandomForestClassifier(n_estimators=100)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
model = RandomForestClassifier(n_estimators=200)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
model = RandomForestClassifier(n_estimators=400)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
model = RandomForestClassifier(n_estimators=600)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
from sklearn import svm



model = svm.SVC()

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model
model = svm.SVC(kernel='linear')

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = svm.SVC(kernel='poly')

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = svm.SVC(kernel='poly', degree=2)

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = svm.SVC(kernel='poly', degree=4)

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = svm.SVC(kernel='poly', degree=5)

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = svm.SVC(kernel='poly', degree=4)

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')



sns.boxplot(x=scores_svm)
columns = train.columns[1:-1]



first_name = [i.split("-")[0] for i in columns]

print(set(first_name))

print(len(first_name))

print(len(set(first_name)))

for first in first_name:

    filter_col = [col for col in train_1 if col.startswith(first)]

    test_1.loc[:, first+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)

    train_1.loc[:, first+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)

    test_1.loc[:, first+"-std"] = test_1.loc[:, filter_col].std(axis=1)

    train_1.loc[:, first+"-std"] = train_1.loc[:, filter_col].std(axis=1)
second_name = [i.split("-")[1] for i in columns]

print(set(second_name))

print(len(second_name))

print(len(set(second_name)))

for second in second_name:

    filter_col = [col for col in columns if second==col.split("-")[1]]

    test_1[second+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)

    train_1[second+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)

    test_1[second+"-std"] = test_1.loc[:, filter_col].std(axis=1)

    train_1[second+"-std"] = train_1.loc[:, filter_col].std(axis=1)
third_name = [i.split("-")[2] for i in columns]

print(set(third_name))

print(len(third_name))

print(len(set(third_name)))

for third in third_name:

    filter_col = [col for col in columns if third==col.split("-")[1]]

    test_1[third+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)

    train_1[third+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)

    test_1[third+"-std"] = test_1.loc[:, filter_col].std(axis=1)

    train_1[third+"-std"] = train_1.loc[:, filter_col].std(axis=1)
fourth_name = [i.split("-")[3] for i in columns]

print(set(fourth_name))

print(len(fourth_name))

print(len(set(fourth_name)))

for fourth in fourth_name:

    filter_col = [col for col in columns if fourth==col.split("-")[1]]

    test_1[fourth+"-mean"] = test_1.loc[:, filter_col].mean(axis=1)

    train_1[fourth+"-mean"] = train_1.loc[:, filter_col].mean(axis=1)

    test_1[fourth+"-std"] = test_1.loc[:, filter_col].std(axis=1)

    train_1[fourth+"-std"] = train_1.loc[:, filter_col].std(axis=1)
for col in train_1.columns:

    if (train_1[col].isnull().sum()>0):

        train_1.drop([col], axis=1, inplace=True)

        test_1.drop([col], axis=1, inplace=True)



train_1.shape
test_1.shape
Y = train_1["target"]

X = train_1.drop(["id", "target", "wheezy-copper-turtle-magic"], axis=1)

X.shape



model = svm.SVC(kernel='poly', degree=4)

scores_svm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = RandomForestClassifier(n_estimators=400)

scores_rf = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
model = RandomForestClassifier(n_estimators=400)

model.fit(X,Y)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]



for f in range(X.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
from xgboost import XGBClassifier



model = XGBClassifier(njobs=-1)

scores_xgb = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))
model
model = XGBClassifier(learning_rate=0.01, n_estimators=1000)

scores_xgb = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))
from lightgbm import LGBMClassifier



model = LGBMClassifier(njobs=-1)

scores_gbm = cross_val_score(model, X, Y, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_gbm), "+/-", np.std(scores_gbm))
model
X2 = test_1.drop(["id", "wheezy-copper-turtle-magic"], axis=1)

X2.shape
X.shape
adv = pd.concat((X, X2), axis=0)

adv.head()
adv.shape
label = ["0"]*510+["1"]*250



model = svm.SVC(kernel='poly', degree=4)

scores_svm = cross_val_score(model, adv, label, cv=10, n_jobs=-1, scoring='roc_auc')

print(scores_svm)

print(np.mean(scores_svm), "+/-", np.std(scores_svm))
model = RandomForestClassifier(n_estimators=400)

scores_rf = cross_val_score(model, adv, label, cv=10, n_jobs=-1, scoring='roc_auc')

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
model = RandomForestClassifier(n_estimators=400).fit(adv, label)

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]



for f in range(adv.shape[1]):

    print("%d. feature %s (%f)" % (f + 1, adv.columns[indices[f]], importances[indices[f]]))
sns.distplot(adv.loc[:, "crappy-carmine-eagle-entropy"][0:510])

sns.distplot(adv.loc[:, "crappy-carmine-eagle-entropy"][510:760])