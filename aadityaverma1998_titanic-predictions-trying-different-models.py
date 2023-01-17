# sklearn for ML
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# computation libraries used
import pandas as pd
import numpy as np

#### graphing libraries ####
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
def process_age(df, cuts, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_Categories"] = pd.cut(df["Age"], cuts, labels=label_names)
    return df
cuts = [-1, 0, 5, 12, 18, 35, 60, 100]
labels = ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]
train = process_age(train, cuts, labels)
test = process_age(test, cuts, labels)
def create_dummies(df, col):
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    return df
train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")
train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")
train = create_dummies(train, "Age_Categories")
test = create_dummies(test, "Age_Categories")
train.info()
cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Age_Categories_Missing','Age_Categories_Infant', 'Age_Categories_Child', 'Age_Categories_Teenager', 'Age_Categories_Young Adult', 'Age_Categories_Adult', 'Age_Categories_Senior']

X = train[cols]
y = train["Survived"]
strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)

for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
strat_split_val = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)

for train_index, val_index in strat_split.split(X_train, y_train):
    X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
X_train, X_test, y_train, y_test = train_test_split(train[cols], train["Survived"], test_size=0.1, random_state=0)
scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
knn = KNeighborsClassifier(n_jobs=-1)
knn.fit(X_train_scaled, y_train)
print(accuracy_score(y_val, knn.predict(X_val_scaled)))
print(confusion_matrix(y_val, knn.predict(X_val_scaled)))
log_reg = LogisticRegression(n_jobs=-1)
log_reg.fit(X_train, y_train)
print(accuracy_score(y_val, log_reg.predict(X_val)))
print(confusion_matrix(y_val, log_reg.predict(X_val)))
sgd_cls = SGDClassifier(n_jobs=-1)
sgd_cls.fit(X_train_scaled, y_train)
print(accuracy_score(y_val, sgd_cls.predict(X_val_scaled)))
print(confusion_matrix(y_val, sgd_cls.predict(X_val_scaled)))
svc_cls = SVC()
svc_cls.fit(X_train_scaled, y_train)
print(accuracy_score(y_val, svc_cls.predict(X_val_scaled)))
print(confusion_matrix(y_val, svc_cls.predict(X_val_scaled)))
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print(accuracy_score(y_val, tree.predict(X_val)))
print(confusion_matrix(y_val, tree.predict(X_val)))
rf = RandomForestClassifier(n_jobs=-1, oob_score=True)
rf.fit(X_train, y_train)
print(rf.oob_score_)
print(accuracy_score(y_val, rf.predict(X_val)))
print(confusion_matrix(y_val, rf.predict(X_val)))
xgb_cls = GradientBoostingClassifier()
xgb_cls.fit(X_train, y_train)
print(accuracy_score(y_val, xgb_cls.predict(X_val)))
print(confusion_matrix(y_val, xgb_cls.predict(X_val)))
etree = ExtraTreesClassifier(oob_score=True, n_jobs=-1, bootstrap=True)
etree.fit(X_train, y_train)
print(accuracy_score(y_val, etree.predict(X_val)))
print(etree.oob_score_)
print(confusion_matrix(y_val, etree.predict(X_val)))
etree_scores = cross_val_score(etree, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(etree_scores))
print(np.std(etree_scores))
xgb_cls_scores = cross_val_score(xgb_cls, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(xgb_cls_scores))
print(np.std(xgb_cls_scores))
log_reg_scores = cross_val_score(log_reg, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(log_reg_scores))
print(np.std(log_reg_scores))
rf_scores = cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(rf_scores))
print(np.std(rf_scores))
tree_scores = cross_val_score(tree, X_train, y_train, cv=10, n_jobs=-1)
print(np.mean(tree_scores))
print(np.std(tree_scores))
dic = {
    "PassengerId" : test["PassengerId"],
    "Survived" : rf.predict(test[cols])
}
result = pd.DataFrame(dic)
result.to_csv("result.csv", index=False)
