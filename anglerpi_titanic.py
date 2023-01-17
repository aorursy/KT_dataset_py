import pandas as pd
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
print(train_df.columns.values)
train_df.head()
train_df.tail()
train_df.info()
test_df.info()
train_df.describe()
train_df.describe(include=["O"])
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Sex", "Survived"]].groupby(["Sex"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["SibSp", "Survived"]].groupby(["SibSp"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[["Parch", "Survived"]].groupby(["Parch"], as_index=False).mean().sort_values(by="Survived", ascending=False)
selected_features=["Pclass","Sex","Age","Embarked","Fare"]
X_train=train_df[selected_features]
X_test=test_df[selected_features]
y_train=train_df["Survived"]
print (X_train["Embarked"].value_counts())
print (X_test["Embarked"].value_counts())
X_train["Embarked"].fillna("S",inplace=True)
X_test["Embarked"].fillna("S",inplace=True)
X_train["Age"].fillna(X_train["Age"].mean(),inplace=True)
X_test["Age"].fillna(X_test["Age"].mean(),inplace=True)
X_train.info()
X_test.info()
X_test["Fare"].fillna(X_test["Fare"].mean(),inplace=True)
X_test.info()
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer()
X_train=dict_vec.fit_transform(X_train.to_dict(orient="record"))
X_test=dict_vec.fit_transform(X_test.to_dict(orient="record"))

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_lr_pred=lr.predict(X_test)
lr.score(X_train,y_train)
from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_lsvc_pred=lsvc.predict(X_test)
lsvc.score(X_train,y_train)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)
knn.score(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_dtc_pred=dtc.predict(X_test)
dtc.score(X_train,y_train)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
y_rfc_pred=rfc.predict(X_test)
rfc.score(X_train,y_train)
from sklearn.linear_model import SGDClassifier
sgdc=SGDClassifier()
sgdc.fit(X_train,y_train)
y_sgdc_pred=sgdc.predict(X_test)
sgdc.score(X_train,y_train)
from xgboost import XGBClassifier
xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
y_xgbc_pred=xgbc.predict(X_test)
xgbc.score(X_train,y_train)
from sklearn.cross_validation import cross_val_score
cross_val_score(dtc,X_train,y_train,cv=5).mean()
cross_val_score(rfc,X_train,y_train,cv=5).mean()
cross_val_score(xgbc,X_train,y_train,cv=5).mean()
from sklearn.model_selection import GridSearchCV
params={"max_depth":range(2,7),"n_estimators":range(100,1100,200),"learning_rate":[0.25,0.1,0.25,0.5,1.0]}
xgbc_best=XGBClassifier()
gs=GridSearchCV(xgbc_best,params,n_jobs=-1,cv=5,verbose=1)
gs.fit(X_train,y_train)
print (gs.best_score_)
print (gs.best_params_)
xgbc_best_y_predict=gs.predict(X_test)
gs.score(X_train,y_train)