import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

df
#drop unnecessary features

df.drop(["Unnamed: 32","id"],axis=1,inplace=True)
df.describe()
df.groupby("diagnosis").mean()
plt.figure(figsize=(20,20))

sns.heatmap(df.corr(),annot=True,linewidths=.5)
corr_matrix = df.corr().abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find features with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]



# Drop features 

df =  df.drop(to_drop, axis=1)

df.shape
x = df.drop(["diagnosis"],axis=1)

y = df["diagnosis"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
xgb = XGBClassifier(random_state=42)

xgb.fit(x_train,y_train)

preds = xgb.predict(x_test)

print("Test acc",accuracy_score(y_test,preds))
score = cross_val_score(xgb,x_train,y_train,cv=5).mean()

print("cross_val score",score)
xgb = XGBClassifier(random_state=42)

params = {

    "max_depth" : [3,4,5,7,10],

    "learning_rate" : [0.1,0.15,0.2,0.25,0.3],

    "colsample_bytree" : np.arange(0.3,1,0.1),

    "subsample": np.arange(0.3,1,0.1),

    "gamma" : [0,0.1,0.2]

}

grid_search = GridSearchCV(xgb,params,cv=5)

best = grid_search.fit(x_train,y_train)
best.best_params_
xgb = XGBClassifier(random_state=42,colsample_bytree=0.4,learning_rate=0.3,max_depth=3,subsample=0.4,gamma=0.1)

xgb.fit(x_train,y_train)

testpreds = xgb.predict(x_test)

trainpreds = xgb.predict(x_train)

print("Test acc",accuracy_score(y_test,testpreds))

print("Train acc",accuracy_score(y_train,trainpreds))
score2 = cross_val_score(xgb,x_train,y_train,cv=5).mean()

print("First cross_val score",score)

print("cross_val score after tuning",score2)
sns.heatmap(confusion_matrix(y_test,testpreds),annot=True)
print(classification_report(y_test,testpreds))