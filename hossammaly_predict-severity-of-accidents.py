# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style
users_df = pd.read_csv("../input/users.csv")
places_df = pd.read_csv("../input/places.csv")
caracteristics_df = pd.read_csv("../input/caracteristics.csv", encoding="latin-1")
users_df.describe()
users_df.info()
places_df.describe()
caracteristics_df.describe()
users_df["Num_Acc"].count()
places_df["Num_Acc"].count()
caracteristics_df["Num_Acc"].count()
users_df.head()
users_df = users_df.drop(["place", "catu", "sexe", "trajet", "secu", "locp", "actp", "etatp", "an_nais", "num_veh"], axis=1)



users_df.describe()
users_df.grav.hist()
users_df = users_df.replace(1,0)

users_df = users_df.replace(3,1)

users_df = users_df.replace(4,0)

users_df = users_df.replace(2,1)



users_df.grav.hist()
users_df = users_df.groupby(['Num_Acc'], as_index=False).max()



users_df.grav.hist()
users_df.count()
users_df.head(15)
places_df.describe()
places_df = places_df.drop(["v1", "voie", "v2", "pr", "pr1", "vosp", "prof", "plan", "lartpc", "larrout", "situ", "env1"], axis=1)



places_df.describe()
places_df.info()
places_df["catr"].isna().sum()
places_df["catr"] = places_df["catr"].fillna(9)
places_df["catr"] = places_df["catr"].astype(int)
places_df["circ"].isna().sum()
(places_df["circ"] == 0).sum()
places_df["circ"].hist()
places_df["circ"] = places_df["circ"].fillna(2)

places_df["circ"] = places_df["circ"].replace(0,2)
places_df["circ"] = places_df["circ"].astype(int)
places_df["nbv"].isna().sum()
places_df["nbv"] = places_df["nbv"].where(places_df["nbv"] < 6,0)
(places_df["nbv"] == 0).sum()
places_df["nbv"].hist()
places_df["nbv"] = places_df["nbv"].fillna(2)

places_df["nbv"] = places_df["nbv"].replace(0,2)
places_df["nbv"] = places_df["nbv"].astype(int)
places_df["surf"].isna().sum()
(places_df["surf"] == 0).sum()
places_df["surf"].hist()
(places_df["surf"] == 1).sum()
places_df["surf"] = places_df["surf"].fillna(1)

places_df["surf"] = places_df["surf"].replace(0,1)
places_df["surf"] = places_df["surf"].astype(int)
(places_df["infra"] == 0).sum()
places_df = places_df.drop(["infra"], axis=1)
places_df.info()
places_df.describe()
caracteristics_df.describe()
caracteristics_df = caracteristics_df.drop(["an", "col", "com", "adr", "gps", "lat", "long"], axis=1)



caracteristics_df.describe()
caracteristics_df.info()
caracteristics_df["atm"].isna().sum()
caracteristics_df["atm"] = caracteristics_df["atm"].fillna(1)
caracteristics_df["atm"] = caracteristics_df["atm"].astype(int)
(caracteristics_df["int"] == 0).sum()
caracteristics_df["int"] = caracteristics_df["int"].replace(0,1)
caracteristics_df["hrmn"] = caracteristics_df["hrmn"].div(100).apply(np.floor)
caracteristics_df["hrmn"] = caracteristics_df["hrmn"].astype(int)
caracteristics_df["hrmn"].hist()
caracteristics_df["dep"] = caracteristics_df["dep"].div(10).apply(np.floor)
caracteristics_df["dep"] = caracteristics_df["dep"].astype(int)
caracteristics_df.info()
caracteristics_df.describe()
datamerge = pd.merge(caracteristics_df, places_df, how="outer", on="Num_Acc")
datamerge.describe()
data = pd.merge(datamerge, users_df, how="outer", on="Num_Acc")
data = data.drop("Num_Acc", axis=1)
data.info()
data.describe()
data.head(15)
#data.to_csv("data.csv")
from sklearn.model_selection import train_test_split



X_training, X_test, y_training, y_test = train_test_split(data.drop("grav", axis=1), data["grav"], test_size=0.2, random_state=42)
X_train, X_Val, y_train, y_Val = train_test_split(X_training, y_training, test_size=0.2, random_state=42)
import time

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

print('Time taken :' , time.time()-t0)
y_pred = model_rf.predict(X_Val)

score_rf = accuracy_score(y_Val,y_pred)

print('Accuracy :',score_rf)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model_rf.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances
X_train = X_train.drop(["surf", "lum", "atm", "int", "nbv", "circ"], axis=1)

X_Val = X_Val.drop(["surf", "lum", "atm", "int", "nbv", "circ"], axis=1)

X_test = X_test.drop(["surf", "lum", "atm", "int", "nbv", "circ"], axis=1)
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=100,random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_Val)

score = accuracy_score(y_Val,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=100, max_depth= 5, max_features= 3, random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_Val)

score = accuracy_score(y_Val,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=50, max_depth= 5, max_features= 3, random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_Val)

score = accuracy_score(y_Val,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=7, max_depth= 15, max_features= 6, random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_Val)

score = accuracy_score(y_Val,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
t0=time.time()

model_rf = RandomForestClassifier(n_estimators=7, max_depth= 15, max_features= 6, random_state=0, n_jobs=-1)

model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
y_train.hist()
y_scores = model_rf.predict_proba(X_test)
y_scores = y_scores[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_scores)
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):

    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'r', linewidth=4)

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate (FPR)', fontsize=16)

    plt.ylabel('True Positive Rate (TPR)', fontsize=16)



plt.figure(figsize=(14, 7))

plot_roc_curve(false_positive_rate, true_positive_rate)

plt.show()
r_a_score = roc_auc_score(y_test, y_scores)
r_a_score
model_rf.predict(np.array([[12, 31, 18, 1, 13, 3]]))
model_rf.predict(np.array([[7, 5, 6, 2, 26, 6]]))