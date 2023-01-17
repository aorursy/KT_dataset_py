import pandas as pd
data = pd.read_csv("../input//WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.shape
data.head()
tmp = []
for i in data.columns:
    tmp.append([i, len(data[i].unique())])
    
pd.DataFrame(tmp, columns=["category", "count"]).sort_values(by="count", ascending=False)
data.info()
tmp = sorted(data["TotalCharges"].unique())
print(tmp[:5], tmp[-5:])
(data["TotalCharges"]==" ").sum()
data[data["TotalCharges"] == " "][["tenure", "MonthlyCharges", "TotalCharges"]]
data["TotalCharges"] = data["TotalCharges"].replace(" ", "0").astype(float)
data[data["tenure"] == 0][["tenure", "MonthlyCharges", "TotalCharges"]]
data.info()
tmp = []
for i in data.columns:
    if len(data[i].unique()) < 10:
        tmp.append([i] + data[i].unique().tolist())
        
pd.DataFrame(tmp).fillna("-----")
feature = pd.get_dummies(data.drop("customerID", axis=1))
feature["TotalCharges"] = feature["TotalCharges"] / feature["TotalCharges"].max()
feature["MonthlyCharges"] = feature["MonthlyCharges"] / feature["MonthlyCharges"].max()
feature["tenure"] = feature["tenure"] / feature["tenure"].max()
feature.shape
feature.head()
corr = feature.corr()
corr["Churn_Yes"].plot(kind="bar", figsize=(15, 5), color="royalblue")
corr["Churn_Yes"].sort_values(ascending=False).plot(kind="bar", figsize=(15, 5), color="royalblue")
from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=5, n_components=2)
pos = iso.fit_transform(feature.T)
import matplotlib.pyplot as plt
from random import random as rand

plt.figure(figsize=(20, 20))

tmp = []

cnt = 0
for x, y, t in zip(pos[:, 0], pos[:, 1], feature.columns.tolist()):
    ry = rand()*10
    rx = abs(ry) 
    plt.scatter(x, y)
    if "Churn" in t:
        plt.text(x=x+rx, y=y+ry, s=t, color="red", fontsize=20)
        tmp.append((x, y))
    elif ("Charges" in t) or ("tenure" in t):
        plt.text(x=x+rx, y=y+ry, s=t, color="green", fontsize=15)
    else:
        plt.text(x=x+rx, y=y+ry, s=t)
    plt.plot([x, x+rx], [y, y+ry])

a = -1 / ((tmp[1][1] - tmp[0][1]) / (tmp[1][0] - tmp[0][0]))
b =  (tmp[1][0] + tmp[0][0]) /2, (tmp[1][1] + tmp[0][1]) / 2
plt.plot([b[0]+50, b[0]-150], [b[1]+50*a, b[1]-150*a], linestyle="--")

tmp = []
for i in feature.columns:
    if "_No" not in i:
        tmp.append(i)

feature2 = feature.copy()
feature2 = feature2[tmp]

X = feature2.drop("Churn_Yes", axis=1)
y = feature2["Churn_Yes"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train, y_train)
svc.score(X_test, y_test)
pred = svc.predict(X_test)

result = pd.DataFrame(pred, columns=["pred"])
result["y_test"] = y_test.values

result["correct"] = result["pred"] == result["y_test"]

result = pd.concat([result, X_test.reset_index(drop=True)], axis=1)
tmp = 0
for i in range(0, 20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    score = svc.score(X_test, y_test)
    tmp += score
    print("random seed: ", i, ",", round(score, 4), ", average: ", round(tmp/(i+1), 4))

result[result["correct"]==True].drop("pred", axis=1).drop("y_test", axis=1).drop("correct", axis=1).hist(figsize=(15, 15))
result[result["correct"]==False].drop("pred", axis=1).drop("y_test", axis=1).drop("correct", axis=1).hist(figsize=(15, 15))
