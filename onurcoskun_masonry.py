# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("/kaggle/input/yigma-yapilar/MLYG.csv",sep=";")
test_df = pd.read_csv("/kaggle/input/yigma-yapilar/MLYT.csv",sep=";")



train_df.columns

train_df.info
def bar_plot(variable):
    """
        input: variable ex: "DD"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
    
category1 = ["DB","DT","DD","KA","YC","TDT","TKY","TYA","TS"]
for c in category1:
    bar_plot(c)
    

def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["KA"]
for n in numericVar:
    plot_hist(n)
#Yıgmalarda tek katlı yapılar yogunlukta
# DB vs TS (little relationship)
train_df[["DB","TS"]].groupby(["DB"], as_index = False).mean().sort_values(by="TS",ascending = False)
# KA vs TS (KA => Up  TS converge 1(riskli) )
train_df[["KA","TS"]].groupby(["KA"], as_index = False).mean().sort_values(by="TS",ascending = False)
# DT vs TS (little relationship)
train_df[["DT","TS"]].groupby(["DT"], as_index = False).mean().sort_values(by="TS",ascending = False)
# DD vs TS ((if dd=1 (var) => TS converge 1(riskli) )
train_df[["DD","TS"]].groupby(["DD"], as_index = False).mean().sort_values(by="TS",ascending = False)
# YC vs TS ((if YC=1 (var) => TS converge 1(riskli) )
train_df[["YC","TS"]].groupby(["YC"], as_index = False).mean().sort_values(by="TS",ascending = False)
# TDT vs TS (little relationship)
train_df[["TDT","TS"]].groupby(["TDT"], as_index = False).mean().sort_values(by="TS",ascending = False)
# TKY vs TS (TKY => Up  TS converge 1 )
train_df[["TKY","TS"]].groupby(["TKY"], as_index = False).mean().sort_values(by="TS",ascending = False)
# TYA vs TS (KA => Up  TS converge 0 )
train_df[["TYA","TS"]].groupby(["TYA"], as_index = False).mean().sort_values(by="TS",ascending = False)
train_df.columns[train_df.isnull().any()]
train_df.isnull().sum()
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
list1 = ["TKY", "TYA", "YC", "DD", "KA", "TS"]
sns.heatmap(train_df[list1].corr(), annot = True, fmt = ".2f")
plt.show()
g = sns.factorplot(x = "KA", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
g = sns.factorplot(x = "TYA", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
g = sns.factorplot(x = "YC", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
g = sns.factorplot(x = "DD", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
g = sns.FacetGrid(train_df, col = "TS", row = "YC", size = 2)
g.map(plt.hist, "DD", bins = 25)
g.add_legend()
plt.show()
g = sns.FacetGrid(train_df, col = "TS", row = "YC", size = 2)
g.map(plt.hist, "KA", bins = 25)
g.add_legend()
plt.show()
##yc=0 iken katın artmasında riskli olmuş beklemediğimiz bir durum.
g = sns.FacetGrid(train_df, row = "DT", size = 2)
g.map(sns.pointplot, "TDT","TS","TYA")
g.add_legend()
plt.show()
g = sns.factorplot(x = "KA", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
##can be categorized (KA Drop new variable KS)
train_df["KS"] = train_df["KA"].replace(["1","2","3","4","5","6","7"],"other")
train_df["KS"] = [0 if i < 2 else 1 if i < 3 else 2 if i < 5 else 3 for i in train_df["KA"]]
train_df["KS"].head(20)

g = sns.factorplot(x = "KS", y = "TS", data = train_df, kind = "bar", size = 4)
g.set_ylabels("TS Probability")
plt.show()
train_df.drop(labels = ["KA"], axis = 1, inplace = True)
train_df

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
train_df_len
test_t = train_df[train_df_len:]
testy=test_t.TS.values
test_t.drop(labels = ["TS"],axis = 1, inplace = True)
test=(test_t - np.min(test_t))/(np.max(test_t)-np.min(test_t)).values


train_t = train_df[:train_df_len]
y_train = train_t["TS"]
X_trains = train_t.drop(labels = "TS", axis = 1)
X_train=(X_trains - np.min(X_trains))/(np.max(X_trains)-np.min(X_trains)).values


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42)
print("X_train",len(X_train))
print("X_test",len(X_test))
print("y_train",len(y_train))
print("y_test",len(y_test))
print("test",len(test))
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
from sklearn.linear_model import LogisticRegression
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}  # l1 = lasso ve l2 = ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv = 10)
logreg_cv.fit(X_train,y_train)
print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)
logreg2 = LogisticRegression(C=100,penalty="l2")
logreg2.fit(X_train,y_train)
print("score: ", logreg2.score(X_test,y_test))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = logreg2, X = X_train, y= y_train, cv = 10)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))

##new test K fold CV to logisticRegression
logreg2.fit(X_train,y_train)
print("test accuracy: ",logreg2.score(test,testy))  ##new data test
logreg2.fit(X_train,y_train)
print("score: ", logreg2.score(test,testy))
y_pred = logreg_cv.predict(test)
y_true = testy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)
print("tuned hyperparameters: (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_) 

cm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
prediction=knn.predict(X_test)
print(" {} knn score: {}".format(3,knn.score(X_test, y_test)))
score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(X_train,y_train)
    score_list.append(knn2.score(X_test,y_test))
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
prediction=knn.predict(X_test)
print(" {} knn score: {}".format(14,knn.score(X_test, y_test)))
###new test
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
prediction=knn.predict(test)
print(" {} knn score: {}".format(14,knn.score(test, testy)))

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn, X = X_train, y= y_train, cv = 12)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))
##new test
knn.fit(X_train,y_train)
print("test accuracy: ",knn.score(test,testy))


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("decision tree score: ", dt.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,random_state=1)
rf.fit(X_train, y_train)
print("random forest algo result:", rf.score(X_test, y_test))

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()
#kmeans uygulama   6 cluster but we apply 2
kmeans2 = KMeans(n_clusters=2)
clusters3 = kmeans2.fit_predict(X_train)

X_train["label"] = clusters3
data=X_train
#but 7 cluster
#10 centers plot
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "yellow")
plt.show()
X_train
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(X_train,method="ward")
dendrogram(merg,leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()
from sklearn.cluster import AgglomerativeClustering
hiyerartical_cluster = AgglomerativeClustering(n_clusters = 4,affinity= "euclidean",linkage = "ward")
cluster = hiyerartical_cluster.fit_predict(X_train)
X_train["label"] = cluster
ay=X_train.label
plt.show()

ay.to_csv("ay.csv", index = False)
del X_train["label"]
random_state = 42
classifier = [DecisionTreeClassifier(random_state = random_state),
             SVC(random_state = random_state),
             RandomForestClassifier(random_state = random_state),
             LogisticRegression(random_state = random_state),
             KNeighborsClassifier()]

dt_param_grid = {"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}

svc_param_grid = {"kernel" : ["rbf"],
                 "gamma": [0.001, 0.01, 0.1, 1],
                 "C": [1,10,50,100,200,300,1000]}

rf_param_grid = {"max_features": [1,3,10],
                "min_samples_split":[2,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}

logreg_param_grid = {"C":np.logspace(-3,3,7),
                    "penalty": ["l1","l2"]}

knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),
                 "weights": ["uniform","distance"],
                 "metric":["euclidean","manhattan"]}
classifier_param = [dt_param_grid,
                   svc_param_grid,
                   rf_param_grid,
                   logreg_param_grid,
                   knn_param_grid]
cv_result = []
best_estimators = []
for i in range(len(classifier)):
    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)
    clf.fit(X_train,y_train)
    cv_result.append(clf.best_score_)
    best_estimators.append(clf.best_estimator_)
    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",
             "LogisticRegression",
             "KNeighborsClassifier"]})

g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)
g.set_xlabel("Mean Accuracy")
g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "soft", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(test),testy))
test_survived = pd.Series(votingC.predict(test), name = "TS").astype(int)
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "hard", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(test),testy))


test_survived = pd.Series(votingC.predict(test), name = "TS").astype(int)
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),
                                        ("rfc",best_estimators[2]),
                                        ("lr",best_estimators[3])],
                                        voting = "hard", n_jobs = -1)
votingC = votingC.fit(X_train, y_train)
print(accuracy_score(votingC.predict(X_test), y_test))
