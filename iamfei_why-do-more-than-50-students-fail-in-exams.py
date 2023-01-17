import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn import tree



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn import svm



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
data_mat=pd.read_csv("../input/student-mat.csv")

data_mat["course"]="math"

data_por=pd.read_csv("../input/student-por.csv")

data_por["course"]="portuguese"

data=data_mat.append(data_por)
data.info()

data.head(3)
data["Gmean"]=data.loc[:,("G1","G2","G3")].mean(1)

data["Glevel"]=np.where(data["Gmean"]<12,'fail','pass')

data["Glevel"]=np.where(data["Gmean"]>16,'good',data['Glevel'])



del data["G1"]

del data["G2"]

del data["G3"]
for i in data.columns:

    print("=====Attr:",i,"=====")

    print(100*data[i].value_counts()/len(data[i]))

    print("\n")
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.boxplot(x="course",y="Gmean",data=data)

plt.subplot(222)

sns.boxplot(x="school",y="Gmean",data=data)

plt.subplot(223)

sns.boxplot(x="course",y="Gmean",hue="school",data=data)

plt.subplot(224)

sns.boxplot(x="school",y="Gmean",hue="course",data=data)

plt.show()
plt.figure(figsize=(12,12))

plt.subplot(221)

sns.violinplot(x="Dalc",y="Gmean",data=data)

plt.subplot(222)

sns.swarmplot(x="Walc",y="Gmean",data=data)

plt.subplot(223)

sns.boxplot(x="Dalc",y="Gmean",hue="course",data=data)

plt.subplot(224)

sns.boxplot(x="Walc",y="Gmean",hue="course",data=data)

plt.show()
print("=====For math course=====")

agegrade=pd.crosstab(data["age"][data["course"]=="math"],data["Glevel"][data["course"]=="math"])

agegrade["sum"]=agegrade.sum(1)

agegrade["fail%"]=agegrade["fail"]/agegrade["sum"]*100

agegrade["good%"]=agegrade["good"]/agegrade["sum"]*100

agegrade["pass%"]=agegrade["pass"]/agegrade["sum"]*100

del agegrade["fail"]

del agegrade["good"]

del agegrade["pass"]

del agegrade["sum"]

print(agegrade)

print("\n")



print("=====For portuguese course=====")

agegrade=pd.crosstab(data["age"][data["course"]=="portuguese"],data["Glevel"][data["course"]=="portuguese"])

agegrade["sum"]=agegrade.sum(1)

agegrade["fail%"]=agegrade["fail"]/agegrade["sum"]*100

agegrade["good%"]=agegrade["good"]/agegrade["sum"]*100

agegrade["pass%"]=agegrade["pass"]/agegrade["sum"]*100

del agegrade["fail"]

del agegrade["good"]

del agegrade["pass"]

del agegrade["sum"]

print(agegrade)
#from sklearn.preprocessing import LabelEncoder

#labelencoder=LabelEncoder()

#for col in ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", 

#        "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "course","Glevel"]:

#    data[col] = labelencoder.fit_transform(data[col])



for i in ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup",

          "paid","activities","nursery","higher","internet","romantic","course","Glevel"]:

    data[i]=data[i].astype("category")

    data[i].cat.categories=range(0,len(data[i].unique()),1)

    data[i]=data[i].astype("int")



data.head()
cor=data.corr()

plt.figure(figsize=(12,12))

sns.heatmap(cor,annot=False)

plt.show()
trainfeatures=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Walc", "health", "absences", "course","Glevel"]
cor["Glevel"].drop("Glevel").drop("Gmean").sort_values()
training, testing = train_test_split(data[trainfeatures], test_size=0.3, random_state=0)



X=training.iloc[:,0:27]

y=training.iloc[:,27]

xtest=testing.iloc[:,0:27]

ytest=testing.iloc[:,27]
## Logistic Regression

clf_log = LogisticRegression()

clf_log = clf_log.fit(X,y)

score_log = cross_val_score(clf_log, xtest, ytest, cv=5).mean()

print(score_log)
## Perceptron

clf_pctr = Perceptron(

    class_weight='balanced'

    )

clf_pctr = clf_pctr.fit(X,y)

score_pctr = cross_val_score(clf_pctr, xtest, ytest, cv=5).mean()

print(score_pctr)
## Kneighbor

clf_knn = KNeighborsClassifier(

    n_neighbors=10,

    weights='distance'

    )

clf_knn = clf_knn.fit(X,y)

score_knn = cross_val_score(clf_knn, xtest, ytest, cv=5).mean()

print(score_knn)
## SVM

clf_svm = svm.SVC(

    class_weight='balanced'

    )

clf_svm.fit(X, y)

score_svm = cross_val_score(clf_svm, xtest, ytest, cv=5).mean()

print(score_svm)
## Bagging

bagging = BaggingClassifier(

    KNeighborsClassifier(

        n_neighbors=5,

        weights='distance'

        ),

    oob_score=True,

    max_samples=0.5,

    max_features=1.0

    )

clf_bag = bagging.fit(X,y)

score_bag = clf_bag.oob_score_

print(score_bag)
## Decision Tree

clf_tree = tree.DecisionTreeClassifier(

    #max_depth=3,\

    class_weight="balanced",\

    min_weight_fraction_leaf=0.01\

    )

clf_tree = clf_tree.fit(X,y)

score_tree = cross_val_score(clf_tree, xtest, ytest, cv=5).mean()

print(score_tree)
## Random Forest

clf_rf = RandomForestClassifier(

    n_estimators=1000, \

    n_jobs=-1

    )

clf_rf = clf_rf.fit(X,y)

score_rf = cross_val_score(clf_rf, xtest, ytest, cv=5).mean()

print(score_rf)
## Extra Tree

clf_ext = ExtraTreesClassifier(

    max_features='auto',

    bootstrap=True,

    oob_score=True,

    n_estimators=1000,

    max_depth=None,

    min_samples_split=10

    #class_weight="balanced",

    #min_weight_fraction_leaf=0.02

    )

clf_ext = clf_ext.fit(X,y)

score_ext = cross_val_score(clf_ext, xtest, ytest, cv=5).mean()

print(score_ext)
## Summary of each classifier

odels = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',

              'Perceptron','BaggingClassifier','Random Forest','Decision Tree','Extra Tree'],

    'Score': [score_svm, score_knn, score_log,score_pctr, score_bag,score_rf, score_tree, score_ext]})

print(odels.sort_values("Score",ascending=False))
## Importance of each features

importances = clf_ext.feature_importances_

features = data.columns[0:31]

sort_indices = np.argsort(importances)[::-1]

sorted_features = []

for idx in sort_indices:

    sorted_features.append(features[idx])

plt.figure()

plt.figure(figsize=(14,14))

plt.bar(range(len(importances)), importances[sort_indices], align='center');

plt.xticks(range(len(importances)), sorted_features, rotation='vertical');

plt.xlim([-1, len(importances)])

plt.grid(False)

plt.show()



result=pd.DataFrame({'factor':sorted_features,'weight':importances[sort_indices]})

print(result.sort_values("weight",ascending=False))