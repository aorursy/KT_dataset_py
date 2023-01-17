import pandas as pd

import numpy as np



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
#Load the data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



IDTest = test["PassengerId"]
def detect_outliers(df,n,features):

    outlier_indices = []

    

    for col in features:

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col],75)

        IQR = Q3 - Q1

    

        outlier_step = 1.5 * IQR

        

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

dataset =  pd.concat(objs=[train, test], axis=0, sort = False).reset_index(drop=True)
dataset = dataset.fillna(np.nan)



dataset.isnull().sum()
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
dataset["Embarked"] = dataset["Embarked"].fillna("S")
nan_vals = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in nan_vals :

    med_age = dataset["Age"].median()

    common_ = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(common_) :

        dataset['Age'].iloc[i] = common_

    else:

        dataset['Age'].iloc[i] = med_age
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
dataset["fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
dataset['SingleF'] = dataset['fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['fsize'].map(lambda s: 1 if s >= 5 else 0)
dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
off_train = len(train)



train = dataset[:off_train]

test = dataset[off_train:]



#Removing the target column, created when we merge the data.

test.drop(labels=["Survived"],axis = 1,inplace=True)
train["Survived"] = train["Survived"].astype(int)



Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
kfold = StratifiedKFold(n_splits=10)



random_state = 42
clsf = []

clsf.append(DecisionTreeClassifier(random_state=random_state))

clsf.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

clsf.append(RandomForestClassifier(random_state=random_state))

clsf.append(GradientBoostingClassifier(random_state=random_state))

clsf.append(KNeighborsClassifier())

clsf.append(SVC(random_state=random_state))

clsf.append(ExtraTreesClassifier(random_state=random_state))

cv_results = []

for classifier in clsf :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())
models = ["DecisionTree","AdaBoost","RandomForest","GradientBoosting","KNeighboors", "SVM", "ExtraTrees"]





for e in range(7):

    print("The mean result of ", models[e], " is: ", cv_means[e])

# Adaboost

dtc = DecisionTreeClassifier()



ada = AdaBoostClassifier(dtc, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "n_estimators" :[1, 2, 3, 4, 6, 8, 10],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsada = GridSearchCV(ada,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsada.fit(X_train,Y_train)



ada_best = gsada.best_estimator_



gsada.best_score_
RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 5, 7, 8, 10],

              "min_samples_split": [2, 3, 5, 7, 10],

              "min_samples_leaf": [1, 3, 5, 7, 9, 10],

              "bootstrap": [False],

              "n_estimators" :[100, 200, 300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,Y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100, 150, 200, 250, 300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 5, 6, 7, 8],

              'min_samples_leaf': [100,125,175, 150],

              'max_features': [0.3, 0.25,  0.15, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100, 200, 300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,Y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
ExtC = ExtraTreesClassifier()



ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train,Y_train)



ExtC_best = gsExtC.best_estimator_



gsExtC.best_score_
ensemble_ = VotingClassifier(estimators=[('rfc', RFC_best), ('adac',ada_best),('gbc',GBC_best), ('extc', ExtC_best),

('svc', SVMC_best)], voting='soft', n_jobs=4)



predict_baby = ensemble_.fit(X_train, Y_train)
predictions = ensemble_.predict(test)



output = pd.DataFrame({'PassengerId': IDTest, 'Survived': predictions})

output.to_csv('submission.csv', index=False)