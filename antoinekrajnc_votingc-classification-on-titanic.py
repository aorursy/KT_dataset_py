# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv")
dataset.head(50)
X = dataset.iloc[:,2:]
y = dataset.iloc[:, dataset.columns=="Survived"]
del X["Name"], X["Ticket"]
X["FamilySize"] = X["SibSp"] + X["Parch"]
X["IsAlone"] = X["FamilySize"].apply(lambda x:1 if x>0 else 0)
X["Cabin"] = X["Cabin"].apply(lambda x:x[0] if type(x) == str else 0)
X = pd.get_dummies(X, drop_first=True)
X.head()
X["Age"] = X["Age"].interpolate()
pd.concat([X,y], axis=1).corr()
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_x.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, np.ravel(y_train), scoring = "accuracy", cv = 10, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_res.sort_values("CrossValMeans")
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,np.ravel(y_train))

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
randF = RandomForestClassifier()

randF_param_grid = {'bootstrap': [True, False],
                     'max_depth': [10, 50, 80, None],
                     'max_features': ['auto', 'sqrt'],
                     'min_samples_leaf': [1, 4],
                     'min_samples_split': [2, 10],
                     'n_estimators': [50, 200]}

gsrandF = GridSearchCV(randF,randF_param_grid, cv=5, scoring="accuracy", n_jobs= 6, verbose = 1)
gsrandF.fit(X_train, np.ravel(y_train))

gsrandF_best = gsrandF.best_estimator_
gsrandF.best_score_
lda = LinearDiscriminantAnalysis()

lda_param_grid = {"solver": ["svd", "lsqr"]}
gslda = GridSearchCV(lda,lda_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)
gslda.fit(X_train, np.ravel(y_train))
gslda_best = gslda.best_estimator_
gslda.best_score_
#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,np.ravel(y_train))

ExtC_best = gsExtC.best_estimator_
Log = LogisticRegression()
log_param_grid = {"solver":["newton-cg", "liblinear", "sag", "saga"],
                 "class_weight": [None, "balanced"],
                 "C": [0.5, 0.6,0.7,0.8,0.9,1.0],
                 "tol": [1e-4, 1e-3]}
gsLog = GridSearchCV(Log, log_param_grid, cv =10, scoring ='accuracy', n_jobs=4, verbose=1)
gsLog.fit(X_train, np.ravel(y_train))
gsLog_best = gsLog.best_estimator_
gsLog.best_score_
nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("GradientBoosting", GBC_best),("ExtraTrees",ExtC_best),("RF",gsrandF_best),("LDA",gslda_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
test_Survived_ExtC = pd.Series(ExtC_best.predict(X_test), name="ExtC")
test_Survived_LogReg = pd.Series(gsLog_best.predict(X_test), name="Log")
test_Survived_GBC = pd.Series(GBC_best.predict(X_test), name="GBC")
test_Survived_LDA = pd.Series(gslda_best.predict(X_test), name="LDA")
test_Survived_RF = pd.Series(gsrandF_best.predict(X_test), name="RF")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_ExtC,test_Survived_GBC, test_Survived_LogReg,test_Survived_LDA, test_Survived_RF],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[("GradientBoosting", GBC_best),("ExtraTrees",ExtC_best),("RF",gsrandF_best),("LDA",gslda_best), ("LogReg",gsLog_best)], 
                           voting='hard', 
                           n_jobs=4)

votingC = votingC.fit(X_train, np.ravel(y_train))
cross_val_score(votingC, X_train, np.ravel(y_train), scoring = "accuracy", cv = 10, n_jobs=4).mean()
test = pd.read_csv("../input/test.csv")
X = test.iloc[:,1:]
del X["Name"], X["Ticket"]
X["FamilySize"] = X["SibSp"] + X["Parch"]
X["IsAlone"] = X["FamilySize"].apply(lambda x:1 if x>0 else 0)
X["Cabin"] = X["Cabin"].apply(lambda x:x[0] if type(x) == str else 0)
X["Age"] = X["Age"].interpolate()
X["Fare"] = X["Fare"].interpolate()
X = pd.get_dummies(X, drop_first=True)
X.insert(15, "Cabin_T", 0)
X.head()
X.isnull().any()
sc_x.fit_transform(X)
y_pred = votingC.predict(X)
submission = pd.concat([test.iloc[:,0], pd.DataFrame(y_pred, columns=["Survived"])], axis=1)
submission.to_csv("submission.csv", index=False)
