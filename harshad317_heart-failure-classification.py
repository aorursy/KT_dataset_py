# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

sns.set(style='white', context='notebook', palette='deep')
dataset = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

dataset.head()
col = dataset.columns
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(dataset,2,[col])
dataset.loc[Outliers_to_drop]
dataset.isnull().sum()
plt.figure(figsize= (15,10))

sns.heatmap(dataset.corr(), annot = True, fmt = '.2', cmap = 'coolwarm')
dataset.head()
g = sns.distplot(dataset["ejection_fraction"], color="m", label="Skewness : %.2f"%(dataset["ejection_fraction"].skew()))

g = g.legend(loc="best")
dataset["ejection_fraction"] = dataset["ejection_fraction"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["ejection_fraction"], color="m", label="Skewness : %.2f"%(dataset["ejection_fraction"].skew()))

g = g.legend(loc="best")
g = sns.distplot(dataset["creatinine_phosphokinase"], color="m", label="Skewness : %.2f"%(dataset["creatinine_phosphokinase"].skew()))

g = g.legend(loc="best")
dataset["creatinine_phosphokinase"] = dataset["creatinine_phosphokinase"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(dataset["creatinine_phosphokinase"], color="m", label="Skewness : %.2f"%(dataset["creatinine_phosphokinase"].skew()))

g = g.legend(loc="best")
g = sns.distplot(dataset["platelets"], color="m", label="Skewness : %.2f"%(dataset["platelets"].skew()))

g = g.legend(loc="best")
g = sns.distplot(dataset["serum_creatinine"], color="m", label="Skewness : %.2f"%(dataset["serum_creatinine"].skew()))

g = g.legend(loc="best")
dataset["serum_creatinine"] = dataset["serum_creatinine"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(dataset["serum_creatinine"], color="m", label="Skewness : %.2f"%(dataset["serum_creatinine"].skew()))

g = g.legend(loc="best")
g = sns.distplot(dataset["serum_sodium"], color="m", label="Skewness : %.2f"%(dataset["serum_sodium"].skew()))

g = g.legend(loc="best")
g = sns.distplot(dataset["time"], color="m", label="Skewness : %.2f"%(dataset["time"].skew()))

g = g.legend(loc="best")
x = dataset.drop('DEATH_EVENT', axis = 1)

y = dataset['DEATH_EVENT']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

sc.fit(x_train)

x_train = sc.transform(x_train)

x_test = sc.transform(x_test)
kfold = StratifiedKFold(n_splits=10)
from xgboost import XGBClassifier
random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(CatBoostClassifier(iterations = 2000, depth = 3))

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(XGBClassifier(random_state = random_state))



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","CatBoostClassifier","LogisticRegression","XGBClassifier"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(x_train,y_train)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsExtC.fit(x_train,y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsRFC.fit(x_train,y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300, 400, 500],

              'learning_rate': [0.1, 0.05, 0.01, 0.001],

              'max_depth': [3, 4, 5, 8],

              'min_samples_leaf': [100,150, 200],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(x_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsSVMC.fit(x_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
cat = CatBoostClassifier()

cat_param_grid = {'iterations': [1000, 2000, 3000],

                 'depth': [3, 5, 7],

                 'learning_rate': [0.1, 0.01, 0.05, 0.001]}



gscat = GridSearchCV(cat,param_grid = cat_param_grid, cv= 5, scoring="accuracy", n_jobs= -1, verbose = 1)



gscat.fit(x_train, y_train)

cat_best = gscat.best_estimator_

                  

gscat.best_score_
def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",x_train,y_train,cv=kfold)

g = plot_learning_curve(gscat.best_estimator_,"CatBoost learning curves",x_train,y_train,cv=kfold)
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best), ('cat', cat_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(x_train, y_train)
test = pd.Series(votingC.predict(x_test), name="DEATH_EVENT")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test, test)
cm = confusion_matrix(y_test, test)

plt.figure(figsize=(5,5))

sns.heatmap(cm,annot = True, cmap=plt.cm.Blues)

plt.title("Random Forest Model - Confusion Matrix")

plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)

plt.show()
test.to_csv("voting.csv",index=False)