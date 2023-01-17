import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
df.shape
df.columns
df.head()
df.drop_duplicates(inplace=True)
df.shape
Catagorical_Features = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
sns.violinplot(x=df['Exited'], y=df['CreditScore'])
sns.boxplot(x=df['Exited'], y=df['CreditScore'])
sns.barplot(x=df['Geography'], y=df['Exited'])
sns.barplot(x=df['Gender'], y=df['Exited'])
sns.violinplot(x=df['Exited'], y=df['Age'])
sns.boxplot(x=df['Exited'], y=df['Age'])
sns.barplot(x=df['Tenure'], y=df['Exited'])
sns.kdeplot(data=df['Balance'],shade=True)
sns.violinplot(x='Exited', y='Balance', data=df)
sns.barplot(x=df['NumOfProducts'], y=df['Exited'])
sns.barplot(x='HasCrCard', y='Exited', data=df)
sns.barplot(x = df['IsActiveMember'], y= df['Exited'])
sns.boxplot(x=df['Exited'], y=df['EstimatedSalary'])
sns.violinplot(x=df['Exited'], y=df['EstimatedSalary'])
df.head(5)
df = pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)
df.head(5)
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'],inplace=True)
df.head()
df.shape
def person_corr(df):

    df_dup = df.copy()

    df_corr = df.corr() # Find Correlation of dataframe

    col_name = df_corr.columns

    col = list()

    for i in df_corr:

        for j in col_name:

            if (df_corr[i][j]>0.0) & (i!=j) & (i not in col): # set threshold 0.85

                col.append(j)

    df_dup.drop(columns=col,inplace=True)

    return df_dup
df_diff_col = person_corr(df)
df.corr()
X_train = df.drop(columns=['Exited'])

y_train = df['Exited']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
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

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res)

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")

#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {

#               "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[50,100,200,300],

              "criterion": ["gini"]}

etc_folds = []

etcc = []

for i in range(5,18,2):



    kfold =i

    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



    gsExtC.fit(X_train,y_train)



    ExtC_best = gsExtC.best_estimator_

    etc_folds.append(gsExtC.best_score_)

    etcc.append(ExtC_best)

# Best score

gsExtC.best_score_

# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {

#               "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[50,100,200,300],

              "criterion": ["gini"]}



rfc_folds =[]

rfcc = []

for i in range(5,18,2):

    kfold = i



    gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



    gsRFC.fit(X_train,y_train)



    RFC_best = gsRFC.best_estimator_

    

    rfc_folds.append(gsRFC.best_score_)

    rfcc.append(RFC_best)

# Best score

gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [50,100,200,300,400],

              'learning_rate': [0.1, 0.05, 0.01,10],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }

gbdt_folds = []

gbdtt = []

for i in range(3,10,2):

    kfold = i

    gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



    gsGBC.fit(X_train,y_train)



    GBC_best = gsGBC.best_estimator_

    

    gbdt_folds.append(gsGBC.best_score_)

    gbdtt.append(GBC_best)

# Best score

gsGBC.best_score_
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
votingC = VotingClassifier(estimators=[('etc', etcc[etc_folds.index(max(etc_folds))]),('rfc', rfcc[rfc_folds.index(max(rfc_folds))]), ('gbdt',gbdtt[gbdt_folds.index(max(gbdt_folds))])], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, y_train)



exited_pred = pd.Series(votingC.predict(X_test), name="Exited_pred")



y_test.reset_index(drop=True, inplace=True)



results = pd.concat([y_test, exited_pred],axis=1)



results.to_csv("churn_modling.csv",index=False)