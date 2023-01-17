import os
print(os.listdir("../input/"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import re

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
train = pd.read_csv("../input/train.csv")
train['label'] = 'train'
test = pd.read_csv("../input/test.csv")
test['label'] = 'test'
test_passengerId = test.PassengerId  #Save test passengerId. It will be required at the end
df = train.append(test)
df.sample(2)
df.info()
df.isnull().sum()
df.describe(include = 'all')
#Fill missing value
df['Embarked'].fillna('S', inplace = True)    #top value with freq 914
df[df.Fare.isnull()]
df.corr().Fare
print(df[df.Pclass == 1].Fare.quantile([0.25, 0.50, 0.75]))
print(df[df.Pclass == 2].Fare.quantile([0.25, 0.50, 0.75]))
print(df[df.Pclass == 3].Fare.quantile([0.25, 0.50, 0.75]))
sns.factorplot(x = 'Pclass', y = 'Fare', data = df)
df['Fare'].fillna(df[df.Pclass == 3].Fare.median(), inplace = True)   #Fare is dependent on Pclass
print("Age column has", df.Age.isnull().sum(), "missing values out of", len(df), ". Missing value percentage =", df.Age.isnull().sum()/len(df)*100)
df.corr().Age
df.pivot_table(values = 'Age', index = 'Pclass').Age.plot.bar()
df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp'], aggfunc = 'median').Age.plot.bar()
df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp', 'Parch'], aggfunc = 'median')
df.Age.isnull().sum()
age_null = df.Age.isnull()
group_med_age = df.pivot_table(values = 'Age', index = ['Pclass', 'SibSp'], aggfunc = 'median')
df.loc[age_null, 'Age'] = df.loc[age_null, ['Pclass', 'SibSp']].apply(lambda x: group_med_age.loc[(group_med_age.index.get_level_values('Pclass') == x.Pclass) & (group_med_age.index.get_level_values('SibSp') == x.SibSp)].Age.values[0], axis = 1)
df.Age.isnull().sum()
print("Cabin has", df.Cabin.isnull().sum(), "missing values out of", len(df))
df['Cabin'] = df.Cabin.str[0]
df.Cabin.unique()
df.Cabin.fillna('O', inplace = True)
df.isnull().sum()
df.sample(2)
sns.factorplot(data = df, x = 'Sex', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Sex').Survived.plot.bar()
plt.ylabel('Survival Probability')
q = sns.kdeplot(df.Age[df.Survived == 1], shade = True, color = 'red')
q = sns.kdeplot(df.Age[df.Survived == 0], shade = True, color = 'blue')
q.set_xlabel("Age")
q.set_ylabel("Frequency")
q = q.legend(['Survived', 'Not Survived'])
q = sns.FacetGrid(df, col = 'Survived')
q.map(sns.distplot, 'Age')
sns.factorplot(data = df, x = 'Embarked', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Embarked').Survived.plot.bar()
plt.ylabel('Survival Probability')
df.pivot_table(values = 'Survived', index = ['Sex','Embarked']).Survived.plot.bar()
plt.ylabel('Survival Probability')
fig, ax =plt.subplots(1,2)
sns.countplot(data = df[df.Sex == 'female'], x = 'Embarked', hue = 'Survived', ax = ax[0])
sns.countplot(data = df[df.Sex == 'male'], x = 'Embarked', hue = 'Survived', ax = ax[1])
fig.show()
sns.factorplot(data = df, x = 'Parch', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Parch').Survived.plot.bar()
plt.ylabel('Survival Probability')
sns.factorplot(data = df, x = 'Pclass', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Pclass').Survived.plot.bar()
plt.ylabel('Survival Probability')
df.pivot_table(values = 'Survived', index = ['Sex', 'Pclass']).Survived.plot.bar()
plt.ylabel('Survival Probability')
sns.factorplot(data = df, x = 'Cabin', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Cabin').Survived.plot.bar()
plt.ylabel('Survival Probability')
plt.boxplot(train.Fare, showmeans = True)
plt.title('Fare Boxplot')
plt.ylabel('Fares')
sns.distplot(df.Fare)
df.Fare.skew()    #Measure of skewness level
df['Fare_log'] = df.Fare.map(lambda i: np.log(i) if i > 0 else 0)
sns.distplot(df.Fare_log)
df.Fare_log.skew()
df['Family_size'] = 1 + df.Parch + df.SibSp
df['Alone'] = np.where(df.Family_size == 1, 1, 0)
print(df.Family_size.value_counts())
print(df.Alone.value_counts())
sns.factorplot(data = df, x = 'Family_size', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Family_size').Survived.plot.bar()
plt.ylabel('Survival Probability')
df.loc[df['Family_size'] == 1, 'Family_size_bin'] = 0
df.loc[(df['Family_size'] >= 2) & (df['Family_size'] <= 4), 'Family_size_bin'] = 1
df.loc[df['Family_size'] >=5, 'Family_size_bin'] = 2
sns.factorplot(data = df, x = 'Alone', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Alone').Survived.plot.bar()
plt.ylabel('Survival Probability')
df['Title'] = df.Name.str.split(", ", expand = True)[1].str.split(".", expand = True)[0]
df.Title.value_counts()
minor_titles = df.Title.value_counts() <= 4
df['Title'] = df.Title.apply(lambda x: 'Others' if minor_titles.loc[x] == True else x)
df.Title.value_counts()
sns.factorplot(data = df, x = 'Title', hue = 'Survived', kind = 'count')
df.pivot_table(values = 'Survived', index = 'Title').Survived.plot.bar()
plt.ylabel('Survival Probability')
df['Fare_bin'] = pd.qcut(df.Fare, 4, labels = [0,1,2,3]).astype(int)
df['Age_bin'] = pd.cut(df.Age.astype(int), 5, labels = [0,1,2,3,4]).astype(int)
sns.factorplot(data = df, x = 'Age_bin', hue = 'Survived', kind = 'count')
sns.factorplot(data = df, x = 'Fare_bin', hue = 'Survived', kind = 'count')
fig, axs = plt.subplots(1, 3,figsize=(15,5))

sns.pointplot(x = 'Fare_bin', y = 'Survived',  data=df, ax = axs[0])
sns.pointplot(x = 'Age_bin', y = 'Survived',  data=df, ax = axs[1])
sns.pointplot(x = 'Family_size', y = 'Survived', data=df, ax = axs[2])
label = LabelEncoder()
df['Title'] = label.fit_transform(df.Title)
df['Sex'] = label.fit_transform(df.Sex)
df['Embarked'] = label.fit_transform(df.Embarked)
df['Cabin'] = label.fit_transform(df.Cabin)
df.sample(2)
#We will look at correlation between variables. So before working with ticket column, save all variables we worked on yet.
#This id because we will use get_dummies on ticket and not label encoding.
corr_columns = list(df.drop(['Name', 'PassengerId', 'Ticket', 'label'], axis = 1).columns)
df['Ticket'] = df.Ticket.map(lambda x: re.sub(r'\W+', '', x))   #Remove special characters
#If ticket is of digit value, make them a character X
Ticket = []
for i in list(df.Ticket):
    if not i.isdigit():
        Ticket.append(i[:2])
    else:
        Ticket.append("X")
df['Ticket'] = Ticket
df.Ticket.unique()
df = pd.get_dummies(df, columns = ['Ticket'], prefix = 'T')
cat_variables = [x for x in df.columns if df.dtypes[x] == 'object']
cat_variables
df.drop(['Name', 'PassengerId'], axis = 1, inplace = True)
df.sample(2)
train = df.loc[df.label == 'train'].drop('label', axis = 1)
test = df.loc[df.label == 'test'].drop(['label', 'Survived'], axis = 1)
plt.figure(figsize = [14,10])
sns.heatmap(train[corr_columns].corr(), cmap = 'RdBu', annot = True)
plt.figure(figsize = [14,10])
sns.heatmap(train[corr_columns].corr(method = 'spearman'), cmap = 'RdBu', annot = True)
X_train = train.drop(['Survived'], axis = 1)
y_train = train['Survived'].astype(int)
X_test = test
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits = 5)
classifiers = []
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(LogisticRegression(random_state = 0))
classifiers.append(LinearSVC(random_state = 0))
classifiers.append(SVC(random_state = 0))
classifiers.append(RandomForestClassifier(random_state = 0))
classifiers.append(ExtraTreesClassifier(random_state = 0))
classifiers.append(XGBClassifier(random_state = 0))
classifiers.append(LGBMClassifier(random_state = 0))
classifiers.append(MLPClassifier())

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_train, y_train, scoring = 'accuracy', cv = kfold, n_jobs = -1))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({'CV_means':cv_means, 'CV_std':cv_std, 'Algorithm':['KNN', 'LinearDiscriminantAnalysis', 'LogisticRegression', 'LinearSVC', 'SVC', 'RandomForest', 'ExtraTrees', 'XGB', 'LGB', 'MLP']})

g = sns.barplot("CV_means", "Algorithm", data = cv_res, **{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
LDA_best = LinearDiscriminantAnalysis().fit(X_train, y_train)
RF = RandomForestClassifier(random_state = 0)
RF_params = {'n_estimators' : [10,50,100],
             'criterion' : ['gini', 'entropy'],
             'max_depth' : [5,8,None],
             'min_samples_split' : [2,5,8],
             'min_samples_leaf' : [1,3,5],
             'max_features' : ['auto', 'log2', None]}
GS_RF = GridSearchCV(RF, param_grid = RF_params, cv = kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_RF.fit(X_train, y_train)
RF_best = GS_RF.best_estimator_
print("Best parameters :", RF_best)
print("Best score :", GS_RF.best_score_)
ET = ExtraTreesClassifier(random_state = 0)
ET_params = {'n_estimators' : [10,50,100],
             'criterion' : ['gini', 'entropy'],
             'max_depth' : [5,8,None],
             'min_samples_split' : [2,5,8],
             'min_samples_leaf' : [1,3,5],
             'max_features' : ['auto', 'log2', None]}
GS_ET = GridSearchCV(ET, param_grid = ET_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_ET.fit(X_train, y_train)
ET_best = GS_ET.best_estimator_
print('Best parameters :', ET_best)
print('Best score :', GS_ET.best_score_)
import warnings
warnings.filterwarnings('ignore')
XGB = XGBClassifier(random_state = 0)
XGB_params = {'n_estimators' : [100,200,500],
              'max_depth' : [3,4,5],
              'learning_rate' : [0.01,0.05,0.1,0.2],
              'booster' : ['gbtree', 'gblinear', 'dart']}
GS_XGB = GridSearchCV(XGB, param_grid = XGB_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_XGB.fit(X_train, y_train)
XGB_best = GS_XGB.best_estimator_
print('Best parameters :', XGB_best)
print('Best score :', GS_XGB.best_score_)
LGB = LGBMClassifier(random_state = 0)
LGB_params = {'n_estimators' : [100,200,500],
              'max_depth' : [5,8,-1],
              'learning_rate' : [0.01,0.05,0.1,0.2],
              'boosting_type' : ['gbdt', 'goss', 'dart']}
GS_LGB = GridSearchCV(LGB, param_grid = LGB_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_LGB.fit(X_train, y_train)
LGB_best = GS_LGB.best_estimator_
print('Best parameters :', LGB_best)
print('Best score :', GS_LGB.best_score_)
MLP = MLPClassifier(random_state = 0)
MLP_params = {'hidden_layer_sizes' : [[10], [10,10], [10,100], [100,100]],
              'activation' : ['relu', 'tanh', 'logistic'],
              'alpha' : [0.0001,0.001,0.01]}
GS_MLP = GridSearchCV(MLP, param_grid = MLP_params, cv= kfold, scoring = 'accuracy', n_jobs = -1, verbose = 1)
GS_MLP.fit(X_train, y_train)
MLP_best = GS_MLP.best_estimator_
print('Best parameters :', MLP_best)
print('Best score :', GS_MLP.best_score_)
fig, axes = plt.subplots(2, 2, figsize = [20,10])
fig.subplots_adjust(hspace = 0.7)
classifiers_list = [["RandomForest", RF_best], ["ExtraTrees", ET_best],
                    ["XGBoost", XGB_best], ["LGBoost", LGB_best]]

nClassifier = 0
for row in range(2):
    for col in range(2):
        name = classifiers_list[nClassifier][0]
        classifier = classifiers_list[nClassifier][1]
        feature = pd.Series(classifier.feature_importances_, X_train.columns).sort_values(ascending = False)
        feature.plot.bar(ax = axes[row,col])
        axes[row,col].set_xlabel("Features")
        axes[row,col].set_ylabel("Relative Importance")
        axes[row,col].set_title(name + " Feature Importance")
        nClassifier +=1
LDA_pred = pd.Series(LDA_best.predict(X_test), name = 'LDA')
MLP_pred = pd.Series(MLP_best.predict(X_test), name = "MLP")
RF_pred = pd.Series(RF_best.predict(X_test), name = "RFC")
ET_pred = pd.Series(ET_best.predict(X_test), name = 'ETC')
XGB_pred = pd.Series(XGB_best.predict(X_test), name = "XGB")
LGB_pred = pd.Series(LGB_best.predict(X_test), name = "LGB")

ensemble_results = pd.concat([LDA_pred, MLP_pred, RF_pred, ET_pred, XGB_pred, LGB_pred], axis = 1)
plt.figure(figsize = [8,5])
sns.heatmap(ensemble_results.corr(), annot = True)
voting = VotingClassifier(estimators = [['LDA', LDA_best], ["MLP", MLP_best],
                                        ['RFC', RF_best], ['ETC', ET_best],
                                        ['XGB', XGB_best], ['LGB', LGB_best]], voting = 'soft', n_jobs = -1)
voting = voting.fit(X_train, y_train)
results = pd.DataFrame(test_passengerId, columns = ['PassengerId']).assign(Survived = pd.Series(voting.predict(X_test)))
results.to_csv('models_voting.csv', index = None)