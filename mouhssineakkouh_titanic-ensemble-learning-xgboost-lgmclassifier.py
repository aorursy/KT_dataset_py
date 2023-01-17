# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from collections import Counter


import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


from sklearn.metrics import accuracy_score, log_loss

from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from mlxtend.classifier import EnsembleVoteClassifier



from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

pd.set_option('max_columns', 50)
pd.options.display.max_colwidth = 200

# Load in the train and test datasets
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# Store our passenger ID for easy access
PassengerId = test['PassengerId'].astype("object")

## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# Preview the data
train.head()
train.info()

print("*"*40)

test.info()
train.describe()
train.describe(include=['O'])

plot1 = sns.barplot(x="Pclass" , y="Survived" , data = train)
plt.ylim(0,1)
for p in plot1.patches:
    plot1.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2.,0.1 + p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving for the three classes");
plot2 = sns.barplot(x = "Sex" , y = "Survived" , data = train)
plt.ylim(0,1)
for p in plot2.patches:
    plot2.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., 0.1 + p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving for males and females");
plot3 = sns.barplot(x = "SibSp", y = "Survived",data = train , errwidth= 0)

plt.ylim(0,1)
for p in plot3.patches:
    plot3.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., + p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving by number of siblings / spouses aboard the titanic\n");
plot4 = sns.barplot(x = "Parch" , y = "Survived" , data = train , errwidth = 0)

plt.ylim(0,0.7)
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
for p in plot4.patches:
    plot4.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving by number of parents / children aboard the titanic")

train["Embarked"].value_counts()

plot4 = sns.barplot(x = "Embarked" , y = "Survived" , data = train , errwidth = 0)

plt.ylim(0,0.7)
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6])
for p in plot4.patches:
    plot4.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving by number of parents / children aboard the titanic");

sns.distplot(train["Fare"], label="Skewness : %.2f"%(dataset["Fare"].skew()))
plt.legend(loc="best");
# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")


# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])
f,ax=plt.subplots(1,2,figsize=(12,5))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.drop("PassengerId",axis =1, inplace =False).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True);
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
plot5 = sns.barplot(x = "FamilySize" , y = "Survived" , data = dataset[:train_len] ,errwidth =0)

plt.ylim(0,1)
for p in plot5.patches:
    plot5.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving by family size");
# Create new features of family size
dataset['Single'] = dataset['FamilySize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['FamilySize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['FamilySize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.catplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.catplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Apply log to Fare to reduce skewness distribution
dataset["LogFare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(dataset[:train_len]["LogFare"], color="b", label="Skewness : %.2f"%(dataset["LogFare"].skew()))
g = g.legend(loc="best")
dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

dataset['CategoricalFare'] = dataset["Fare"].copy()

# Mapping Fare
dataset.loc[ dataset['CategoricalFare'] <= 7.91, 'CategoricalFare']                               = 0
dataset.loc[(dataset['CategoricalFare'] > 7.91) & (dataset['CategoricalFare'] <= 14.454), 'CategoricalFare'] = 1
dataset.loc[(dataset['CategoricalFare'] > 14.454) & (dataset['CategoricalFare'] <= 31), 'CategoricalFare']   = 2
dataset.loc[ dataset['CategoricalFare'] > 31, 'CategoricalFare']                                  = 3
dataset['CategoricalFare'] = dataset['CategoricalFare'].astype(int)

plot7 = sns.barplot(x= "CategoricalFare" , y = "Survived" , data =dataset[:train_len] ,errwidth =0)

plt.ylim(0,1)
for p in plot7.patches:
    plot7.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
plt.title("Percentage of surviving for each categorical Fare");
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.catplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.catplot(y="Age",x="SibSp", data=dataset,kind="box")
g = sns.heatmap(dataset[:train_len][["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
dataset["Age1"] = dataset["Age"].copy()

# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age1"][dataset["Age1"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset[:train_len]["Age1"].median()
    age_pred = dataset["Age1"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age1'].iloc[i] = age_pred
    else :
        dataset['Age1'].iloc[i] = age_med


age_avg    = train['Age'].mean()
age_std    = train['Age'].std()
age_null_count = dataset['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
dataset['Age'] = dataset['Age'].astype(int)

    
dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)
# Or

dataset['CategoricalAge'] = dataset["Age"].copy()


# Mapping Age
dataset.loc[ dataset['CategoricalAge'] <= 16, 'CategoricalAge']                          = 0
dataset.loc[(dataset['CategoricalAge'] > 16) & (dataset['CategoricalAge'] <= 32), 'CategoricalAge'] = 1
dataset.loc[(dataset['CategoricalAge'] > 32) & (dataset['CategoricalAge'] <= 48), 'CategoricalAge'] = 2
dataset.loc[(dataset['CategoricalAge'] > 48) & (dataset['CategoricalAge'] <= 64), 'CategoricalAge'] = 3
dataset.loc[ dataset['CategoricalAge'] > 64, 'CategoricalAge']                            = 4
# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 
# Replace Rare Titles 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Mme','Miss','Mrs'], 'Mrs-Miss')

g = sns.countplot(dataset["Title"])
plt.ylim(0,850)

for p in g.patches:
    g.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")
# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
dataset['CatTitle'] = dataset['Title'].map(title_mapping)
dataset['CatTitle'] = dataset['CatTitle'].fillna(0).astype(int)
# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")

dataset["Ticket"].head()
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket

dataset.head()
dataset = pd.get_dummies(dataset, columns = ["Title"],drop_first = True)
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em",drop_first = True)
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin",drop_first = True)
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T",drop_first = True)
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc",drop_first = True)
# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.head()
dataset.columns.values
DF_train = dataset[:train_len]
DF_test = dataset[train_len:]

DF_test.drop(labels=["Survived"],axis = 1,inplace=True)

Y = DF_train["Survived"].astype(int)


DF_train.drop(labels=["Survived"],axis = 1,inplace=True)
Train0 = DF_train.copy()
Test0  = DF_test.copy()

Col1 = [ 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize',
                   'CatTitle','Single', 'SmallF', 'MedF', 'LargeF', 'Em_Q', 'Em_S', 'Cabin_B', 'Cabin_C', 'Cabin_D',
                   'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_X', 'T_A4',
                   'T_A5', 'T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON',
                   'T_FC', 'T_FCC', 'T_Fa', 'T_LINE', 'T_LP', 'T_PC', 'T_PP', 'T_PPP',
                   'T_SC', 'T_SCA3', 'T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS',
                   'T_SCParis', 'T_SOC', 'T_SOP', 'T_SOPP', 'T_SOTONO2', 'T_SOTONOQ',
                   'T_SP', 'T_STONO', 'T_STONO2', 'T_STONOQ', 'T_SWPP', 'T_WC',
                   'T_WEP', 'T_X', 'Pc_2', 'Pc_3']

Train1 = DF_train[Col1]
Test1  = DF_test[Col1]

Col2 = [ 'Sex', 'SibSp', 'Parch', 'FamilySize',
                   'LogFare',  'Age1', 
                   'Single', 'SmallF', 'MedF', 'LargeF', 'Title_Mr', 'Title_Mrs-Miss',
                   'Title_Rare', 'Em_Q', 'Em_S', 'Cabin_B', 'Cabin_C', 'Cabin_D',
                   'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_X', 'T_A4',
                   'T_A5', 'T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON',
                   'T_FC', 'T_FCC', 'T_Fa', 'T_LINE', 'T_LP', 'T_PC', 'T_PP', 'T_PPP',
                   'T_SC', 'T_SCA3', 'T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS',
                   'T_SCParis', 'T_SOC', 'T_SOP', 'T_SOPP', 'T_SOTONO2', 'T_SOTONOQ',
                   'T_SP', 'T_STONO', 'T_STONO2', 'T_STONOQ', 'T_SWPP', 'T_WC',
                   'T_WEP', 'T_X', 'Pc_2', 'Pc_3']

Train2 = DF_train[Col2]
Test2  = DF_test[Col2]


Col3  =  [ 'Sex', 'SibSp', 'Parch', 'FamilySize',
                    'CategoricalFare', 'CategoricalAge', 
                   'Single', 'SmallF', 'MedF', 'LargeF', 'Title_Mr', 'Title_Mrs-Miss',
                   'Title_Rare', 'Em_Q', 'Em_S', 'Cabin_B', 'Cabin_C', 'Cabin_D',
                   'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_X', 'T_A4',
                   'T_A5', 'T_AQ3', 'T_AQ4', 'T_AS', 'T_C', 'T_CA', 'T_CASOTON',
                   'T_FC', 'T_FCC', 'T_Fa', 'T_LINE', 'T_LP', 'T_PC', 'T_PP', 'T_PPP',
                   'T_SC', 'T_SCA3', 'T_SCA4', 'T_SCAH', 'T_SCOW', 'T_SCPARIS',
                   'T_SCParis', 'T_SOC', 'T_SOP', 'T_SOPP', 'T_SOTONO2', 'T_SOTONOQ',
                   'T_SP', 'T_STONO', 'T_STONO2', 'T_STONOQ', 'T_SWPP', 'T_WC',
                   'T_WEP', 'T_X', 'Pc_2', 'Pc_3']

Train3 = DF_train[Col3]
Test3  = DF_test[Col3]

Trains = [Train0 , Train1 , Train2, Train3]

Train1.shape , Train2.shape , Train3.shape
classifiers = [
    KNeighborsClassifier(),
    SVC(gamma=0.1,probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier(),
    GaussianNB(),
    XGBClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    Perceptron(),
    MLPClassifier(activation="logistic"),
    LGBMClassifier()]

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
cv_results = []
Names=[]
for classifier in classifiers :
    Names.append(classifier.__class__.__name__)
    cv_results.append(cross_val_score(classifier, Train0, y = Y, scoring = "accuracy", cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":Names}).sort_values(by = 'CrossValMeans')

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

cv_results = []
Names=[]
for classifier in classifiers :
    Names.append(classifier.__class__.__name__)
    cv_results.append(cross_val_score(classifier, Train1, y = Y, scoring = "accuracy", cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":Names}).sort_values(by = 'CrossValMeans')

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_results = []
Names=[]
for classifier in classifiers :
    Names.append(classifier.__class__.__name__)
    cv_results.append(cross_val_score(classifier, Train2, y = Y, scoring = "accuracy", cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":Names}).sort_values(by = 'CrossValMeans')

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_results = []
Names=[]
for classifier in classifiers :
    Names.append(classifier.__class__.__name__)
    cv_results.append(cross_val_score(classifier, Train3, y = Y, scoring = "accuracy", cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":Names}).sort_values(by = 'CrossValMeans')

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
SVC_par = [
          {"kernel":["linear"]},
          {"kernel":["rbf"],"C":[1, 10, 50, 100,200,300, 1000],"gamma":[0.001,0.01,0.1,1]},
          {"kernel":["poly"],"degree":[2,3,4,5]}
         ]
SVC_Grid= GridSearchCV(SVC(),SVC_par,scoring="accuracy",cv=10,n_jobs=-1)
SVC_Grid.fit(Train2,Y)
SVC_scores = pd.DataFrame(SVC_Grid.cv_results_)

SVC_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
GB_par = {  'n_estimators' : [100,200,300],
            'learning_rate': [0.2, 0.1, 0.05, 0.01],
            'max_depth': [4, 8],
            'min_samples_leaf': [50,100,150],
            'max_features': [0.5,0.3, 0.1],
            'min_samples_split':[2,5,10]
              }
GB_Grid= GridSearchCV(GradientBoostingClassifier(),GB_par,scoring="accuracy",cv=10,n_jobs=-1)
GB_Grid.fit(Train2,Y)
GB_scores = pd.DataFrame(GB_Grid.cv_results_)

GB_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
LGBM_par = { 'boosting_type':['gbdt','dart','goss','rf'],
            'learning_rate':[0.1, 0.2, 0.5, 0.01],
            'n_estimators':[100, 200, 300],
            'objective':["binary"]  
    
}
LGBM_Grid= GridSearchCV(LGBMClassifier(),LGBM_par,scoring="accuracy",cv=10,n_jobs=-1)
LGBM_Grid.fit(Train2,Y)
LGBM_scores = pd.DataFrame(LGBM_Grid.cv_results_)

LGBM_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
LR_par = {'penalty':['l1', 'l2', 'elasticnet', 'none'],
          'C':[0.5,1,2,5,10,100],
          'fit_intercept':[True , False],
          'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'max_iter':[100,200,300]
 
}
LR_Grid= GridSearchCV(LogisticRegression(),LR_par,scoring="accuracy",cv=10,n_jobs=-1)
LR_Grid.fit(Train2,Y)
LR_scores = pd.DataFrame(LR_Grid.cv_results_)

LR_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
LDA_par = {'solver':['svd', 'lsqr']}
LDA_Grid= GridSearchCV(LinearDiscriminantAnalysis(),LDA_par,scoring="accuracy",cv=10,n_jobs=-1)
LDA_Grid.fit(Train2,Y)
LDA_scores = pd.DataFrame(LDA_Grid.cv_results_)

LDA_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
XGB_par = {'n_estimators':[50, 100, 200],
           'max_depth':[4, 8, 10],
           'learning_rate':[0.1, 0.2, 0.5, 0.01],
  
}
XGB_Grid= GridSearchCV(XGBClassifier(),XGB_par,scoring="accuracy",cv=10,n_jobs=-1)
XGB_Grid.fit(Train2,Y)
XGB_scores = pd.DataFrame(XGB_Grid.cv_results_)

XGB_scores.sort_values(by = "rank_test_score")[["params","rank_test_score","rank_test_score","mean_test_score","std_test_score"]].head(10)
XGB = XGB_Grid.best_estimator_
LDA = LDA_Grid.best_estimator_
SVC = SVC_Grid.best_estimator_
LR = LR_Grid.best_estimator_
LGBM = LGBM_Grid.best_estimator_
GB = GB_Grid.best_estimator_

votingC = VotingClassifier(estimators=[('XGB', XGB),
                                       ('LDA', LDA),
                                       ('SVC', SVC),
                                       ('LR', LR),
                                       ('LGBM', LGBM),
                                       ('GB', GB)],
                           voting='hard', n_jobs=-1)


votingC_score = cross_val_score(votingC, Train2, y = Y, scoring = "accuracy", cv = 10, n_jobs=-1).mean()

votingC_score
votingC.fit(Train2, Y)

Pred_Survived = pd.Series(votingC.predict(Test2), name="Survived")

results = pd.concat([PassengerId,Pred_Survived],axis=1)

results.to_csv("Titanic Ensemble Voting.csv",index=False)