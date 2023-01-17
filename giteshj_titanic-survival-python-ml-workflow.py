import re
import sklearn
import xgboost as xgb 
import matplotlib.pyplot as plt
# Going to use these 5 base models 
from sklearn.ensemble import (RandomForestClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')
# import train and test to play with it
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
IDtest = df_test["PassengerId"]
df_train.head()
df_test.head()
df_train.describe(include="all")
df_test.describe()
print(df_train.isnull().sum())
print(df_test.info())
data = [df_train, df_test]
for dataset in data:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
df_train.drop(drop_column, axis=1, inplace = True)
df_test.drop(drop_column, axis=1, inplace = True)
print(df_train.isnull().sum())
print("-"*10)
print(df_test.isnull().sum())
print(df_train.isnull().sum())
print(df_test.info())
g = sns.FacetGrid(df_train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
ax= sns.boxplot(x="Pclass", y="Age", data=df_train)
ax= sns.stripplot(x="Pclass", y="Age", data=df_train, jitter=True, edgecolor="gray")
plt.show()
df_train.hist(figsize=(15,20), color = "orange")
plt.figure()
df_train["Age"].hist();
f,ax=plt.subplots(1,2,figsize=(20,10))
df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='orange')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='Lime',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=df_train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()
# scatter plot matrix
pd.plotting.scatter_matrix(df_train,figsize=(15,18))
plt.figure()
plt.figure(figsize=(7,4)) 
sns.heatmap(df_train.corr(),annot=True,cmap='BuGn') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()

pp = sns.pairplot(df_train, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
pal = {'male':"green", 'female':"Orange"}
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived", 
            data=df_train, 
            palette = pal,
            linewidth=2 )
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
pal = {1:"seagreen", 0:"orange"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(x = "Sex", 
                   hue="Survived",
                   data = df_train, 
                   linewidth=2, 
                   palette = pal
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()
plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=df_train, 
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(df_train.Pclass[df_train.Survived == 0] , 
               color='orange',
               shade=True,
               label='not survived')
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(df_train.Pclass.unique()), labels);

fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 0),'Fare'] , color='orange',shade=True,label='not survived')
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize = 15)

fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 0),'Age'] , color='orange',shade=True,label='not survived')
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 25)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15);
pal = {1:"seagreen", 0:"orange"}
g = sns.FacetGrid(df_train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
g = sns.FacetGrid(df_train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal
                  )
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
sib = pd.crosstab(df_train['SibSp'], df_train['Sex'])
dummy = sib.div(sib.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Siblings')
dummy = plt.ylabel('Percentage')

parch = pd.crosstab(df_train['Parch'], df_train['Sex'])
dummy = parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Parent/Children')
dummy = plt.ylabel('Percentage')

sns.barplot(x="SibSp", y="Survived", data=df_train)

#I won't be printing individual percent values for all of these.
print("Percentage of SibSp = 0 who survived:", df_train["Survived"][df_train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", df_train["Survived"][df_train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", df_train["Survived"][df_train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
#draw a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=df_train)

#I won't be printing individual percent values for all of these.
print("Percentage of Parch = 0 who survived:", df_train["Survived"][df_train["Parch"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 1 who survived:", df_train["Survived"][df_train["Parch"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Parch = 2 who survived:", df_train["Survived"][df_train["Parch"] == 2].value_counts(normalize = True)[1]*100)
#sort the ages into logical categories
df_train["Age"] = df_train["Age"].fillna(-0.5)
df_test["Age"] = df_test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
df_train['AgeGroup'] = pd.cut(df_train["Age"], bins, labels = labels)
df_test['AgeGroup'] = pd.cut(df_test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival

plt.figure(figsize=(10,5))
sns.barplot(x="AgeGroup", y="Survived", data=df_train, )

plt.show()

# data = [df_train, df_test] declared at the top
for dataset in data:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


    
stat_min = 10 
#this will create a true false series with title name as index
title_names = (df_train['Title'].value_counts() < stat_min) 


df_train['Title'] = df_train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
#this will create a true false series with title name as index
title_names = (df_test['Title'].value_counts() < stat_min) 


df_test['Title'] = df_test['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(df_train['Title'].value_counts())
print(df_test['Title'].value_counts())

#code categorical data
label = LabelEncoder()
for dataset in data:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
df_train.head(1)
Target = ['Survived']
features = ['Sex_Code','Pclass', 'Embarked_Code', 
               'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code', 'IsAlone']
print(df_train[features].shape)
print(df_test[features].shape)
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
#Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(XGBClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
cv_results = []
for classifier in classifiers :
    #print(classifier)
    cv_results.append(cross_val_score(classifier, df_train[features],
                y = df_train[Target], scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
         "Algorithm":["XGB", "DecisionTree","RandomForest",
            "ExtraTrees","GradientBoosting"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3"
                ,orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
g = sns.scatterplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3")
import warnings
warnings.filterwarnings("ignore")
nrows = 5
ncols = 1
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,50),squeeze=False)
name = ["XGB", "DecisionTree","RandomForest",
            "ExtraTrees","GradientBoosting"]
classifiers = []
classifiers.append(XGBClassifier(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
row=0
col=0
l=[]
for classifier in classifiers :
    classifier.fit(df_train[features],df_train[Target])
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    try:
        g = sns.barplot(y=df_train[features].columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
    except:
        print("Dfd")
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name[row] + " feature importance")
    row+=1
test_Survived_X = pd.Series(classifiers[0].predict(df_test[features]), name="XGB")
test_Survived_B = pd.Series(classifiers[1].predict(df_test[features]), name="DecisionTree")
test_Survived_R = pd.Series(classifiers[2].predict(df_test[features]), name="RandomFOrest")
test_Survived_E = pd.Series(classifiers[3].predict(df_test[features]), name="ExtraTree")
test_Survived_G = pd.Series(classifiers[4].predict(df_test[features]), name="GradientBoosting")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_X,test_Survived_B,test_Survived_R,test_Survived_E, test_Survived_G],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)
i = 0
for clf in classifiers:
    test_Survived = pd.Series(clf.predict(df_test[features]), name="Survived")
    results = pd.concat([IDtest,test_Survived],axis=1)
    file_name = name[i] + ".csv"
    results.to_csv(file_name,index=False)
    i += 1
    print(file_name)
