#Import libraries



import numpy as np

from numpy.random import random_integers

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

from sklearn.cross_validation import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from scipy.stats import pointbiserialr, spearmanr

%matplotlib inline



print('Libraries Ready!')
#Load training data



path = '../input/'

df = pd.read_csv(path+'train.csv')

df.head()
Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    } 



df['Title'] = df['Name'].apply(lambda x: Title_Dictionary[x.split(',')[1].split('.')[0].strip()])



df.head()
def Ticket_Prefix(s):

    s=s.split()[0]

    if s.isdigit():

        return 'NoClue'

    else:

        return s



df['TicketPrefix'] = df['Ticket'].apply(lambda x: Ticket_Prefix(x))



df.head()
df.info()
mask_Age = df.Age.notnull()

Age_Sex_Title_Pclass = df.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]

Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()

Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)



mask_Age = df.Age.isnull()

Age_Sex_Title_Pclass_missing = df.loc[mask_Age, ["Title", "Sex", "Pclass"]]



def Age_filler(row):

    if row.Sex == "female":

        age = Filler_Ages.female.loc[row["Title"], row["Pclass"]]

        return age

    

    elif row.Sex == "male":

        age = Filler_Ages.male.loc[row["Title"], row["Pclass"]]

        return age

    

Age_Sex_Title_Pclass_missing["Age"]  = Age_Sex_Title_Pclass_missing.apply(Age_filler, axis = 1)   



df["Age"] = pd.concat([Age_Sex_Title_Pclass["Age"], Age_Sex_Title_Pclass_missing["Age"]])    



df.head()
df['Fare']=df['Fare'].fillna(value=df.Fare.mean())

df.head()
df['FamilySize'] = df['SibSp'] + df['Parch']

df = df.drop(['Ticket', 'Cabin'], axis=1)

df.head()
dummies_Sex=pd.get_dummies(df['Sex'],prefix='Sex')

dummies_Embarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked') 

dummies_Pclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')

dummies_Title = pd.get_dummies(df['Title'], prefix= 'Title')

dummies_TicketPrefix = pd.get_dummies(df['TicketPrefix'], prefix='TicketPrefix')

df = pd.concat([df, dummies_Sex, dummies_Embarked, dummies_Pclass, dummies_Title, dummies_TicketPrefix], axis=1)

df = df.drop(['Sex','Embarked','Pclass','Title','Name','TicketPrefix'], axis=1)



df.head()
df = df.set_index(['PassengerId'])

df.head()
columns = df.columns.values



param=[]

correlation=[]

abs_corr=[]



for c in columns:

    #Check if binary or continuous

    if len(df[c].unique())<=2:

        corr = spearmanr(df['Survived'],df[c])[0]

    else:

        corr = pointbiserialr(df['Survived'],df[c])[0]

    param.append(c)

    correlation.append(corr)

    abs_corr.append(abs(corr))



#Create dataframe for visualization

param_df=pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})



#Sort by absolute correlation

param_df=param_df.sort_values(by=['abs_corr'], ascending=False)



#Set parameter name as index

param_df=param_df.set_index('parameter')



param_df.head()
scoresCV = []

scores = []



for i in range(1,len(param_df)):

    new_df=df[param_df.index[0:i+1].values]

    X = new_df.ix[:,1::]

    y = new_df.ix[:,0]

    clf = DecisionTreeClassifier()

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

    

plt.figure(figsize=(15,5))

plt.plot(range(1,len(scores)+1),scores, '.-')

plt.axis("tight")

plt.title('Feature Selection', fontsize=14)

plt.xlabel('# Features', fontsize=12)

plt.ylabel('Score', fontsize=12)

plt.grid();
best_features=param_df.index[1:10+1].values

print('Best features:\t',best_features)
df[best_features].hist(figsize=(20,15));
X = df[best_features]

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)
plt.figure(figsize=(15,7))



#Max Features

plt.subplot(2,3,1)

feature_param = ['auto','sqrt','log2',None]

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(max_features=feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Features')

plt.xticks(range(len(feature_param)), feature_param)

plt.grid();



#Max Depth

plt.subplot(2,3,2)

feature_param = range(1,51)

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(max_depth=feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Depth')

plt.grid();



#Min Samples Split

plt.subplot(2,3,3)

feature_param = range(1,51)

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(min_samples_split =feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Min Samples Split')

plt.grid();



#Min Samples Leaf

plt.subplot(2,3,4)

feature_param = range(1,51)

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(min_samples_leaf =feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Min Samples Leaf')

plt.grid();



#Min Weight Fraction Leaf

plt.subplot(2,3,5)

feature_param = np.linspace(0,0.5,10)

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Min Weight Fraction Leaf')

plt.grid();



#Max Leaf Nodes

plt.subplot(2,3,6)

feature_param = range(2,21)

scores=[]

for feature in feature_param:

    clf = DecisionTreeClassifier(max_leaf_nodes=feature)

    clf.fit(X_train,y_train)

    scoreCV = sklearn.cross_validation.cross_val_score(clf, X, y, cv= 10)

    scores.append(np.mean(scoreCV))

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Leaf Nodes')

plt.grid();
plt.figure(figsize=(15,10))



#N Estimators

plt.subplot(3,3,1)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(n_estimators=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('N Estimators')

plt.grid();



#Criterion

plt.subplot(3,3,2)

feature_param = ['gini','entropy']

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(criterion=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(scores, '.-')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Criterion')

plt.xticks(range(len(feature_param)), feature_param)

plt.grid();



#Max Features

plt.subplot(3,3,3)

feature_param = ['auto','sqrt','log2',None]

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_features=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Features')

plt.xticks(range(len(feature_param)), feature_param)

plt.grid();



#Max Depth

plt.subplot(3,3,4)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_depth=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Depth')

plt.grid();



#Min Samples Split

plt.subplot(3,3,5)

feature_param = range(1,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_samples_split =feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Min Samples Split')

plt.grid();



#Min Weight Fraction Leaf

plt.subplot(3,3,6)

feature_param = np.linspace(0,0.5,10)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Min Weight Fraction Leaf')

plt.grid();



#Max Leaf Nodes

plt.subplot(3,3,7)

feature_param = range(2,21)

scores=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_leaf_nodes=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Leaf Nodes')

plt.grid();



df.keys()
importance = clf.feature_importances_

importance
importance = clf.feature_importances_

plt.plot(df.keys()[0:10],importance)

plt.show()