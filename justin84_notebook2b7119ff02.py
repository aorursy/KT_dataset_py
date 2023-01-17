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



path = './'

df = pd.read_csv(path+'train.csv')

df.head()
####Process Data



#People with stronger titles tend to have more help on board. Hence, we will categorize passengers based on titles.
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
#The ticket prefix may determine the status or cabin on board and hence will be included



# Blockquote
def Ticket_Prefix(s):

    s=s.split()[0]

    if s.isdigit():

        return 'NoClue'

    else:

        return s



df['TicketPrefix'] = df['Ticket'].apply(lambda x: Ticket_Prefix(x))



df.head()
#Now let's check for data types and missing values*emphasized text*
df.info()
#We can see that Age and Embarked has missing data.



#Simply dropping the Age NaNs would mean throwing away too much data.



#We add in the median age based on the Title, Pclass and Sex of each passenger.
# We do not agree with this so we are dropping the rows with no age

#mask_Age = df.Age.notnull()

#Age_Sex_Title_Pclass = df.loc[mask_Age, ["Age", "Title", "Sex", "Pclass"]]

#Filler_Ages = Age_Sex_Title_Pclass.groupby(by = ["Title", "Pclass", "Sex"]).median()

#Filler_Ages = Filler_Ages.Age.unstack(level = -1).unstack(level = -1)



#mask_Age = df.Age.isnull()

#Age_Sex_Title_Pclass_missing = df.loc[mask_Age, ["Title", "Sex", "Pclass"]]



#def Age_filler(row):

#    if row.Sex == "female":

#        age = Filler_Ages.female.loc[row["Title"], row["Pclass"]]

#       return age

#   

#    elif row.Sex == "male":

#        age = Filler_Ages.male.loc[row["Title"], row["Pclass"]]

#        return age

#    

# Age_Sex_Title_Pclass_missing["Age"]  = Age_Sex_Title_Pclass_missing.apply(Age_filler, axis = 1)   

#

# df["Age"] = pd.concat([Age_Sex_Title_Pclass["Age"], Age_Sex_Title_Pclass_missing["Age"]])    

#

# df.head()
#Next we fill in the missing Fare. (We disagree again)

#df['Fare']=df['Fare'].fillna(value=df.Fare.mean())

#df.head()
# We have to drop the rows with the missing age and fares

df.dropna(subset = ['Age', 'Fare']);
#We do not need Cabin and Ticket and hence can be dropped from our DataFrame.



#We also can combine SibSp and Parch to FamilySize.
df['FamilySize'] = df['SibSp'] + df['Parch']

df = df.drop(['Ticket', 'Cabin'], axis=1)

df.head()
#Now we deal with categorical data using dummy variables.
dummies_Sex=pd.get_dummies(df['Sex'],prefix='Sex')

dummies_Embarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked') 

dummies_Pclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')

dummies_Title = pd.get_dummies(df['Title'], prefix= 'Title')

dummies_TicketPrefix = pd.get_dummies(df['TicketPrefix'], prefix='TicketPrefix')

df = pd.concat([df, dummies_Sex, dummies_Embarked, dummies_Pclass, dummies_Title, dummies_TicketPrefix], axis=1)

df = df.drop(['Sex','Embarked','Pclass','Title','Name','TicketPrefix'], axis=1)



df.head()
#Finally, we set our PassengerId as our index.
df = df.set_index(['PassengerId'])

df.head()
####Feature Selection

#

#For feature selection, we will look at the correlation of each feature against Survived.

#Based on our data types, we will use the following aglorithms:



# - Spearman-Rank correlation for nominal vs nominal data

# - Point-Biserial correlation for nominal vs continuous data
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

#param_df
#Now that we have our correlation, we can use the Decision Tree classifier to see the score agaisnt feature space.



# We are going to use the following parameters (sorted by correlation to survival)

# Sex

# Pclass_3

# Pclass_1

# Fare

# ...

# We are dropping the titles because the important ones (mr, mrs, ms) are selfcontained in sex, while the special ones

# are non-correlated to survival rates



df = df.drop(['Title_Mr', 'Title_Mrs' , 'Title_Miss', 'Age'], axis=1)

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



#param_df.head()

param_df
# Based on the plot, a feature space of 10 dimensions provides the most reliable result while avoiding overfit.
# We are using ten

best_features=param_df.index[1:10+1].values

print('Best features:\t',best_features)

# Looking at out best features.
# Shows the histograms for the best features

df[best_features].hist(figsize=(20,15));
# Creating the train and test datasets.
X = df[best_features]

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44)



best_features=param_df.index[1:].values



X2 = df[best_features]

y2 = df['Survived']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.33, random_state=44)
plt.figure(figsize=(15,10))



#N Estimators

plt.subplot(3,3,1)

feature_param = range(1,21)

scores=[]

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(n_estimators=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(n_estimators=feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(scores, '.-')

plt.plot(scores2, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('N Estimators')

plt.grid();



#Criterion

plt.subplot(3,3,2)

feature_param = ['gini','entropy']

scores=[]

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(criterion=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(criterion=feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(scores2, '.-')

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

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_features=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(max_features=feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(scores2, '.-')

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

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_depth=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(max_depth=feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(feature_param, scores2, '.-')

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

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_samples_split =feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(min_samples_split =feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(feature_param, scores2, '.-')

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

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(min_weight_fraction_leaf =feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(min_weight_fraction_leaf =feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(feature_param, scores2, '.-')

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

scores2=[]

for feature in feature_param:

    clf = RandomForestClassifier(max_leaf_nodes=feature)

    clf.fit(X_train,y_train)

    scoreCV = clf.score(X_test,y_test)

    scores.append(scoreCV)

    # With all properties

    clf2 = RandomForestClassifier(max_leaf_nodes=feature)

    clf2.fit(X2_train,y2_train)

    scoreCV2 = clf2.score(X2_test,y2_test)

    scores2.append(scoreCV2)

plt.plot(feature_param, scores2, '.-')

plt.plot(feature_param, scores, '.-')

plt.axis('tight')

# plt.xlabel('parameter')

# plt.ylabel('score')

plt.title('Max Leaf Nodes')

plt.grid();
#Max Leaf Nodes

feature_param = range(2,21)

scores=[]

scores2=[]



clf = RandomForestClassifier()

clf.fit(X_train,y_train)

scoreCV = clf.score(X_test,y_test)

scores.append(scoreCV)

# With all properties

clf2 = RandomForestClassifier()

clf2.fit(X2_train,y2_train)

scoreCV2 = clf2.score(X2_test,y2_test)

scores2.append(scoreCV2)

    



[scores, scores2 ]
#Max Leaf Nodes

feature_param = range(2,21)

scores=[]

scores2=[]



clf = RandomForestClassifier()

clf.fit(X_train,y_train)

scoreCV = clf.score(X_test,y_test)

scores.append(scoreCV)

# With all properties

clf2 = RandomForestClassifier()

clf2.fit(X2_train,y2_train)

scoreCV2 = clf2.score(X2_test,y2_test)

scores2.append(scoreCV2)

    



[scores, scores2 ]
#df.keys()


def plotsomefunction(ax, x):



    return ax.plot(x, np.sin(x))



def plotsomeotherfunction(ax, x):



    return ax.plot(x,np.cos(x))





fig, ax = plt.subplots(1,1)

x = np.linspace(0,np.pi,1000)

l1 = plotsomefunction(ax, x)

l2 = plotsomeotherfunction(ax, x)

plt.show()
importance = clf.feature_importances_

importance
importance = clf.feature_importances_

plt.plot(df.keys()[0:10],importance)

plt.show()