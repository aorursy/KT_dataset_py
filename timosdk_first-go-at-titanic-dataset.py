import numpy as np 
import pandas as pd 
import re
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn import svm, neighbors, naive_bayes
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation

# plotting setup
%matplotlib inline
%config InlineBackend.figure_format='retina'
plt.rcParams['figure.figsize'] = [14.0, 6.0]

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# training data
data_train = pd.read_csv("../input/train.csv")

# test data
data_test = pd.read_csv("../input/test.csv")

# combine test and train into one dataframe - can always split again by ['Survived']
data_full = pd.concat([data_train, data_test])
print('Full dataset columns with null values: \n', data_full.isnull().sum())

# show us some representative data.
data_full.sample(5)
print("Chance of surviving in training set:", data_train['Survived'].sum()/len(data_train['Survived']))
print("Chance of surviving if you are a male:",data_train.loc[data_train['Sex'] == 'male']['Survived'].sum()/data_train['Sex'].value_counts()['male'])
print("Chance of surviving if you are a female:",data_train.loc[data_train['Sex'] == 'female']['Survived'].sum()/data_train['Sex'].value_counts()['female'])

data_full['IsFemale'] = 1
data_full.loc[data_full['Sex'] == 'male', 'IsFemale'] = 0

data_full["NameLength"] = data_full["Name"].apply(lambda x: len(x))

fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(data_full.loc[(data_train['Survived'] == 0), 'NameLength'], color='gray',shade=True,label='dead', bw=3)
ax=sns.kdeplot(data_full.loc[(data_train['Survived'] == 1), 'NameLength'], color='g',shade=True,label='alive', bw=3)
plt.title('Name Length Distribution - Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Name Length", fontsize = 15)

print("Longest Name:", data_full.loc[data_full['NameLength']==data_full['NameLength'].max()].Name.values[0])

test_titles = data_train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print(test_titles.value_counts())

print("-"*10)

# The most common ones are tied to gender and age and the less common ones are so uncommon that they make 
# little sense to encode - could encode as common vs fancy - but likely the fancy ones are also in high
# class and or expensive ticket.

title = 'Mr'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Miss'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Mrs'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Master'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Dr'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

title = 'Rev'
print("Chance of surviving if you are a ", title, ":", data_train.loc[test_titles == title]['Survived'].sum()/test_titles.value_counts()[title])

print("-"*10)


# get title from name
data_full['Titles'] = data_full['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

# fix these simple ones
data_full['Titles'] = data_full['Titles'].replace('Mlle', 'Miss')
data_full['Titles'] = data_full['Titles'].replace('Ms', 'Miss')
data_full['Titles'] = data_full['Titles'].replace('Mme', 'Mrs')

# get a list of the 6 most frequent titles
mostfrequenttitles = data_full['Titles'].value_counts().nlargest(6).keys()

# if your title is not in the top 6 you are a fancy person
data_full.loc[(data_full['Titles'].isin(mostfrequenttitles)==False), 'Titles'] = "Fancy"

# here is the value counts for the full dataset
print ("Title frequencies\n", data_full['Titles'].value_counts())
# create dummies from titles
dummies = pd.get_dummies(data_full['Titles'])
data_full = pd.concat([data_full, dummies], axis = 1)

data_full.sample(5)
fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 0) & (data_train['Age'].isnull() == False), 'Age'], color='gray',shade=True,label='dead', bw=3)
ax=sns.kdeplot(data_train.loc[(data_train['Survived'] == 1) & (data_train['Age'].isnull() == False), 'Age'], color='g',shade=True, label='survived', bw=3)
ax=sns.kdeplot(data_full.loc[(data_train['Age'].isnull() == False), 'Age'], color='b',shade=False, label='full dataset', bw=3)
plt.title('Age Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Age", fontsize = 15)
# For demonstration purposes only - cutting on the separated set gives rise to different and nonsensical cutoffs:
pd.cut(data_train['Age'], bins = 5)
#[(0.34, 16.336] < (16.336, 32.252] < (32.252, 48.168] < (48.168, 64.084] < (64.084, 80.0]]
pd.cut(data_test['Age'], bins = 5)
#[(0.0942, 15.336] < (15.336, 30.502] < (30.502, 45.668] < (45.668, 60.834] < (60.834, 76.0]]

# Now we will create age bins for the full dataset
ages = data_full['Age']
ages = ages.append(pd.Series([0, 80]))
bins = pd.cut(ages, bins = 8, labels = [1, 2, 3, 4, 5, 6, 7, 8])

data_full['AgeBin'] = bins[:-2].astype(float)

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='AgeBin', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True)
data_full['IsKid'] = 0
data_full.loc[data_full['AgeBin'] == 1, 'IsKid'] = 1
# iterate through titles and fill in missing values
titles = data_full['Titles'].value_counts().keys()
for title in titles:
    age_mode = data_full.loc[data_full['Titles']==title, 'AgeBin'].mode().values[0]
    data_full.loc[(data_full['Titles']==title) & (data_full['AgeBin'].isnull()), 'AgeBin'] = age_mode

# now convert agebin to int
data_full['AgeBin'] = data_full['AgeBin'].astype(int)

print('Full dataset columns with null values: \n', data_full.isnull().sum())

# Family size
data_full['FamilySize'] = data_full['SibSp'] + data_full['Parch'] + 1

# Is this person alone on the ship
data_full['IsAlone'] = 0
data_full.loc[data_full['FamilySize'] == 1, 'IsAlone'] = 1

# the guy is on 3rd class so lets use the median fare of 3rd class to fill this value
data_full.loc[data_full['Fare'].isnull(), 'Fare'] = data_full.loc[data_full['Pclass'] == 3, 'Fare'].median()

data_full.loc[data_full['Ticket']=='3701']
print("Number of unique fares:", data_full['Fare'].nunique())

plt.figure(figsize=[15,6])

plt.subplot(121)
plt.boxplot(x=data_full['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(122)
plt.hist(x = data_full['Fare'], color = ['g'], bins = 8)
plt.title('Fare Histogram')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers - log scale')
plt.yscale('log', nonposy='clip')
# add fare bins to the full dataset
data_full['FareBin'] = pd.qcut(data_full['Fare'], 6, labels = [1, 2, 3, 4, 5, 6]).astype(int)

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='FareBin', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True)
# create table with counts of people per ticket
ticket_table = pd.DataFrame(data_full["Ticket"].value_counts())
ticket_table.rename(columns={'Ticket': 'People_on_ticket'}, inplace = True)

ticket_table['Dead_female_on_ticket'] = data_full.Ticket[(data_full.AgeBin > 1) & (data_full.Survived < 1) & (data_full.IsFemale)].value_counts()
ticket_table['Dead_female_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Dead_female_on_ticket'] > 0, 'Dead_female_on_ticket'] = 1
ticket_table['Dead_female_on_ticket'] = ticket_table['Dead_female_on_ticket'].astype(int)

ticket_table['Dead_kid_on_ticket'] = data_full.Ticket[(data_full.AgeBin == 1) & (data_full.Survived < 1)].value_counts()
ticket_table['Dead_kid_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Dead_kid_on_ticket'] > 0, 'Dead_kid_on_ticket'] = 1
ticket_table['Dead_kid_on_ticket'] = ticket_table['Dead_kid_on_ticket'].astype(int)

ticket_table['Alive_male_on_ticket'] = data_full.Ticket[(data_full.AgeBin > 1) & (data_full.Survived > 0) & (data_full.IsFemale == False)].value_counts()
ticket_table['Alive_male_on_ticket'].fillna(0, inplace=True)
ticket_table.loc[ticket_table['Alive_male_on_ticket'] > 0, 'Alive_male_on_ticket'] = 1
ticket_table['Alive_male_on_ticket'] = ticket_table['Alive_male_on_ticket'].astype(int)

# unique identifiers for tickets with more than 2 people
ticket_table["Ticket_id"]= pd.Categorical(ticket_table.index).codes
ticket_table.loc[ticket_table["People_on_ticket"] < 3, 'Ticket_id' ] = -1

# merge with the data_full
data_full = pd.merge(data_full, ticket_table, left_on="Ticket",right_index=True,how='left', sort=False)
fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(15,6))
sns.pointplot(x='Dead_female_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis1)
sns.pointplot(x='Alive_male_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis2)
sns.pointplot(x='Dead_kid_on_ticket', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis3)
data_full['Lastname'] = data_full["Name"].apply(lambda x: x.split(',')[0].lower())

lastname_table = pd.DataFrame(data_full["Lastname"].value_counts())
lastname_table.rename(columns={'Lastname': 'People_w_lastname'}, inplace = True)

lastname_table['Dead_mom_w_lastname'] = data_full.Lastname[(data_full.AgeBin > 1) & (data_full.Survived < 1) & (data_full.FamilySize > 1) & (data_full.IsFemale)].value_counts()
lastname_table['Dead_mom_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Dead_mom_w_lastname'] > 0, 'Dead_mom_w_lastname'] = 1
lastname_table['Dead_mom_w_lastname'] = lastname_table['Dead_mom_w_lastname'].astype(int)

lastname_table['Dead_kid_w_lastname'] = data_full.Lastname[(data_full.AgeBin == 1) & (data_full.Survived < 1) & (data_full.FamilySize > 1)].value_counts()
lastname_table['Dead_kid_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Dead_kid_w_lastname'] > 0, 'Dead_kid_w_lastname'] = 1
lastname_table['Dead_kid_w_lastname'] = lastname_table['Dead_kid_w_lastname'].astype(int)

lastname_table['Alive_dad_w_lastname'] = data_full.Lastname[(data_full.AgeBin > 1) & (data_full.Survived > 0) & (data_full.IsFemale==False) & (data_full.FamilySize > 1)].value_counts()
lastname_table['Alive_dad_w_lastname'].fillna(0, inplace=True)
lastname_table.loc[lastname_table['Alive_dad_w_lastname'] > 0, 'Alive_dad_w_lastname'] = 1
lastname_table['Alive_dad_w_lastname'] = lastname_table['Alive_dad_w_lastname'].astype(int)

# unique identifiers for lastname with more than 2 people
lastname_table["Lastname_id"]= pd.Categorical(lastname_table.index).codes
lastname_table.loc[lastname_table["People_w_lastname"] < 3, 'Lastname_id' ] = -1

# merge with the data_full table
data_full = pd.merge(data_full, lastname_table, left_on="Lastname",right_index=True,how='left', sort=False)
fig, (maxis1, maxis2, maxis3) = plt.subplots(1, 3,figsize=(15,6))
sns.pointplot(x='Dead_mom_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis1)
sns.pointplot(x='Alive_dad_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis2)
sns.pointplot(x='Dead_kid_w_lastname', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, ax = maxis3)
e = sns.FacetGrid(data_train, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep', order = [1, 2, 3], hue_order = ['male', 'female'])
e.add_legend()

# fill embared with mode
data_full['Embarked'] = data_full['Embarked'].fillna(data_full['Embarked'].mode().values[0])

# add dummies for the embarked as there is no linear relationship here
dummies = pd.get_dummies(data_full['Embarked'], prefix = 'embrk')
data_full = pd.concat([data_full, dummies], axis = 1)
data_full['Deck'] = data_full['Cabin'].str[:1]

# setup has cabin
data_full['HasCabin'] = 1
data_full.loc[data_full['Cabin'].isnull(), 'HasCabin'] = 0

fig = plt.figure(figsize=(15,8))
sns.pointplot(x='Deck', y='Survived', ci=95.0, hue = 'Sex', data = data_full, dodge=True, order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T'])

# transform decks to integers
data_full['Deck'] = pd.Categorical(data_full['Deck'].fillna('N')).codes
# create dummies from Pclass

dummies = pd.get_dummies(data_full['Pclass'], prefix = 'class')
data_full = pd.concat([data_full, dummies], axis = 1)
data_full.sample(10)
# get rid of superfluous columns
final_full_fram = data_full.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 
                                  'Embarked', 'Titles', 'Cabin',
                                  'Fare', 'Age', 'Ticket_id', 'Lastname',
                                  'Lastname_id', 'People_w_lastname', 
                                  'Alive_dad_w_lastname', 
                                  'Rev', 'People_on_ticket', 'Fancy',
                                  'Dr', 
                                 ], axis = 1)


# Split the data back to test and train
data_train1 = final_full_fram[final_full_fram['Survived'].isnull() == False]
print('Train columns (', data_train1.shape[1], ') with null values:\n', data_train1.isnull().sum())

print(" -"*20)

data_test1 = final_full_fram[final_full_fram['Survived'].isnull() == True]
print('Train columns (', data_test1.shape[1], ') with null values:\n', data_test1.isnull().sum())
# make sure we only have numerical values in our dataframe
#data_test1.info()

#check ranges
data_test1.describe()
features = data_train1.columns
features = features.drop(['PassengerId', 'Survived'])
print ("These are the features we will use for modeling:\n", features)

# create the np arrays that we need for training and testing
np_train_features = data_train1.as_matrix(columns = features)
print ("training features shape", np_train_features.shape)

np_train_labels = data_train1['Survived']
print ("training labels shape", np_train_labels.shape)

np_test_features = data_test1.as_matrix(columns = features)
print ("testing features shape", np_test_features.shape)

# fit scaler on full dataset
scaler = StandardScaler()
scaler.fit(np.concatenate((np_train_features, np_test_features), axis=0))

print(scaler.mean_)

np_train_features = scaler.transform(np_train_features)
np_test_features  = scaler.transform(np_test_features) 
selector = SelectKBest(f_classif, k=len(features))
selector.fit(data_train1[features], np_train_labels)

scores = selector.pvalues_
indices = np.argsort(scores)[::1]
print("Features p-values :")
for f in range(len(scores)):
    print("%.3e %s" % (scores[indices[f]],features[indices[f]]))
svc = svm.SVC()
parameters = {'kernel': ['linear', 'rbf'],
              'C':[1, 2, 4, 8]}

clf_svm = model_selection.GridSearchCV(svc, parameters, n_jobs = 2)
clf_svm.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_svm.best_score_, clf_svm.best_params_))
rf_regr = RandomForestClassifier()
parameters = {"min_samples_split" :[4]
            ,"n_estimators" : [50, 100]
            ,"criterion": ('gini','entropy')
             }

clf_rf = model_selection.GridSearchCV(rf_regr, parameters, n_jobs = 2)
clf_rf.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_rf.best_score_, clf_rf.best_params_))

xgb = XGBClassifier()
parameters = {'learning_rate': [0.05, 0.1, .25, 0.5], 
              'max_depth': [1,2,4,8], 
              'n_estimators': [50, 100]
             }

clf_xgb = model_selection.GridSearchCV(xgb, parameters, n_jobs = 2)
clf_xgb.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_xgb.best_score_, clf_xgb.best_params_))
nb = naive_bayes.GaussianNB()
parameters = {'priors': [None]}

clf_nb = model_selection.GridSearchCV(nb, parameters)
clf_nb.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_nb.best_score_, clf_nb.best_params_))
knn = neighbors.KNeighborsClassifier()
parameters = {'n_neighbors': [1,2,3,4,5,6,7],
              'weights': ['uniform', 'distance'], 
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
             }

clf_knn = model_selection.GridSearchCV(knn, parameters, n_jobs = 2)
clf_knn.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_knn.best_score_, clf_knn.best_params_))
gbk = GradientBoostingClassifier()

parameters = {
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100], 
            'max_depth': [2,3,4,5]   
             }

clf_gbk = model_selection.GridSearchCV(gbk, parameters, n_jobs = 2)
clf_gbk.fit(np_train_features, np_train_labels)

print("Best score {} came from {}".format(clf_gbk.best_score_, clf_gbk.best_params_))
rfc = RandomForestClassifier(n_estimators=50, min_samples_split=4, class_weight={0:0.675,1:0.325})

# for this one we will use kfold validation since we already have hyper parameters specified
kf = cross_validation.KFold(np_train_labels.shape[0], n_folds=3, random_state=42)

scores = cross_validation.cross_val_score(rfc, np_train_features, np_train_labels, cv=kf)
print("Accuracy on 3-fold XV: %0.3f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))

rfc.fit(np_train_features, np_train_labels)
score = rfc.score(np_train_features, np_train_labels)
print("Accuracy on full set: %0.3f" % (score*100))

print(" *"*15)
print("Feature importances in this model:")

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(len(features)):
    print("%0.2f%% %s" % (importances[indices[f]]*100, features[indices[f]]))


data_test1.loc[:, 'Survived-SVM'] = clf_svm.best_estimator_.predict(np_test_features)
print("SVM predicted number of survivors:", data_test1['Survived-SVM'].sum())

data_test1.loc[:, 'Survived-RF'] = clf_rf.best_estimator_.predict(np_test_features)
print("RF predicted number of survivors:", data_test1['Survived-RF'].sum())

data_test1.loc[:, 'Survived-RFC'] = rfc.predict(np_test_features)
print("RFC predicted number of survivors:", data_test1['Survived-RFC'].sum())

data_test1.loc[:, 'Survived-XGB'] = clf_xgb.best_estimator_.predict(np_test_features)
print("XGB predicted number of survivors:", data_test1['Survived-XGB'].sum())

# use the RFC data
data_test1.loc[:, 'Survived'] = data_test1['Survived-RFC'].astype(int)

submit = data_test1[['PassengerId', 'Survived']]
submit.to_csv("submit.csv", index=False)

data_test1[['Survived-SVM', 'Survived-RF', 'Survived-RFC', 'Survived-XGB', 'Survived']]

