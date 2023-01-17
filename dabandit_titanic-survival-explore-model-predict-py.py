import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



#grab datasets

train = pd.read_csv('../input/train.csv')

test_features = pd.read_csv('../input/test.csv')

test_labels = pd.read_csv('../input/genderclassmodel.csv')



#merge test to have Survival feature

test = pd.merge(test_features, test_labels, on='PassengerId')



#create full dataset that is a union of test and train.

frames = [train, test]

full = pd.concat(frames).reset_index()

full_x = full.drop(['index', 'Survived'], 1)

full_y = full[['Survived']]



#Re-partition data to create new train and test. 

features_train, features_test, labels_train, labels_test = train_test_split(full_x,full_y, test_size=.319, random_state=42)



#Combine train X and train Y. This will be used in our data exploration. 

train = pd.merge(features_train, labels_train, left_index=True, right_index=True)

train.head()

#import super cool nerd stuff and define some super cool nerd functions

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn import metrics

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import statsmodels.formula.api as smf

import patsy

import openpyxl

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import math

%matplotlib inline



def cross_tab(feature, target, kind, colors, stacked, grid):

    cross = pd.crosstab(feature, target)

    cross.plot(kind=kind, stacked=stacked, color=colors, grid=grid,figsize=(12,7))  

    

def display_scores(scores):

     print("Scores:", scores)

     print("Mean:", scores.mean())

     print("Standard deviation:", scores.std())



def show_importances(forest, X):    

    importances = forest.best_estimator_.feature_importances_

    print (importances)

    std = np.std([tree.feature_importances_ for tree in forest.best_estimator_.estimators_],

                 axis=0)

    indices = np.argsort(importances)[::-1]

    print ("Indices in order of importance: " , indices)

    print ("Indices type: " , type(indices))

    

    names = list(X.columns)

    print('Type: ', type(names))

    print('Names before sorted: ', names)

    names = list(np.array(names)[indices])

    #names =names.sort(key=indices)

    # Print the feature ranking

    print("Feature ranking:")



    for f in range(X.shape[1]):

        print("%d. feature %s (%f)" % (f + 1, names[f], importances[indices[f]]))



    # Plot the feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), importances[indices],

           color="r", yerr=std[indices], align="center")

    plt.xticks(range(X.shape[1]), names)

    plt.xlim([-1, X.shape[1]])

    plt.show()

train.info()

train.isnull().sum()
train.describe()
train.head()
#Initial visuals to test the waters 



#Age

cross_tab(feature=train.Age, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#Sex

cross_tab(feature=train.Sex, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#Pclass

cross_tab(feature=train.Pclass, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#SibSb

cross_tab(feature=train.SibSp, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#Parch

cross_tab(feature=train.Parch, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#Fare

cross_tab(feature=train.Fare, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)



#Embarked

cross_tab(feature=train.Embarked, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)
#Gender

train.groupby(['Sex']).agg({'Survived':['mean', 'count']})
#Pclass

train.groupby(['Pclass']).agg({'Survived':['mean', 'count']})
train.groupby(['Pclass','Sex']).agg({'Survived':['mean', 'count']})
#Testing Correlations between numerical datatypes.

sns.set(style='white', context='notebook', palette='bright')

fig, ax = plt.subplots(figsize=(15,10))  

foo = sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True, ax=ax)

#Age Drilldown

cross = pd.crosstab(train.Age, train.Survived.astype(bool))

agePlot = cross.plot(kind='bar', stacked=True, color=['red','blue'], grid=False,figsize=(16,10))  
#Test old/young hypothesis above.

def label_Age_Factor (row):

   if row['Age'] < 17 :

      return 'Under17'  

   if row['Age'] > 60 :

      return 'Over60'  

   return 'AgeBetween'



train['Age_Factor'] = train.apply (lambda row: label_Age_Factor (row),axis=1)



train.groupby(['Age_Factor']).agg({'Survived':['mean', 'count']})

null_ages = train.loc[train['Age'].isnull()]

len(null_ages)
#checking relative gender proportions between age categories.

train.groupby(['Age_Factor','Sex']).agg({'Survived':['mean', 'count']})
#checking relative Pclass proportions between age categories.

train.groupby(['Age_Factor','Pclass',"Sex"]).agg({'Survived':['mean', 'count']})
def find_between( s, first, last ):

        try:

            start = s.index( first ) + len( first )

            end = s.index( last, start )

            return s[start:end].replace(" ", "")

        except ValueError:

            return ""





def give_title(data):

    newCol = []



    for x in data.Name:

        newCol.append(find_between(x, ',', '.'))



    data['Title'] = newCol

    data.head()

    data.Title.value_counts()





    rare_title = ['Dona', 'Lady', 'theCountess','Capt', 'Col', 'Don', 

                    'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



    data.loc[data.Title == 'Mlle', 'Title'] = "Miss"

    data.loc[data.Title == 'Ms', 'Title'] = "Miss"

    data.loc[data.Title == 'Mme', 'Title'] = "Mrs"

    data.loc[data.Title.isin(rare_title), 'Title'] = "Rare"

    

give_title(full)

give_title(train)
#Title

cross_tab(feature=train.Title, target=train.Survived.astype(bool), 

          kind='bar', colors=['red','blue'], stacked=True, grid=False)
train.groupby(['Sex','Pclass','Title']).agg({'Survived':['mean', 'count']})
#Lets fill in null ages



#Table used to make predictions

null_age_map = full.groupby(['Sex' ,'Pclass','Title']).agg({'Age':['mean','std']}).reset_index()

null_age_map


def label_age_nulls(row, null_age_map, ages_pred):

    if math.isnan(row['Age'])==False:

        #print('Non Nan found')

        return row['Age']

    else:

        

        toR = null_age_map.loc[(null_age_map['Pclass']==row['Pclass'])&(null_age_map['Sex']==row['Sex'])&(null_age_map['Title']==row['Title'])]['Age']['mean'].iloc[0]

        toR_rounded = round(toR)

        print ('Est age: ',row['Pclass'],row['Sex'],row['Title'] ,toR_rounded)

        ages_pred.append(toR_rounded)

        return round(toR_rounded)
#Before nulls filled

train.Age.hist()
#fill nulls

ages_pred = []

train['Age'] = train.apply (lambda row: label_age_nulls (row , null_age_map, ages_pred),axis=1)
#After Nulls filled 

train.Age.hist()
#checking relative Pclass proportions between age categories.

train['Age_Factor'] = train.apply (lambda row: label_Age_Factor (row),axis=1)

train.groupby(['Age_Factor']).agg({'Survived':['mean', 'count']})
#Now that nulls filled...

#re-confirm relative Pclass proportions between age categories stayed the same.



train.groupby(['Age_Factor','Pclass',"Sex"]).agg({'Survived':['mean', 'count']})
#Embarked?

ports = pd.crosstab(train['Embarked'], train['Survived'])

print(ports)

dummy = ports.div(ports.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Port embarked')

dummy = plt.ylabel('Percentage')
train.groupby(['Embarked']).agg({'Survived':['mean', 'count']})


#Seeings whats up w embarked. 

embarkedClass = pd.crosstab(train.Embarked, train.Pclass)

dummy = embarkedClass.div(embarkedClass.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Embarked')

dummy = plt.ylabel('Percentage')





#Seeings whats up w embarked. 

embarkedS = pd.crosstab(train.Embarked, train.Sex)

dummy = embarkedS.div(embarkedS.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

dummy = plt.xlabel('Embarked')

dummy = plt.ylabel('Percentage')
#Grouping by Pclass to see if there is a difference amongst them

train.groupby(['Embarked','Pclass']).agg({'Survived':['mean', 'count']})
#Generate attribute for either second class C or third class Q.

def label_Embarked_Class_Differentiator_Thing (row):

   if (row['Embarked'] =='C') & (row['Pclass']==2) :

      return 'C_2'  

   if (row['Embarked'] =='Q') & (row['Pclass']==3) :

      return 'Q_3'   

   return 'Normal'



train['Embarked_Class_Outlier'] = train.apply (lambda row: label_Embarked_Class_Differentiator_Thing (row),axis=1)



train.groupby(['Embarked_Class_Outlier']).agg({'Survived':['mean', 'count']})
len(train['Ticket'].unique())
len(train)
indexes = []

grouped = train.groupby('Ticket')

for n, group in grouped:

    group = grouped.get_group(n)

    if (len(group) > 1):

        

        indexes.append(list(group['PassengerId'].values))

        

        print(group.loc[:,['Survived','Name', 'Age', 'SibSp', 'Parch']])



# flat_list = [item for sublist in indexes for item in sublist]

# ticketsGroup = train.loc[train['PassengerId'].isin(flat_list)]



# ticketGroupSex = pd.crosstab(ticketsGroup.Sex, ticketsGroup.Survived)

# dummy = ticketGroupSex.div(ticketGroupSex.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

# dummy = plt.xlabel('Sex')

# dummy = plt.ylabel('Percentage')
third_family_men = train.loc[(train['Pclass']==3)&(train['Sex']=='male')&((train['Parch']>0)|(train['SibSp']>0))]

print('3rd class male w/ Parch or SibSp Over 0 % Survived',sum(third_family_men.Survived)/len(third_family_men))





third_single_men = train.loc[(train['Pclass']==3)&(train['Sex']=='male')&(train['Parch']==0)&(train['SibSp']==0)]

print('3rd class male w/ Parch and SibSp equal to 0 % Survived',sum(third_single_men.Survived)/len(third_single_men))
third_fam_parch_men = train.loc[(train['Pclass']==3)&(train['Sex']=='male')&(train['Parch']>0)]

print('3rd class Male PARCH Over 0 % Survived',sum(third_fam_parch_men.Survived)/len(third_fam_parch_men))

third_sing_parch_men = train.loc[(train['Sex']=='male')&(train['Parch']==0)]

print('3rd class Male w/ Parch and SibSp equal to 0 % Survived',sum(third_sing_parch_men.Survived)/len(third_sing_parch_men))



print('')



third_fam_sibsp_men = train.loc[(train['Pclass']==3)&(train['Sex']=='male')&(train['SibSp']>0)]

print('3rd class Male PARCH Over 0 % Survived',sum(third_fam_sibsp_men.Survived)/len(third_fam_sibsp_men))

third_sing_sibsp_men = train.loc[(train['Pclass']==3)&(train['Sex']=='male')&(train['SibSp']==0)]

print('3rd class Male w/ Parch and SibSp equal to 0 % Survived',sum(third_sing_sibsp_men.Survived)/len(third_sing_sibsp_men))
def process_data(data,null_age_map):

    

    give_title(data)

    data.info()

    

    data['Age_Factor'] = data.apply (lambda row: label_Age_Factor (row),axis=1)

    data['IsAlone'] = 0

    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

    

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['Embarked_Class_Outlier'] = data.apply (lambda row: label_Embarked_Class_Differentiator_Thing (row),axis=1)



    data['Age'] = data.apply (lambda row: label_age_nulls (row , null_age_map, ages_pred),axis=1)

    

    #Map feature to ints so ml alg can understand

    encoder = LabelEncoder()

    data['Title'] = encoder.fit_transform(data["Title"])

    data['Sex'] = encoder.fit_transform(data["Sex"])

    data['Age_Factor'] = encoder.fit_transform(data["Age_Factor"])

    data['Embarked_Class_Outlier'] = encoder.fit_transform(data['Embarked_Class_Outlier'])

    

    #set nans to -1. Just in case we use

    data['Embarked'] = data['Embarked'].factorize()[0]

    

    #Map feature to ints so ML algs can understand

    embarked_cat = data["Embarked"]

    data['Embarked'] = encoder.fit_transform(embarked_cat)

    

    return data



features_train = process_data(features_train,null_age_map)

features_test = process_data(features_test,null_age_map)

train = process_data(train,null_age_map)



sns.set(style='white', context='notebook', palette='bright')

plt.figure(figsize=(12,10))

foo = sns.heatmap(train.corr(), vmax=0.6, square=True, annot=True)
#Training RandomForestClassifier

features_train = features_train[['Title','IsAlone','FamilySize', 'Pclass', 'Sex', 'Embarked_Class_Outlier', 'Age_Factor']]



features_test = features_test[['Title','IsAlone','FamilySize', 'Pclass', 'Sex', 'Embarked_Class_Outlier', 'Age_Factor']]





tuned_parameters = {'min_samples_split': [10,20,40,50], 'n_estimators': [30,40,50],

                     'criterion': ['gini','entropy']}





clf_RFC = GridSearchCV(RandomForestClassifier(random_state=42), tuned_parameters)

clf_RFC.fit(features_train, labels_train.values.reshape(len(labels_train),))





pred = clf_RFC.predict(features_test)

clf_RFC.best_params_



print('Score On CLF prediction test data', clf_RFC.score(features_test,labels_test.values.reshape(len(labels_test),) ))

accuracy_score(labels_test.values.reshape(len(labels_test),), pred)



show_importances(clf_RFC, features_train)



scores = cross_val_score(clf_RFC, features_train, labels_train.values.reshape(len(labels_train),), cv=5)

display_scores(scores)

print(accuracy_score(labels_test.values.reshape(len(labels_test),), pred))
features_train1 = features_train[['Sex', 'IsAlone','FamilySize','Pclass', 'Title', 'Embarked_Class_Outlier', 'Age_Factor']]

features_test1 = features_test[['Sex','IsAlone', 'FamilySize','Pclass', 'Title', 'Embarked_Class_Outlier', 'Age_Factor']]



tuned_parameters = {'penalty': ('l1', 'l2'), 'fit_intercept': [True,False]}



clf1 = GridSearchCV(LogisticRegression(dual=False), tuned_parameters)



clf1.fit(features_train1, labels_train.values.reshape(891,))



print(clf1.best_params_)



pred1 = clf1.predict(features_test1)

#labels_test.values.reshape(len(labels_test),)

accuracy_score(labels_test, pred1)

from sklearn.neighbors import KNeighborsClassifier

features_train2 = features_train[['Sex','IsAlone','FamilySize', 'Pclass', 'Title', 'Embarked_Class_Outlier','Age_Factor']]

features_test2 = features_test[['Sex','IsAlone', 'FamilySize','Pclass', 'Title', 'Embarked_Class_Outlier','Age_Factor']]



clf2 = KNeighborsClassifier(n_neighbors=4)

clf2.fit(features_train2,labels_train)

pred2 = clf2.predict(features_test2)

acc = accuracy_score(pred2, labels_test)

print(acc)
from sklearn.svm import SVC



features_train3 = features_train[['Sex','IsAlone', 'FamilySize','Pclass', 'Title', 'Embarked_Class_Outlier', 'Age_Factor']]

features_test3 = features_test[['Sex','IsAlone','FamilySize', 'Pclass', 'Title', 'Embarked_Class_Outlier', 'Age_Factor']]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]



clf3 =  GridSearchCV(SVC(), tuned_parameters)



clf3.fit(features_train3, labels_train.values.reshape(891,))



pred3 = clf3.predict(features_test3)



print(clf3.best_params_)



accuracy_score(labels_test, pred3)