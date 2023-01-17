#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#import data

train = pd.read_csv('../input/train.csv',index_col=0)
#view data

train.head(5)
#Number and types of columns

train.info()
#Look at statistics

train.describe()
#Look for missing data

sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
#Comparing survival by features

plt.figure(figsize=[15,5])

plt.subplot(1,3,1)

sns.countplot(x="Sex",data=train, hue="Survived")

plt.title('Gender comparison')



plt.subplot(1,3,2)

sns.countplot(x='Pclass',data=train, hue="Survived")

plt.title('Class comparison')



plt.subplot(1,3,3)

sns.countplot(x='Embarked',data=train, hue="Survived")

plt.title('Class comparison')
#Looking at spreads in data

plt.figure(figsize=[15,10])

plt.subplot(2,2,1)

sns.distplot(train['Age'].dropna(),kde=True,bins=10)

plt.xlim(0)

plt.title('Age spread')



plt.subplot(2,2,2)

sns.distplot(train['Fare'],kde=True,bins=50)

plt.xlim(0)

plt.title('Fare spread')



plt.subplot(2,2,3)

sns.countplot(x='SibSp',data=train)

plt.title('Sibling spread')



plt.subplot(2,2,4)

sns.countplot(x='Parch',data=train)

plt.title('Parent spread')
#Comparing age by features

plt.figure(figsize=[15,5])

plt.subplot(1,4,1)

sns.boxplot(x='Pclass',y='Age',data=train)

plt.title('Class comparison')



plt.subplot(1,4,2)

sns.boxplot(x='Sex',y='Age',data=train)

plt.title('Sex comparison')



plt.subplot(1,4,3)

sns.boxplot(x='SibSp',y='Age',data=train)

plt.title('Sibling comparison')



#Parents not used

plt.subplot(1,4,4)

sns.boxplot(x='Parch',y='Age',data=train)

plt.title('Parent comparison')
#Considered fare but not progressed with this as no clear correlation

sns.lmplot(x='Age',y='Fare',data=train)
#Creating a feature and observation set

train_age = train.drop(['Survived','Name','Parch','Ticket','Fare','Cabin','Embarked'],axis=1).dropna()

train_age_features = train_age.drop('Age',axis=1)

train_age_observations=train_age['Age']

print(train_age_features.shape)

print(train_age_observations.shape)
train_age_features.head(5)
#Split passenger classes

pclasses = pd.get_dummies(train_age_features['Pclass'],drop_first=True)

#Split sex

genders = pd.get_dummies(train_age_features['Sex'],drop_first=True)



genders.head(5)         
train_age_features.drop(['Pclass','Sex'],axis=1,inplace=True)
train_age_features_2 = pd.concat([train_age_features,pclasses,genders],axis=1)

train_age_features_2.head(5)
train_age_observations.head(5)
from sklearn.linear_model import LinearRegression

predict_age = LinearRegression()

predict_age.fit(train_age_features_2,train_age_observations)

print(train_age_features_2.columns.values)

print(predict_age.coef_)

print(predict_age.intercept_)
#define predictive function



def add_age(mylist):

    Age = mylist['Age']

    Pclass = mylist['Pclass']

    SibSp = mylist['SibSp']

    if mylist['Sex'] == 'Male':

        SibSp = 1

    else:

        Sex = 0

    

    if pd.isnull(Age):

        if Pclass == 1:

            predicted_age = predict_age.predict([[SibSp,0,0,Sex]]).round()[0]



        elif Pclass == 2:

            predicted_age = predict_age.predict([[SibSp,1,0,Sex]]).round()[0]



        else:

            predicted_age = predict_age.predict([[SibSp,0,1,Sex]]).round()[0]



    else:

        predicted_age = Age

    

    if predicted_age < 0:

        return 1

    else: 

        return predicted_age
train['Age_predict'] = train[['Age','Pclass','SibSp','Sex']].apply(add_age, axis =1)
#Age distibution of passengers

sns.distplot(train['Age_predict'],kde=True,bins=20,color='Red',label="Predicted ages")

sns.distplot(train['Age'].dropna(),kde=True,bins=20,color='Blue',label="Known ages")

plt.xlim(-5)

plt.legend()
#Showing predicted ages has no nulls now

sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
train.head(5)
print(train['Age'].min())

print(train['Age'].max())

print(train['Age_predict'].min())

print(train['Age_predict'].max())
train['Cabin'].describe()
train['Cabin'].head(5)
#Extract cabin numbers and letters from Cabin field

train['Cabin_number'] = train.Cabin.str.extract('(\d+)')

train['Cabin_deck_letter'] =train.Cabin.str.extract('(\D+)')
train['Cabin_deck_letter'].unique()
sns.countplot(x='Cabin_deck_letter',data=train, hue='Survived')
train['Name'].describe()
train['Name'].head()
train['Title'] = train.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0])

train['First_names'] = train.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[1])

train['Surname'] = train.Name.str.split(',').apply(lambda x: x[0])

train['First_name_length']=train.First_names.apply(len)

train['Surname_length']=train.Surname.apply(len)
train['First_names'].head()
train[['Title','First_names','Surname','First_name_length']].head(5)
#Exploring name lengths

plt.figure(figsize=[15,10])

plt.subplot(2,2,1)

sns.distplot(train['First_name_length'],kde=True,bins=10,color='Blue')

plt.title('First name')



plt.subplot(2,2,2)

sns.distplot(train['Surname_length'],kde=True,bins=10,color='Red')

plt.title('Surname')
#Exploring name lengths

plt.figure(figsize=[15,10])

plt.subplot(2,2,1)

sns.boxplot(x='Survived',y='First_name_length',data=train)

plt.title('First name')



plt.subplot(2,2,2)

sns.boxplot(x='Survived',y='Surname_length',data=train)

plt.title('Surname')
#Exploring titles

train.Title.unique()
# Survival chances vs title

plt.figure(figsize=[15,10])

plt.subplot(2,2,1)

sns.countplot(x='Title',hue='Survived',data=train)

plt.ylim(0)



plt.subplot(2,2,2)

sns.countplot(x='Title',hue='Survived',data=train)

plt.ylim(0,10)

plt.title('Zoomed')
train.head(5)
#Replace empty cabin letters with 'Unknown'

train.Cabin_deck_letter = train.Cabin_deck_letter.fillna(value="Unknown")
#Add dummies for categoric variables

Pclass_dummy = pd.get_dummies(train['Pclass'],drop_first = True)

Sex_dummy = pd.get_dummies(train['Sex'],drop_first = True)

Embarked_dummy = pd.get_dummies(train['Embarked'],drop_first = True)

Cabin_deck_letter_dummy = pd.get_dummies(train['Cabin_deck_letter'],drop_first = True)
#Drop categoric columns

train_features = train.drop(['Title','Survived','Name','Age','Ticket','Cabin','Cabin_number','Surname','First_names','Surname_length','Pclass','Sex','Embarked','Cabin_deck_letter'],axis=1)
#Add dummy columns. This is the feature dataframe.

X = pd.concat([train_features,Pclass_dummy,Sex_dummy,Embarked_dummy,Cabin_deck_letter_dummy],axis=1)

X.head(10)
#The observation series

y = train.Survived

y.head(5)
#Check they are the same shape

print(X.shape)

print(y.shape)
#Check the correlation between variables

plt.figure(figsize=[15,10])

Z=pd.concat([X,y],axis=1)

sns.heatmap(Z.corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")
# import the class

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import cross_val_score
# instantiate the model 

logreg_1 = LogisticRegression(solver='lbfgs',max_iter=1000,C=1,penalty='l2')
#1st score

scores_logreg_1 = cross_val_score(logreg_1, X, y, cv=10, scoring='accuracy')

print("Accuracy: %0.3f (+/- %0.3f)" % (scores_logreg_1.mean(), scores_logreg_1.std() * 2))
#Dropping individual features

testing_list = ['SibSp','Parch','Fare','Age_predict','First_name_length',[2,3],'male',['Q','S'],['B','C','D','E','F','F E','F G','G','T','Unknown']]



X_temp = X

for col in testing_list:

    X_temp= X.drop(col,axis=1)

    if len(X_temp.columns)>0:

        scores_logreg_temp = cross_val_score(logreg_1, X_temp, y, cv=10, scoring='accuracy')

        print("Dropping %s - Accuracy: %0.3f (+/- %0.3f)" % (col, scores_logreg_temp.mean(), scores_logreg_temp.std() * 2))
#Exhausting combinations of dropping features to check nothing is missed

testing_list = ['SibSp','Parch','Fare','Age_predict','First_name_length',[2,3],'male',['Q','S'],['B','C','D','E','F','F E','F G','G','T','Unknown']]

scores_list = []

sd_list =[]

accomp_features_list = []



X_temp = X

for col in testing_list:

    X_temp= X.drop(col,axis=1)

    scores_logreg_temp = cross_val_score(logreg_1, X_temp, y, cv=10, scoring='accuracy')

    scores_list.append(scores_logreg_temp.mean())

    sd_list.append(scores_logreg_temp.std())

    accomp_features_list.append(col)

    for col_2 in testing_list:

        X_temp_2 = X_temp

        if col != col_2:

            X_temp_2 = X_temp.drop(col_2,axis=1)    

        scores_logreg_temp = cross_val_score(logreg_1, X_temp_2, y, cv=10, scoring='accuracy')

        scores_list.append(scores_logreg_temp.mean())

        sd_list.append(scores_logreg_temp.std())

        accomp_features_list.append([col,col_2])

            

print("Complete")
#Show feature results in table

feature_testing = pd.concat([pd.Series(accomp_features_list),pd.Series(scores_list),pd.Series(sd_list)],axis=1)

feature_testing.columns = (['FeaturesDropped','PredictionAccuracy','StandardDeviation'])

feature_testing.head(5)
print(feature_testing.iloc[feature_testing['PredictionAccuracy'].idxmax()])

fig, ax1 = plt.subplots()



color = 'tab:red'

ax1.set_xlabel('Index')

ax1.set_ylabel('Accuracy', color=color)

ax1.plot(feature_testing.index, feature_testing.PredictionAccuracy, color=color, alpha=0.5)

ax1.tick_params(axis='y', labelcolor=color)

ax1.set_ylim([0.5,0.9])



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis



color = 'tab:blue'

ax2.set_ylabel('Standard Deviation', color=color)  # we already handled the x-label with ax1

ax2.plot(feature_testing.index, feature_testing.StandardDeviation, color=color, ls='--', alpha = 0.5)

ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim([0,0.1])



fig.tight_layout()  # otherwise the right y-label is slightly clipped
X_final_log_reg = X.drop(['Parch','First_name_length','T'],axis=1)

print(X_final_log_reg.shape)

X_final_log_reg.head(5)
#Model to be used for final run

logreg_final = LogisticRegression(solver='lbfgs',max_iter=1000,C=1,penalty='l2')

logreg_final.fit(X_final_log_reg,y)



predictions = logreg_final.predict(X_final_log_reg)
from sklearn.metrics import classification_report

print(classification_report(y,predictions))
#import data

test = pd.read_csv('../input/test.csv',index_col=0)
#view data

test.head(5)
#Checking for any missing data

sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
#Replace missing fare with mean

test.Fare = test.Fare.fillna(value = test.Fare.mean())

sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
#Checking data is similar to above

#Train

plt.figure(figsize=[15,20])

plt.subplot(4,2,1)

sns.distplot(train['Age'].dropna(),kde=True,bins=10)

plt.xlim(0)

plt.title('Train - Age spread')



plt.subplot(4,2,2)

sns.distplot(test['Age'].dropna(),kde=True,bins=10)

plt.xlim(0)

plt.title('Test - Age spread')



plt.subplot(4,2,3)

sns.distplot(train['Fare'],kde=True,bins=50)

plt.xlim(0)

plt.title('Train - Fare spread')



plt.subplot(4,2,4)

sns.distplot(test['Fare'],kde=True,bins=50)

plt.xlim(0)

plt.title('Test - Fare spread')



plt.subplot(4,2,5)

sns.countplot(x='SibSp',data=train)

plt.title('Train - Sibling spread')



plt.subplot(4,2,6)

sns.countplot(x='SibSp',data=test)

plt.title('Test - Sibling spread')



plt.subplot(4,2,7)

sns.countplot(x='Parch',data=train)

plt.title('Train - Parent spread')



plt.subplot(4,2,8)

sns.countplot(x='Parch',data=test)

plt.title('Test - Parent spread')
#Cleaning process as above

test['Cabin_number'] = test.Cabin.str.extract('(\d+)')

test['Cabin_deck_letter'] =test.Cabin.str.extract('(\D+)')

test['Age_predict'] = test[['Age','Pclass','SibSp','Sex']].apply(add_age, axis =1)

test['Title'] = test.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[0])

test['First_names'] = test.Name.str.split(',').apply(lambda x: x[1]).str.split('.').apply(lambda x: x[1])

test['Surname'] = test.Name.str.split(',').apply(lambda x: x[0])

test['First_name_length']=test.First_names.apply(len)

test['Surname_length']=test.Surname.apply(len)

test.Cabin_deck_letter = test.Cabin_deck_letter.fillna(value="Unknown")

Pclass_dummy = pd.get_dummies(test['Pclass'],drop_first = True)

Sex_dummy = pd.get_dummies(test['Sex'],drop_first = True)

Embarked_dummy = pd.get_dummies(test['Embarked'],drop_first = True)

Cabin_deck_letter_dummy = pd.get_dummies(test['Cabin_deck_letter'],drop_first = True)

test_features = test.drop(['Title','Name','Age','Ticket','Cabin','Cabin_number','Surname','First_names','Surname_length','Pclass','Sex','Embarked','Cabin_deck_letter'],axis=1)

X_test = pd.concat([test_features,Pclass_dummy,Sex_dummy,Embarked_dummy,Cabin_deck_letter_dummy],axis=1)

X_test.drop(['Parch','First_name_length'],axis=1,inplace=True)

print(X_test.shape)

X_test.head(10)
#Making predictions

predictions = logreg_final.predict(X_test)
#Output for submission

output = pd.concat([pd.Series(X_test.index),pd.Series(predictions)],axis=1)

output.columns = ['PassengerId','Survived']

output.to_csv('output.csv', index=False)