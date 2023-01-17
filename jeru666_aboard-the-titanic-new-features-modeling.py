import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print('Train set info')

print(train_df.info())

print('\n Train Data')

print(train_df.head())

print('\n Test Data')

print(test_df.head())
print(train_df.columns)

print(train_df.shape)
#--- Checking whether any missing values exist---

print(train_df.isnull().values.any())

print(test_df.isnull().values.any())
#--- Print list of columns containing Nan values ---

missing_col = train_df.columns[train_df.isnull().any()]

print(missing_col)



#--- Print list of columns containing Nan values in test set ---

missing_col_test = test_df.columns[test_df.isnull().any()]

print(missing_col_test)
#--- Number of rows with missing values in each column ---



list1 = []

for i in missing_col:

    print(i + ': {}'.format(train_df[i].isnull().sum()))

    list1.append(train_df[i].isnull().sum())

    

#sns.countplot(x = train_df[i].isnull().sum(), data = train_df)
#train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

nan_num = train_df['Age'].isnull().sum()

age_mean = train_df['Age'].mean()

age_std = train_df['Age'].std()

filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

train_df['Age'][train_df['Age'].isnull()==True] = filling



#--- Check whether nan values still exist in Age column ---

print(train_df['Age'].isnull().sum())
train_df['Cabin'].nunique()
#train_df = train_df.drop(['Cabin'], axis=1)



#--- Replace missing values with 'U' ---

train_df.Cabin.fillna('U', inplace=True)



#--- Create a new column with new cabin values containing first letter only and drop exsiting column ---

train_df['Cabin_new'] = train_df.Cabin.str[0]

train_df = train_df.drop(['Cabin'], axis=1)                                                

#--- Check whether nan values still exist in Cabin_new column ---

print(train_df['Cabin_new'].isnull().sum())
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)



#--- Checking if null values still exist ---

train_df['Embarked'].isnull().sum()
print(train_df.isnull().values.any())
#--- Number of rows with missing values in each column ---



list1 = []

for i in missing_col_test:

    print(i + ': {}'.format(test_df[i].isnull().sum()))

    list1.append(test_df[i].isnull().sum())
#test_df = test_df.drop(['Cabin'], axis=1) 



#--- Replace missing values with 'U' ---

test_df.Cabin.fillna('U', inplace=True)



#--- Create a new column with new cabin values containing first letter only and drop exsiting column ---

test_df['Cabin_new'] = test_df.Cabin.str[0]

test_df = test_df.drop(['Cabin'], axis=1)



#--- Check whether nan values still exist in Cabin_new column ---

print(test_df['Cabin_new'].isnull().sum())
#test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

nan_num = test_df['Age'].isnull().sum()

age_mean = test_df['Age'].mean()

age_std = test_df['Age'].std()

filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)

test_df['Age'][test_df['Age'].isnull()==True] = filling



#--- Check nan values still exist in Age column ---

print(test_df['Age'].isnull().sum())
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)



#--- Check nan values still exist in Age column ---

print(test_df['Fare'].isnull().sum())
#--- Final check to see if any Nan values remain in test_df ---

print(test_df.isnull().values.any())
#--- Statistics of the `Age` column ----

print(train_df['Age'].describe())



#--- Number of unique ages in the df ---

print(train_df['Age'].nunique())  
#--- Plotting the various ages of people with the count ---



fig, ax = plt.subplots(figsize=(20, 20))

ax = sns.countplot(x = 'Age', data = train_df)

plt.xticks( rotation=90)

plt.show()
def age_group(x):

    if 0 < x <= 3:

        return 1        #--- Infant

    elif 3 < x <= 12:

        return 2        #--- Child

    elif 12 < x <= 19:

        return 3        #--- Teen 

    elif 19 < x <= 29:  

        return 4        #--- Young Adult

    elif 29 < x <= 59:  

        return 5        #--- Adult

    else:

        return 6        #--- Aged 



train_df['age_group'] = train_df['Age'].apply(age_group)

test_df['age_group'] = test_df['Age'].apply(age_group)
#--- Plotting the distribution of the various age groups ---



ax = sns.countplot(x = 'age_group', data = train_df)

ax.set(xlabel='Age Groups', ylabel='Count')

age_group_list=['Infant','Child','Teen','Young Adult', 'Adult', 'Aged']

plt.xticks(range(6), age_group_list, rotation=45)

plt.show()
train_df['age_group'].value_counts()
Infant = train_df[train_df['age_group']==1]['Sex'].value_counts()

Child = train_df[train_df['age_group']==2]['Sex'].value_counts()

Teen = train_df[train_df['age_group']==3]['Sex'].value_counts()

Young_Adult = train_df[train_df['age_group']==4]['Sex'].value_counts()

Adult = train_df[train_df['age_group']==5]['Sex'].value_counts()

Aged = train_df[train_df['age_group']==6]['Sex'].value_counts()



df = pd.DataFrame([Infant, Child, Teen, Young_Adult, Adult, Aged])

df.index = ['Infant','Child','Teen','Young Adult', 'Adult', 'Aged']

df.plot(kind='bar',stacked=True, figsize=(16,8))
Infant = train_df[train_df['age_group']==1]['Pclass'].value_counts()

Child = train_df[train_df['age_group']==2]['Pclass'].value_counts()

Teen = train_df[train_df['age_group']==3]['Pclass'].value_counts()

Young_Adult = train_df[train_df['age_group']==4]['Pclass'].value_counts()

Adult = train_df[train_df['age_group']==5]['Pclass'].value_counts()

Aged = train_df[train_df['age_group']==6]['Pclass'].value_counts()



df = pd.DataFrame([Infant, Child, Teen, Young_Adult, Adult, Aged])

df.index = ['Infant','Child','Teen','Young Adult', 'Adult', 'Aged']

df.plot(kind='bar',stacked=True, figsize=(16,8))
#--- Plotting distribution of passengers based on their class ----

sns.countplot(x = 'Pclass', data = train_df)
Pclass1 = train_df[train_df['Pclass']==1]['Sex'].value_counts()

Pclass2 = train_df[train_df['Pclass']==2]['Sex'].value_counts()

Pclass3 = train_df[train_df['Pclass']==3]['Sex'].value_counts()



df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['Pclass1', 'Pclass2', 'Pclass3']

df.plot(kind='bar',stacked=True, figsize=(12,8))
#--- Statistics of the column ---

train_df['Fare'].describe()
#--- Finding number of unique fares ---

train_df['Fare'].nunique()
def fare_group(x):

    if x == 0:

        return 0        #--- Free

    elif 1 < x <= 50:

        return 1        #--- Fare1

    elif 50 < x <= 100:

        return 2        #--- Fare2 

    elif 100 < x <= 200:  

        return 3        #--- Fare3

    elif 200 < x <= 300:  

        return 4        #--- Fare4

    elif 300 < x <= 400:  

        return 5        #--- Fare5

    elif 400 < x <= 500:  

        return 6        #--- Fare6

    else:

        return 7        #--- Fare7 



train_df['fare_group'] = train_df['Fare'].apply(fare_group)

test_df['fare_group'] = test_df['Fare'].apply(fare_group)
#--- Plotting the distribution of the various age groups ---



ax = sns.countplot(x = 'fare_group', data = train_df)

ax.set(xlabel='Fare Groups', ylabel='Count')

fare_group_list=['Free', 'Fare1', 'Fare2', 'Fare3', 'Fare4', 'Fare5', 'Fare6', 'Fare7']

plt.xticks(range(len(fare_group_list)), fare_group_list, rotation=45)

plt.show()
print(train_df['fare_group'].value_counts())
train_df['Embarked'].describe()
train_df['Embarked'].value_counts()
#--- Plotting the distribution ---

ax = sns.countplot(x = 'Embarked', data = train_df)

ax.set(xlabel='Embarked Ports', ylabel='Count')

embarked_list=['Southampton', 'Cherbourg', 'Queenstown']

plt.xticks(range(len(embarked_list)), embarked_list, rotation=45)

plt.show()
sns.countplot(x = 'Sex', data = train_df)
print(train_df['Name'].head(20))
train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(train_df.Title.value_counts())



test_df['Title'] = test_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['TitleCat'] = train_df['Title']

train_df.TitleCat.replace(to_replace=['Rev','Dr','Col','Major','Mlle','Ms','Countess','Capt','Dona','Don','Sir','Lady','Jonkheer','Mme'],

                        value=0, inplace=True)

train_df.TitleCat.replace('Mr',1,inplace=True)

train_df.TitleCat.replace('Miss',2,inplace=True)

train_df.TitleCat.replace('Mrs',3,inplace=True)

train_df.TitleCat.replace('Master',4,inplace=True)                                            

print(train_df.TitleCat.value_counts())
test_df['TitleCat'] = test_df['Title']

test_df.TitleCat.replace(to_replace=['Rev','Dr','Col','Major','Mlle','Ms','Countess','Capt','Dona','Don','Sir','Lady','Jonkheer','Mme'],

                        value=0, inplace=True)

test_df.TitleCat.replace('Mr',1,inplace=True)

test_df.TitleCat.replace('Miss',2,inplace=True)

test_df.TitleCat.replace('Mrs',3,inplace=True)

test_df.TitleCat.replace('Master',4,inplace=True) 
train_df = train_df.drop(['Name'], axis=1) 

test_df = test_df.drop(['Name'], axis=1) 



train_df = train_df.drop(['Title'], axis=1) 

test_df = test_df.drop(['Title'], axis=1) 
#--- Get number of unique ticket numbers ---

print(train_df['Ticket'].nunique() )
print(train_df['Ticket'].describe() )
#--- Create a new column with new ticket values containing first letter only and drop exsiting column ---

train_df['Ticket_new'] = train_df.Ticket.str[0]

train_df = train_df.drop(['Ticket'], axis=1)

test_df['Ticket_new'] = test_df.Ticket.str[0]

test_df = test_df.drop(['Ticket'], axis=1)



#--- Check whether nan values still exist in Cabin_new column ---

print(train_df['Ticket_new'].isnull().sum())

print(test_df['Ticket_new'].isnull().sum())
#--- Get number of unique values ---

print(train_df['Parch'].nunique() )

print(train_df['SibSp'].nunique() )
#--- Plotting the distribution of these columns ---

sns.countplot(x = 'SibSp', data = train_df)
sns.countplot(x = 'Parch', data = train_df)
train_df['Parch_SibSp'] = np.where((train_df['Parch']>0) & (train_df['SibSp']>0), 1, 0)

test_df['Parch_SibSp'] = np.where((test_df['Parch']>0) & (test_df['SibSp']>0), 1, 0)



sns.countplot(x = 'Parch_SibSp', data = train_df)
train_df['Parch_SibSp'].value_counts()
train_df['Family_members'] = train_df['Parch'] + train_df['SibSp'] + 1   # 1 -> self

test_df['Family_members'] = test_df['Parch'] + test_df['SibSp']



sns.countplot(x = 'Family_members', data = train_df)
train_df['Family_members'].value_counts()
def family_size(x):

    if x == 1:

        return 1        #--- Singleton

    elif 1 < x <= 4:

        return 2        #--- Small family

    else:

        return 3        #--- Large family 



train_df['Family_size'] = train_df['Family_members'].apply(family_size)

test_df['Family_size'] = test_df['Family_members'].apply(family_size)
ax = sns.countplot(x = 'Family_size', data = train_df)

ax.set(xlabel='Family size', ylabel='Count')

family_list=['Singleton', 'Small family', 'Large family']

plt.xticks(range(len(family_list)), family_list, rotation=45)

plt.show()



train_df['Family_size'].value_counts()
sns.countplot(x = 'Survived', data = train_df)
survived = train_df[train_df['Survived']==1]['Sex'].value_counts()

dead = train_df[train_df['Survived']==0]['Sex'].value_counts()



df = pd.DataFrame([survived, dead])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(8,8))
survived = train_df[train_df['Survived']==1]['Pclass'].value_counts()

dead = train_df[train_df['Survived']==0]['Pclass'].value_counts()



df = pd.DataFrame([survived, dead])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(8,8))
Infant = train_df[train_df['age_group']==1]['Survived'].value_counts()

Child = train_df[train_df['age_group']==2]['Survived'].value_counts()

Teen = train_df[train_df['age_group']==3]['Survived'].value_counts()

Young_Adult = train_df[train_df['age_group']==4]['Survived'].value_counts()

Adult = train_df[train_df['age_group']==5]['Survived'].value_counts()

Aged = train_df[train_df['age_group']==6]['Survived'].value_counts()



df = pd.DataFrame([Infant, Child, Teen, Young_Adult, Adult, Aged])

df.index = ['Infant','Child','Teen','Young Adult', 'Adult', 'Aged']

df.plot(kind='bar',stacked=True, figsize=(16,8))
Free = train_df[train_df['fare_group']==0]['Survived'].value_counts()

Fare1 = train_df[train_df['fare_group']==1]['Survived'].value_counts()

Fare2 = train_df[train_df['fare_group']==2]['Survived'].value_counts()

Fare3 = train_df[train_df['fare_group']==3]['Survived'].value_counts()

Fare4 = train_df[train_df['fare_group']==4]['Survived'].value_counts()

Fare5 = train_df[train_df['fare_group']==5]['Survived'].value_counts()

Fare6 = train_df[train_df['fare_group']==6]['Survived'].value_counts()

Fare7 = train_df[train_df['fare_group']==7]['Survived'].value_counts()



df = pd.DataFrame([Free, Fare1, Fare2, Fare3, Fare4, Fare5, Fare6, Fare7])

df.index = ['Free', 'Fare1', 'Fare2', 'Fare3', 'Fare4', 'Fare5', 'Fare6', 'Fare7']

df.plot(kind='bar',stacked=True, figsize=(16,8))
Parch_SibSp_0 = train_df[train_df['Parch_SibSp']==0]['Survived'].value_counts()

Parch_SibSp_1 = train_df[train_df['Parch_SibSp']==1]['Survived'].value_counts()



df = pd.DataFrame([Parch_SibSp_0, Parch_SibSp_1])

df.index = ['Parch_SibSp_0', 'Parch_SibSp_1']

df.plot(kind='bar',stacked=True, figsize=(8,8))
Singleton = train_df[train_df['Family_size']==1]['Survived'].value_counts()

small_family = train_df[train_df['Family_size']==2]['Survived'].value_counts()

large_family = train_df[train_df['Family_size']==3]['Survived'].value_counts()



df = pd.DataFrame([Singleton, small_family, large_family])

df.index = ['Singleton', 'small_family', 'large_family']

df.plot(kind='bar',stacked=True, figsize=(10,8))
''' ax = sns.countplot(x = train_df.dtypes.value_counts(), data = train_df)

ax.set(xlabel='Data types', ylabel='Count')

family_list=['Singleton', 'Small family', 'Large family']

plt.xticks(range(len(family_list)), family_list, rotation=45)

plt.show()

''' 

#sns.barplot( y = train_df.dtypes.value_counts(), data = train_df)
print(train_df.dtypes)
#--- Checking the  unique values in both these columns ---



print(train_df['Sex'].unique())

print(train_df['Embarked'].unique())

print(train_df['Cabin_new'].unique())

print(train_df['Ticket_new'].unique())
#---For 'Sex' and 'Embarked' columns ---



#--- Replacing the unique string  values with numerical values ---



train_df['Sex'].replace( 'male', 1, inplace=True)

train_df['Sex'].replace( 'female', 0, inplace=True)

test_df['Sex'].replace( 'male', 1, inplace=True)

test_df['Sex'].replace( 'female', 0, inplace=True)



#--- Convert the column to type `int8` ---

train_df.Sex = train_df.Sex.astype(np.int8)

test_df.Sex = test_df.Sex.astype(np.int8)



train_df['Embarked'].replace( 'S', 1, inplace=True)

train_df['Embarked'].replace( 'C', 2, inplace=True)

train_df['Embarked'].replace( 'Q', 3, inplace=True)

test_df['Embarked'].replace( 'S', 1, inplace=True)

test_df['Embarked'].replace( 'C', 2, inplace=True)

test_df['Embarked'].replace( 'Q', 3, inplace=True)



#--- Convert the column to type `int8` ---

train_df.Embarked = train_df.Embarked.astype(np.int8)

test_df.Embarked = test_df.Embarked.astype(np.int8)
cabin_dummies = pd.get_dummies(train_df['Cabin_new'], prefix='Cabin_new')

train_df = pd.concat([train_df, cabin_dummies], axis=1)

train_df.drop('Cabin_new', axis=1, inplace=True)



cabin_dummies1 = pd.get_dummies(train_df['Ticket_new'], prefix='Ticket_new')

train_df = pd.concat([train_df,cabin_dummies1], axis=1)

train_df.drop('Ticket_new', axis=1, inplace=True)



cabin_dummies2 = pd.get_dummies(test_df['Cabin_new'], prefix='Cabin_new')

test_df = pd.concat([test_df, cabin_dummies2], axis=1)

test_df.drop('Cabin_new', axis=1, inplace=True)



cabin_dummies3 = pd.get_dummies(test_df['Ticket_new'], prefix='Ticket_new')

test_df = pd.concat([test_df,cabin_dummies3], axis=1)

test_df.drop('Ticket_new', axis=1, inplace=True)
#--- Checking the datatypes again ---

print(train_df.dtypes)

print('\n')

print(test_df.dtypes)
train_df = train_df.drop('Cabin_new_T', 1)

train_df = train_df.drop('Ticket_new_5', 1)

train_df = train_df.drop('Ticket_new_8', 1)
target = train_df['Survived']

train_df = train_df.drop(['Survived'], axis=1)   
print(len(train_df.columns))

print(len(test_df.columns))
from sklearn.model_selection import cross_val_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

XGB_clf = XGBClassifier(n_estimators=10, learning_rate=0.1, max_depth=3, min_child_weight=1, gamma=0.1, subsample=1)

XGB_clf.fit(train_df, target)

pred = XGB_clf.predict(test_df)

test_labels = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':list(map(int,pred))})

test_labels.to_csv('XGBoost_Title_7.csv',index=False)

print('DONE!!!')
model = LogisticRegression()

score = cross_val_score(model, train_df, target, cv=10)

print('Score - {} :: Mean score - {} '.format(score.mean(),score))
DTC = DecisionTreeClassifier()

score = cross_val_score(DTC, train_df, target, cv=10)

print('Score - {} :: Mean score - {} '.format(score.mean(),score))
#''' 

GBC = GradientBoostingClassifier()

score = cross_val_score(GBC, train_df, target, cv=10)

print('Score - {} :: Mean score - {} '.format(score.mean(),score))

#'''
#'''

RFC = RandomForestClassifier()

score = cross_val_score(RFC, train_df, target, cv=10)

print('Score - {} :: Mean score - {} '.format(score.mean(),score))

#'''
#model = AdaBoostClassifier()

#score = cross_val_score(model, train_df, target, cv=5)

#print('Score - {} :: Mean score - {} '.format(score.mean(),score))
#'''

GBC.fit(train_df, target)

print (GBC)



RFC.fit(train_df, target)

print (RFC)



DTC.fit(train_df, target)

print (DTC)

#'''
#--- List of important features for Gradient Boosting Classifier ---

#'''

features_list = train_df.columns.values

feature_importance = GBC.feature_importances_

sorted_idx = np.argsort(feature_importance)



print(sorted_idx)



plt.figure(figsize=(15, 15))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()

#'''
#--- List of important features for Random Forest Classifier ---

#'''

features_list = train_df.columns.values

feature_importance = RFC.feature_importances_

sorted_idx = np.argsort(feature_importance)



print(sorted_idx)



plt.figure(figsize=(15, 15))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()

#'''
#--- List of important features for Decision Tree Classifier ---

#'''

features_list = train_df.columns.values

feature_importance = DTC.feature_importances_

sorted_idx = np.argsort(feature_importance)



print(sorted_idx)



plt.figure(figsize=(15, 15))

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')

plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])

plt.xlabel('Importance')

plt.title('Feature importances')

plt.draw()

plt.show()

#'''
#--- Predicting Gradient boost result for test data ---

y_GBC = GBC.predict(test_df)
#--- Predicting Random Forest result for test data ---

y_RFC = RFC.predict(test_df)
#--- Predicting Decision Tree result for test data ---

y_DTC = DTC.predict(test_df)
#'''

final = pd.DataFrame()

final['PassengerId'] = test_df['PassengerId']

final['Survived'] = y_GBC

final.to_csv('Gradient_Boosting_Title_8.csv', index=False)

print('DONE!!')

#'''
#'''

final = pd.DataFrame()

final['PassengerId'] = test_df['PassengerId']

final['Survived'] = y_RFC

final.to_csv('Random_Forest_Title_9.csv', index=False)

print('DONE!!')

#'''
#'''

final = pd.DataFrame()

final['PassengerId'] = test_df['PassengerId']

final['Survived'] = y_DTC

final.to_csv('Decision_Tree_Title_10.csv', index=False)

print('DONE!!')

#'''