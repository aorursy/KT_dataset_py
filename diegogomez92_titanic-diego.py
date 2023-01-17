# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



#this data set has all females surviving and all males not surviving

#get ball-park estimate about what feautures are important in the data set

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")



#testing data has no labels (i.e., does not include 'Survived' variable)

test = pd.read_csv("../input/titanic/test.csv")

#training set has labels and is usde to train our model

train = pd.read_csv("../input/titanic/train.csv")
train = pd.read_csv("/kaggle/input/titanic/train.csv")

#show top 5 rows of the data set 

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test.head()
#query train data by column name 'Sex', where it equals 'female'

women = train.loc[train.Sex == 'female']["Survived"]



#sum of women that survived / total number of women

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train.loc[train.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
#import Shallow Machine Learning library (i.e., sklearn)

#import the Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier



#DEPENDENT VARIABLE

y = train["Survived"]



#INDIPENDENT VARIABLES

#the features we will include in our model

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train[features])

X_test = pd.get_dummies(test[features])



#100 random forest trees

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y) #fit the model

predictions = model.predict(X_test) 



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
#get an idea aboutt the data set

train.shape
#Descriptive Statistics

train.describe()
#object data types are a type of data in Pandas

train.describe(include=['O'])
train.info()
#There are 177 rows with missing Age, 687 rows with missing Cabin and 2 rows with missing Embarked information.

train.isnull().sum()
print(train.shape)

test.shape
test.info()
#There are 86 rows with missing Age, 327 rows with missing Cabin and 1 row with missing Fare information.



#shows total number of missing values

test.isnull().sum()
#shows number/percent of survived

survived = train[train['Survived'] == 1]

#shows number/percent of those that did not survived

not_survived = train[train['Survived'] == 0]



print ("Survived: %i (%.1f%%)"%(len(survived), float(len(survived))/len(train)*100.0))

print ("Not Survived: %i (%.1f%%)"%(len(not_survived), float(len(not_survived))/len(train)*100.0))

print ("Total: %i"%len(train))
train.Pclass.value_counts()
#count number of survived/did not survive per class

train.groupby('Pclass').Survived.value_counts()
#percent survived by class

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
import seaborn as sns

import matplotlib.pyplot as plt



#train.groupby('Pclass').Survived.mean().plot(kind='bar')

sns.barplot(x='Pclass', y='Survived', data=train)
#train.groupby('Sex').Survived.mean().plot(kind='bar')

sns.barplot(x='Sex', y='Survived', data=train)

#number of passengers by sex 

train.Sex.value_counts()
#Female ratio of surviavl indicates females more likely to have survived

train.groupby('Sex').Survived.value_counts()
#mean values for survival 

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
tab = pd.crosstab(train['Pclass'], train['Sex'])

print (tab)



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=False)

plt.xlabel('Pclass')

plt.ylabel('Percentage')
# Shows that Males in all classes less likely to survive than Females in corresponding classes

# Shows that Males in 2nd and 3rd classe less likely to survive than Males in 1st class

sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=3, data=train)
#embark = port of embarkment (C = Cherbourg, Q = Queenstown, S = Southampton)

sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)
#count for each port of embarkment (highest is S)

train.Embarked.value_counts()
#appears embarked from Q less than 50% chance of survival (second worst)

#emarkment S passengers much more likely to die (worst)

#C passengers more likely to survive (least worst)

train.groupby('Embarked').Survived.value_counts()

#mean survival chance by each port (C passengers over 50% chance of survival)

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
#different plot but decided to go with the one below 

#train.groupby('Embarked').Survived.mean().plot(kind='bar')



#chose this plot because it is easier to decipher that C embarked passengers had a higher chance survival 

sns.barplot(x='Embarked', y='Survived', data=train)
#the majority of passengers did not have parents / children

train.Parch.value_counts()
#passengers with just 1 parents / children had over a 50% chance of survival

train.groupby('Parch').Survived.value_counts()
#passengers with 1 to 3 parents / children had a 50% or above chance of survival

train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
#train.groupby('Parch').Survived.mean().plot(kind='bar')



# ci=None will hide the error bar

#decided to move it because it makes the barplot distracting

sns.barplot(x='Parch', y='Survived', ci=None,data=train) 
#most did not have siblings

train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
#higher survivial mean for passengers with just one sibling/spouse 

train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
#train.groupby('SibSp').Survived.mean().plot(kind='bar')

sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar
#use violin plot since Age is a continuous feature

#1) Age on y-axis; 2) a. Embarked b. Pclass c. Sex on the x-axis per violin plot



fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(131)

ax2 = fig.add_subplot(132)

ax3 = fig.add_subplot(133)



sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True, ax=ax1)

sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True, ax=ax2)

sns.violinplot(x="Sex", y="Age", hue="Survived", data=train, split=True, ax=ax3)
#shows negative and positive correlations

#Correlations go from 1 to -1

#positive numbers indicate a positive correlation among features 

#negative numbers indicate a negative correlation among features 



#dimension size for heat map

plt.figure(figsize=(10,10))

#make the heat map a squares and the heat spectrum vmax at 0.8

sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.8, square=True, annot=True)


#merge the train and test data set

combined_data = [train, test] 



#create new column with name titles, such as Mr., Mrs., etc.

for dataset in combined_data:

    dataset['NAME_TITLE'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

    

    

#replace uncommon name titles to a category named 'other'

for dataset in combined_data:

    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].replace('Mlle', 'Miss')

    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].replace('Ms', 'Miss')

    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].replace('Mme', 'Mrs')



#show mean survival rate by name title 

#notice females ['Mrs.' and 'Miss'] have higher survival mean

train[['NAME_TITLE', 'Survived']].groupby(['NAME_TITLE'], as_index=False).mean()
#turn name titles into numerical values

quant_title_name = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in combined_data:

    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].map(quant_title_name)

    dataset['NAME_TITLE'] = dataset['NAME_TITLE'].fillna(0)

    

#column 'Name_Title' now has numerical values

train.head()
#males == 0

#fmeales == 1

for dataset in combined_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#notice the 'Sex' column 

train.head()
#check which category occurs more often

train.Embarked.value_counts()
#since 'S' occcures more often, we will replace missing values with 'S'

for dataset in combined_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

train.head()
#convert categories in 'Embarked' to numerical values

for dataset in combined_data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

train.head()
#get the mean and standard deviation for age

#count the number of null values in the age column 

for dataset in combined_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    null_age_count = dataset['Age'].isnull().sum()

    

    #add random numbers to nill_age_list that are the outcome of subtracting/adding mean_age and std_age

    null_age_count_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=null_age_count)

    dataset['Age'][np.isnan(dataset['Age'])] = null_age_count_list

    dataset['Age'] = dataset['Age'].astype(int)

    

#create a new column AgeRange with 

#break it down into 5 age ranges 

train['AgeRange'] = pd.cut(train['Age'], 5)



#see what age ranges were more likely to survive 

print (train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean())
#see new column AgeRange 

train.head()
#break down Age column by AgeRange

for dataset in combined_data:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
#age column now coded with numerical values 

train.head()
#we chose median instead of mean b/c of the huge std for Fare

print(train['Fare'].median())
#replace missing values with the median value 

for dataset in combined_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
#create a Fare range

#divided into four ranges

#get mean values for each range



#notice the higher the fare the more likely to survive

train['FareRange'] = pd.qcut(train['Fare'], 4)

print (train[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean())
#code Fare values by Fare ranges 

for dataset in combined_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
#Fare column now has numerical values for each range

train.head()
#add values for SibSP and Parch and add 1 to account for lone tavelers

for dataset in combined_data:

    dataset['Size_of_Family'] = dataset['SibSp'] +  dataset['Parch'] + 1



print (train[['Size_of_Family', 'Survived']].groupby(['Size_of_Family'], as_index=False).mean())
#compare traveling alone vs traveling with a family 

for dataset in combined_data:

    dataset['Alone'] = 0

    dataset.loc[dataset['Size_of_Family'] == 1, 'Alone'] = 1

    

#traveling alone == only 30 percent chance of survival 

print (train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean())
#drop passenger Id from training since we do not need it 

#but we kept for testing data, as it is required to submit it on Kaggle

drop_features = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Size_of_Family']

train = train.drop(drop_features, axis=1)

test = test.drop(drop_features, axis=1)

train = train.drop(['PassengerId', 'AgeRange', 'FareRange'], axis=1)
#modified training data set

train.head()
#modified testing data set

test.head()
X_train = train.drop('Survived', axis=1)

y_train = train['Survived']

X_test = test.drop("PassengerId", axis=1).copy()



X_train.shape, y_train.shape, X_test.shape
# Importing Classifier Modules ()

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)



print (str(acc_log_reg) + ' percent')
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred_decision_tree = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)

print (acc_decision_tree)
clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)

y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, y_train) * 100, 2)

print (acc_random_forest)