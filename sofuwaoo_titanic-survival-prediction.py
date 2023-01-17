import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import plotly.express as px



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
train_data = pd.read_csv('../input/titanic/train.csv')

train_data.head()
train_data.describe(include='all')
train_data.isnull().sum()
#Extracting the title from the name using regex. Since names are unique, it would be better to work with just titles.

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train_data['Title'].unique()
#Rather than replacing the null values with the median age across the dataset, i choose to replace the null values with the

#median age for each title.

train_data.groupby(['Title'])['Age'].median()
train_data['Age'] = train_data['Age'].fillna(train_data.groupby(['Title'])['Age'].transform('median'))

train_data['Age'].isnull().sum()
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

train_data['Embarked'].isnull().sum()
train_data['Title'].replace(to_replace=['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 

                                        'Countess','Jonkheer'], value='Rare', inplace=True)
#Mlle representing Mademoiselle, according to Google is a title for an unmarried woman, so i am replacing it with 'Miss' which is 

#commonly used for an unmarried woman.

train_data['Title'].replace(to_replace=['Mlle','Ms'], value='Miss', inplace=True)
train_data['Title'].replace({'Master':'Mr'}, inplace=True)

train_data['Title'].replace({'Mme':'Mrs'}, inplace=True) #Madame refers to married woman. Replacing it with 'Mrs' which is more

#common.

train_data['Title'].unique()
train_data['Embarked'].replace({'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)
#By adding the SibSp and Parch columns i can get the family size. 1 is added to represent those traveling without a sibling, 

#spouse, parent or child.

train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch'] + 1
#People traveling alone would be represented by 1 and those who are traveling with others will be represented by 0.

train_data['Traveling_Alone'] = np.where(train_data['Family_Size'] > 1, 0, 1)
#Creating age bins. This would make it easier to perform analysis and it would also be used in my machine learning model.

bins= [0, 12, 19, 59, np.inf]



#This categorizes children as 0-12, Teenagers as 13-19, Adults 20-59 and Seniors as 60 and above

labels= ['Children', 'Teenagers', 'Adults', 'Seniors']



train_data['Age_Group'] = pd.cut(train_data['Age'], bins=bins, labels=labels)
#Running a check to get the titles in the children age group which should either be Mr or Miss.

train_check1 = train_data[train_data['Age_Group']=='Children']

train_check1['Title'].unique()
#Running a check to get the titles in teenagers age group which should either be Mr or Miss.

train_check2 = train_data[train_data['Age_Group']=='Teenagers']

train_check2['Title'].unique()
#Checking to see the rows that have title as Mrs. This could be a data collection or data entry error.

train_check2[train_check2['Title']=='Mrs']
#Replacing the Title for Teenagers that have 'Mrs' with 'Miss'

train_data['Title'] = np.where((train_data.Age_Group.values == 'Teenagers') & 

                               (train_data.Title.values == 'Mrs'),'Miss', train_data.Title.values)
#Running a another check to ensure that the title for the teenagers age group is either Mr or Miss

train_check3 = train_data[train_data['Age_Group']=='Teenagers']

train_check3['Title'].unique()
train_data.rename({'Sex':'Gender'}, axis = 1, inplace = True)
#Dropping columns that won't be used in building the ML model and predicting

train_data.drop(['Cabin','Ticket','Name','SibSp','Parch'], axis=1, inplace=True)
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
test_data.describe(include='all')
test_data.isnull().sum()
#Extracting the title from the name using regex. Since names are unique, it would be better to work with just titles.

test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

test_data['Title'].unique()
test_data['Age'] = test_data['Age'].fillna(test_data.groupby(['Title'])['Age'].transform('median'))

test_data['Age'].isnull().sum()
#Getting the index for the age that still has a null values

age_with_nan = [index for index, row in test_data[['Age']].iterrows() if row.isnull().any()]

print(age_with_nan)
test_data.iloc[88]
#I'll be imputing the null values with the median age of Ms from the train data which is 28.

test_data['Age'].fillna(28,inplace=True)

test_data['Age'].isnull().sum()
test_data.groupby(['Pclass'])['Fare'].median()
#Replacing the null values with the median fare for each Pclass.

test_data['Fare'] = test_data['Fare'].fillna(test_data.groupby(['Pclass'])['Fare'].transform('median'))

test_data['Fare'].isnull().sum()
test_data['Title'].replace(to_replace=['Col', 'Rev', 'Dr', 'Dona'], value='Rare', inplace=True)
test_data['Title'].replace({'Master':'Mr'}, inplace=True)

test_data['Title'].replace({'Ms':'Miss'}, inplace=True)

test_data['Title'].unique()
test_data['Embarked'].replace({'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)
#By adding the SibSp and Parch columns i can get the family size. 1 is added to represent those traveling without a sibling, 

#spouse, parent or child.

test_data['Family_Size'] = test_data['SibSp'] + test_data['Parch'] + 1
#People traveling alone would be represented by 1 and those who are traveling with others will be represented by 0.

test_data['Traveling_Alone'] = np.where(test_data['Family_Size'] > 1, 0, 1)
#Creating age bins. This would make it easier to perform analysis and it would also be used in my machine learning model.

bins= [0, 12, 19, 59, np.inf]



#This categorizes children as 0-12, Teenagers as 13-19, Adults 20-59 and Seniors as 60 and above



test_data['Age_Group'] = pd.cut(test_data['Age'], bins=bins, labels=labels)
#Running a check to get the titles in the children age group which should either be Mr or Miss.

test_check1 = test_data[test_data['Age_Group']=='Children']

test_check1['Title'].unique()
#Running a check to get the titles in the teenagers age group which should either be Mr or Miss.

test_check2 = test_data[test_data['Age_Group']=='Teenagers']

test_check2['Title'].unique()
#Checking to see the rows that have title as Mrs. This could be a data collection or data entry error.

test_check2[test_check2['Title']=='Mrs']
#Replacing the Title for Teenagers that have 'Mrs' with 'Miss'

test_data['Title'] = np.where((test_data.Age_Group.values == 'Teenagers') & 

                               (test_data.Title.values == 'Mrs'),'Miss', test_data.Title.values)
#Running a another check to ensure that the title for the teenager age group is either Mr or Miss

test_check3 = test_data[test_data['Age_Group']=='Teenagers']

test_check3['Title'].unique()
test_data.rename({'Sex':'Gender'}, axis = 1, inplace = True)
#Dropping columns that won't be used in building the ML model and predicting

test_data.drop(['Cabin','Ticket','Name','SibSp','Parch'], axis=1, inplace=True)
fig = plt.figure(figsize=(8,6))

sns.countplot(data= train_data, hue = 'Gender', x = 'Survived')

plt.xlabel('Dead or Survived', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Survival Rate by Gender', fontsize = 18)

plt.show()

#Men had a significant death count than females, so definitely women had a higher survival count than men
fig = plt.figure(figsize=(8,6))

sns.countplot(data= train_data, hue = 'Pclass', x = 'Survived')

plt.xlabel('Dead or Survived', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Survival Rate by Ticket Class', fontsize = 18)

plt.show()

#People who had a Pclass 1, that is, a first class ticket had a higher survival count and lower death count than those who had 

# a Pclass 2 or 3, second or third class ticket. Pclass 3 had a significant death toll compared to other Pclass groups.
fig = plt.figure(figsize=(8,6))

sns.countplot(data= train_data, hue= 'Traveling_Alone', x = 'Survived')

plt.xlabel('Dead or Survived', fontsize = 14)

plt.ylabel('Count', fontsize = 14)

plt.title('Traveling Alone - Survival Rate', fontsize = 18)

plt.show()

#People traveling alone(1) had a higher death count than those traveling with other people. People who traveled with others(0)

#had a slightly higher chance or surviving that those who traveled alone
age_group_survival = train_data[['Age_Group','Survived']].groupby(['Age_Group','Survived']).agg({'Survived':'count'})

age_group_survival.columns = ['Survival_Rate']

age_group_survival.reset_index(inplace=True)

age_group_survival
fig = px.bar(age_group_survival, x= 'Survived', y='Survival_Rate', 

             color='Age_Group',

             title='Survival Rate by Age Group',

             labels = {'Survived':'Dead or Survived', 'Survival_Rate':'Count'},

             hover_name='Age_Group',

             hover_data=['Survival_Rate','Survived'],

             barmode='group',

             template='plotly_dark',

             width = 800,

             height = 400)



#aligning the title position to center

fig.update(layout = dict(title = dict(x = 0.5)))



fig.show()

# HOVER OVER THE BARS TO GET MORE INFORMATION
train_data['Age_Group'].value_counts()
poe_survival = train_data[train_data['Survived']==1]

poe_survival = train_data.groupby(['Embarked','Pclass']).agg({'Survived':'count'})

poe_survival.columns = ['Count_of_Survivors']

poe_survival.reset_index(inplace=True)

poe_survival
fig = px.bar(poe_survival, x= 'Pclass', y='Count_of_Survivors', 

             color='Embarked',

             title='Survival Rate by Point of Embarkation',

             labels = {'Pclass':'Ticket Class', 'Count_of_Survivors':'Count of Survivors'},

             hover_name='Embarked',

             hover_data=['Pclass','Count_of_Survivors'],

             barmode='group',

             template='plotly_dark',

             width = 800,

             height = 400)



#aligning the title position to center

fig.update(layout = dict(title = dict(x = 0.5)))



fig.show()

# HOVER OVER THE BARS TO GET MORE INFORMATION
poe_deaths = train_data[train_data['Survived']==0]

poe_deaths = poe_deaths.groupby(['Embarked','Pclass']).agg({'Survived':'count'})

poe_deaths.columns = ['Death_Count']

poe_deaths.reset_index(inplace=True)

poe_deaths
fig = px.bar(poe_deaths, x= 'Pclass', y='Death_Count', 

             color='Embarked',

             title='Death Rate by Point of Embarkation',

             labels = {'Pclass':'Ticket Class', 'Death_Count':'Death Count'},

             hover_name='Embarked',

             hover_data=['Pclass','Death_Count'],

             barmode='group',

             template='plotly_dark',

             width = 800,

             height = 400)



#aligning the title position to center

fig.update(layout = dict(title = dict(x = 0.5)))



fig.show()

# HOVER OVER THE BARS TO GET MORE INFORMATION
train_data['Embarked'].value_counts()
train_data.head()
X = train_data.drop(['Survived','PassengerId'], axis=1)

Y = train_data['Survived']
#Machine Learning models take just numbers so any string values we have in our data will have to be converted to numbers.



#Using Column Transformer and One Hot Encoder rather than Label Encoder and One Hot Encoder as both give the same results.

#Using this method is however more effcient since i use just two lines of code.



#One Hot Encoder sorts the values for each column in ascending order and encodes each category based on this order. Eg male and 

#female, female will have a value of 1, 0 and male 0, 1. The output from One Hot Encoding puts the encoded columns first and 

#then the other columns that were not encoded.



#Since i'm dropping passenger id and survived columns for my X, the number of columns will reduce.

#I will be encoding the gender, embarked, title, age group columns which is represented by [1,4,5,8]



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5, 8])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
print(X[:1])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#Scaling the data so that all values are on the same scale

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
log_classifier = LogisticRegression(random_state = 0)

log_classifier.fit(X_train, y_train)
log_pred = log_classifier.predict(X_test)
log_cm = confusion_matrix(y_test, log_pred)

print(log_cm)

accuracy_score(y_test, log_pred)
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_pred)

print(knn_cm)

accuracy_score(y_test, knn_pred)
svm_classifier = SVC(kernel = 'linear', random_state = 0)

svm_classifier.fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_pred)

print(svm_cm)

accuracy_score(y_test, svm_pred)
ksvm_classifier = SVC(kernel = 'rbf', random_state = 0)

ksvm_classifier.fit(X_train, y_train)
ksvm_pred = ksvm_classifier.predict(X_test)
ksvm_cm = confusion_matrix(y_test, ksvm_pred)

print(ksvm_cm)

accuracy_score(y_test, ksvm_pred)
nb_classifier = GaussianNB()

nb_classifier.fit(X_train, y_train)
nb_pred = nb_classifier.predict(X_test)
nb_cm = confusion_matrix(y_test, nb_pred)

print(nb_cm)

accuracy_score(y_test, nb_pred)
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_cm = confusion_matrix(y_test, dt_pred)

print(dt_cm)

accuracy_score(y_test, dt_pred)
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)

print(rf_cm)

accuracy_score(y_test, rf_pred)
xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train)
xgb_pred = xgb_classifier.predict(X_test)
xgb_cm = confusion_matrix(y_test, xgb_pred)

print(xgb_cm)

accuracy_score(y_test, xgb_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,xgb_pred))
test_data_X = test_data.copy()

test_data_X = test_data_X.drop(['PassengerId'], axis = 1)

test_data_X.head()
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5, 8])], remainder='passthrough')

test_data_X = np.array(ct.fit_transform(test_data_X))
print(test_data_X[:1])
test_data_X = sc.transform(test_data_X)
xgb_test_pred = xgb_classifier.predict(test_data_X)
predicted_values = pd.DataFrame(xgb_test_pred , columns=['Survived'])
titanic_prediction = test_data.merge(predicted_values, left_index = True, right_index = True)
titanic_submission = pd.DataFrame(titanic_prediction[['PassengerId','Survived']])

titanic_submission.head()
titanic_submission.to_csv('my_titanic_submission.csv', index = False)

print('Your submission was successfully saved!')