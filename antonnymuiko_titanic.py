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
# import the python libraries needed 

# data preparation, analysis & cleaning 

import numpy as np

import pandas as pd

# visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

# machine learning libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import SVC, LinearSVC

# metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# gender submission 

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

# train data set

test = pd.read_csv("../input/titanic/test.csv")

# test data set

train = pd.read_csv("../input/titanic/train.csv")
# import pandas_profiling to geneate profile report of the train dataset

import pandas_profiling

# generating the report

train.profile_report()
# the shape and size of the dataset

print("Train data shape:", train.shape)

print("Test data shape:", test.shape)
# brief description of the data sets

train.info()

print("\n")

test.info()
# dealing with missing values

train['Age'].fillna(train['Age'].mean(), inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

# column embarked has missing values that will replace with mode after feature engineering
# Checking for missing values 

print(train.isnull().sum())

print('\n')

test.isnull().sum()
# sns.boxplot(train['Age'], color = 'red')

plt.figure(figsize = (15,11))

sns.boxplot(train['Embarked'],train['Age'] , hue = train['Sex'], palette = 'Accent')

plt.ylabel('Age Distribution', fontsize = 20, color = 'blue')

plt.xticks(fontsize = 15)

plt.xlabel('Embarked',fontsize = 20, color = 'blue')

plt.title('Boxplot showing the outliers per Port of Embarkation', fontsize = 22, color = 'blue')

plt.show()
# dropping the unnecessary colums 

train.drop(['Name', 'Ticket','Fare','Cabin'], axis = 1, inplace = True)

test.drop(['Name', 'Ticket','Fare','Cabin'],axis = 1, inplace = True)
# the distribution of age 

age = train['Age']

# plotting a histogram 

plt.figure(figsize=(13,7), dpi= 80)

# plt.hist(age, bins = 10, histtype = 'bar', rwidth = 0.9, color = 'black')

sns.distplot(train['Age'], color = "darkgreen", label = "Age" )

# labeling the axis 

plt.ylabel('Frequency', fontsize = 20)

plt.xlabel('Passengers Age', fontsize = 20)

# title

plt.title('Passenger Age Distribution', fontsize = 22)

plt.show()
# distribution of Age of the different genders in the ship

plt.figure(figsize=(13,7), dpi= 80)

sns.kdeplot(train.loc[train['Sex'] == 'male', "Age"], shade=True, color="red", label="Male", alpha=.7)

sns.kdeplot(train.loc[train['Sex'] == 'female', "Age"], shade=True, color="green", label="Female", alpha=.7)

plt.title('Distribution of Age as per respective gender', fontsize = 22, color = 'blue')

plt.ylabel('Frequency', fontsize = 20, color = 'blue')

plt.xlabel('Passenger Age', fontsize = 20, color = 'blue')

plt.show()
#Creating Visualisation of passengers who survived or died 

def plot(passenger_info):

    survived = train[train.Survived == 1][passenger_info].value_counts()

    not_survived = train[train.Survived == 0][passenger_info].value_counts()

    df = pd.DataFrame([survived, not_survived])

    df.index = ['Survived', 'Not_survived']

    sns.set()

    df.plot(kind = 'bar', stacked = 'True', figsize = (14, 8))

    plt.xticks(fontsize = 18)
# passengers who survived based on gender 

plot('Sex')
# passengers who survived based on place they embarked 

plot('Embarked')
# based on the class

plot('Pclass')
plot('SibSp')
plot('Parch')
survive = train.groupby(['Survived'])['Survived'].count()

survive

plt.figure(figsize = (16, 12))

explode = (0,0.05)

labels = (['Not_Survived' , 'Survived'])

colors = ['Red','Green']

plt.pie(survive.values, labels = labels, explode = explode, autopct = '%1.2f%%', colors = colors, startangle = 135, textprops = {'color':'black', 'style': 'oblique', 'size':18})

plt.title('Pie Chart Displaying passngers fate', fontsize = 20)

plt.show()
train.head()
test.head()
train.isnull().sum()

test.isnull().sum()

# replacing null value in train embarked column

train['Embarked'].value_counts()

train['Embarked'].fillna('S', inplace = True) # replaced using the mode of the stations 
# using label encoder for feature engineering

# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder 

label_encoder = LabelEncoder()
# encode labels in column sex and embarked

cols = ['Sex','Embarked']

# using a loop 

for col in cols:

    train[col] = label_encoder.fit_transform(train[col])

    test[col] = label_encoder.fit_transform(test[col])

# data used to train the model

train.head()
# data used for testing

test.head()
# splitting the train data into attributes and label

test_features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']

X = train[test_features].values



y = train['Survived'].values
# splitting into train and test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# fitting the logistic regression

regressor = LogisticRegression(random_state = 42)

regressor.fit(X_train, y_train)
# retrieve the intercept 

print(regressor.intercept_)

# retrieve the coeffecient 

print(regressor.coef_)
# use the model to make prediction 

y_pred = regressor.predict(X_test)
# create a data frame for comparison

regressor_df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})

regressor_df.head()
# model metrics

print('Model performance based on confusion matrix report:')

confusion_matrix(y_test, y_pred, labels = [0,1])
print('Accuracy score of the model is',np.round((accuracy_score(y_test, y_pred)*100),4), '%')
# fitting the decisiontree classifier 

desClf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3,

                                min_samples_split = 5, random_state = 42)

desClf.fit(X_train, y_train)
# use the model to make prediction

desClf_pred = desClf.predict(X_test)
# Create a data frame for comparison

desClf_df = pd.DataFrame({'Actual': y_test, 'Predicted': desClf_pred})

desClf_df.head()
# model evaluation 

print('Model evaluation based on the confusion matrix')

confusion_matrix(y_test, desClf_pred)
print('Accuracy score of the model is',np.round((accuracy_score(y_test, desClf_pred)*100),4), '%')
# creating a random forest classifier function with 100 number of decision trees

ranf = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = 6, min_samples_split = 4)

# training on train data 

ranf.fit(X_train, y_train)
# use the model to make prediction

ranf_pred = ranf.predict(X_test)
# making comparison

ranf_df = pd.DataFrame({'Actual': y_test, 'Predicted': ranf_pred})

ranf_df.tail(10)
# model evaluation 

print('Model evaluation based on the confusion matrix')

confusion_matrix(y_test, ranf_pred)
print('Accuracy score of the model is',np.round((accuracy_score(y_test, ranf_pred)*100),4), '%')
# making predictions on the test data set 

pred = ranf.predict(test)
# creating a dataframe 

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

# display the last 10 rows

submission.head(10)
# Converting

antonny_titanic = submission.to_csv('antonny_titanic.csv', index = False)