import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# To ignore all the warnings

import warnings

warnings.filterwarnings('ignore')
# Load training data

train_data = pd.read_csv("../input/titanic/train.csv")



# Load test data

test_data = pd.read_csv("../input/titanic/test.csv")
# First look at the train data

train_data.sample(5)
train_data.set_index('PassengerId', inplace=True)

test_data.set_index('PassengerId', inplace=True)
# Check the information of datasets

print('*' * 20,'Training data', '*' * 20)

print(train_data.info())

print('*' * 20,'Test data', '*' * 20)

print(test_data.info())
# Descriptive measures of data

train_data.describe(include='all')
print(train_data.Cabin.describe())





print('*' * 20, 'Unique values in cabin', '*' * 20)

print(train_data['Cabin'].unique())
train_data.Cabin.isnull().sum()/len(train_data) * 100
# Checking the skewness

train_data.skew()
# Pclass

print('*' * 20, "Passenger's count by Socio - economic status", '*' * 20)

print(train_data['Pclass'].value_counts())



# Sex

print('*' * 20, "Passenger's count by Gender", '*' * 20)

print(train_data['Sex'].value_counts())



# Embarked

print('*' * 20, "Passenger's count by Port of Embarkation", '*' * 20)

print(train_data['Embarked'].value_counts())
# pclass and sex

print('*' * 20, "Frequency count of passengers by Passenger's class and gender", '*' * 20)

table1=pd.crosstab(train_data['Pclass'],train_data['Sex'],margins=True,margins_name='Sum')

print(table1)



# Pclass and Embarked

print('*' * 20, "Frequency count of passengers by Passenger's class and their port of Embarkation", '*' * 20)

table2=pd.crosstab(train_data['Pclass'],train_data['Embarked'],margins=True,margins_name='Sum')

print(table2)



# Gender and Embarked

print('*' * 20, "Frequency count of passengers by Passenger's port of Embarkation and gender", '*' * 20)

table3=pd.crosstab(train_data['Sex'],train_data['Embarked'],margins=True,margins_name='Sum')

print(table3)
# Age

plt.figure(figsize=(10,6))

plot = plt.hist(train_data.Age, bins = 8, histtype='bar')   # plot a histogram

plt.ylabel('Count of values in each bin')

plt.xlabel('Ranges')

for i in range(8):

    plt.text(plot[1][i],plot[0][i],str(plot[0][i]))   # display the count of values falling in each range
# check the skewness of Age

train_data.Age.skew()
# Fare

plt.figure(figsize=(10,6))

plot = plt.hist(train_data.Fare, bins = 10, histtype='bar')

plt.ylabel('Count of values in each bin')

plt.xlabel('Ranges')

for i in range(10):

    plt.text(plot[1][i],plot[0][i],str(plot[0][i]))   # display the count of values falling in each range
# check skewness of Fare varible

train_data.Fare.skew()
# Boxplot: to observe outliers

plt.figure(figsize=(10,6))

plt.boxplot(train_data.Fare)

plt.ylabel('Fare')
# SibSp

sns.countplot(train_data.SibSp)   # counts the frequency of each values
# Parch

sns.countplot(train_data.Parch) # counts the frequency of each values
# Survived

sns.countplot(train_data.Survived)
# Sex

sns.countplot(train_data.Sex)
# Pclass

sns.countplot(train_data.Pclass)
# Boxplot: shows the distribution of quantitative data in a way 

# that facilitates comparisons between variables or across levels of a categorical variable



plt.figure(figsize=(10,6))

box_plot = sns.boxplot(x = train_data.Survived, y = train_data.Age, data = train_data)





# Adding text in the boxplot like median value, first quartile value and third quartile value

medians = train_data.groupby(['Survived'])['Age'].median()

first_quartile = train_data.groupby(['Survived'])['Age'].quantile(0.25)

third_quartile = train_data.groupby(['Survived'])['Age'].quantile(0.75)



# Vertical distance from lines to display the particular value

vertical_offset_median = train_data['Age'].median() * 0.05 

vertical_offset_fquartile = train_data['Age'].quantile(0.25) * 0.05 

vertical_offset_tquartile = train_data['Age'].quantile(0.75) * 0.05

for xtick in box_plot.get_xticks():

    

    # Display text at median (Second quartile)

    box_plot.text(xtick,medians[xtick] + vertical_offset_median,medians[xtick], 

            horizontalalignment='center',size='medium',color='w',weight='semibold')

    

    # Display text at first quartile

    box_plot.text(xtick,first_quartile[xtick] + vertical_offset_fquartile,first_quartile[xtick], 

            horizontalalignment='center',size='medium',color='w',weight='semibold')

    

    # Display text at third quartile

    box_plot.text(xtick,third_quartile[xtick] + vertical_offset_tquartile,third_quartile[xtick], 

            horizontalalignment='left',size='medium',color='b',weight='semibold')
# Boxplot: shows the distribution of quantitative data in a way 

# that facilitates comparisons between variables or across levels of a categorical variable



plt.figure(figsize=(10,6))

box_plot = sns.boxplot(x = train_data.Survived, y = train_data.Fare, data = train_data)





# Adding text in the boxplot like median value, first quartile value and third quartile value

medians = train_data.groupby(['Survived'])['Fare'].median()

first_quartile = train_data.groupby(['Survived'])['Fare'].quantile(0.25)

third_quartile = train_data.groupby(['Survived'])['Fare'].quantile(0.75)



# Vertical distance from lines to display the particular value

vertical_offset_median = train_data['Fare'].median() * 0.05 

vertical_offset_fquartile = train_data['Fare'].quantile(0.25) * 0.05 

vertical_offset_tquartile = train_data['Fare'].quantile(0.75) * 0.05

for xtick in box_plot.get_xticks():

    

    # Display text at median (Second quartile)

    box_plot.text(xtick,medians[xtick] + vertical_offset_median,medians[xtick], 

            horizontalalignment='center',size='medium',color='w',weight='semibold')

    

    # Display text at third quartile

    box_plot.text(xtick,third_quartile[xtick] + vertical_offset_tquartile,third_quartile[xtick], 

            horizontalalignment='left',size='medium',color='b',weight='semibold')
# Boxplot: shows the distribution of quantitative data in a way 

# that facilitates comparisons between variables or across levels of a categorical variable



plt.figure(figsize=(10,6))

box_plot = sns.boxplot(x = train_data.Sex, y = train_data.Age, data = train_data)





# Adding text in the boxplot like median value, first quartile value and third quartile value

medians = train_data.groupby(['Sex'])['Age'].median()

first_quartile = train_data.groupby(['Sex'])['Age'].quantile(0.25)

third_quartile = train_data.groupby(['Sex'])['Age'].quantile(0.75)



# Vertical distance from lines to display the particular value

vertical_offset_median = train_data['Age'].median() * 0.05 

vertical_offset_fquartile = train_data['Age'].quantile(0.25) * 0.05 

vertical_offset_tquartile = train_data['Age'].quantile(0.75) * 0.1

for xtick in box_plot.get_xticks():

    

    # Display text at median (Second quartile)

    box_plot.text(xtick,medians[xtick] + vertical_offset_median,medians[xtick], 

            horizontalalignment='center',size='medium',color='w',weight='semibold')

    

    # Display text at first quartile

    box_plot.text(xtick,first_quartile[xtick] + vertical_offset_fquartile,first_quartile[xtick], 

            horizontalalignment='center',size='medium',color='b',weight='semibold')

    

    # Display text at third quartile

    box_plot.text(xtick,third_quartile[xtick] + vertical_offset_tquartile,third_quartile[xtick], 

            horizontalalignment='left',size='medium',color='b',weight='semibold')
plt.figure(figsize=(10,6))

count_plot = sns.countplot(x = train_data.Sex, hue = train_data.Survived)
plt.figure(figsize=(10,6))

count_plot = sns.countplot(x = train_data.Pclass, hue = train_data.Survived)
plt.figure(figsize=(10,6))

count_plot = sns.countplot(x = train_data.Embarked, hue = train_data.Survived)
plt.figure(figsize=(10,6))

count_plot = sns.countplot(x = train_data.SibSp, hue = train_data.Survived)
plt.figure(figsize=(10,6))

count_plot = sns.countplot(x = train_data.Parch, hue = train_data.Survived)

plt.legend(bbox_to_anchor=(1, 1))
plt.figure(figsize=(12,8))

sns.heatmap(train_data.corr(), annot=True) # Returns correlation among features which have numerical observations
# Function to calculate missing values

def calc_missing_values(df):

    

    """

    This function will take dataframe as an input and return the missing value information for each features as a dataframe.

    """

    missing_count = df.isnull().sum().sort_values(ascending = False)

    missing_percent = round(missing_count / len(df) * 100, 2)

    missing_info = pd.concat([missing_count, missing_percent], axis = 1, keys=['Missing Value Count','Percent of missing values'])

    return missing_info





print('*' * 20, 'Missing values information of Training data', '*' * 20)

print(calc_missing_values(train_data))



print()

print('*' * 20, 'Missing values information of Test data', '*' * 20)

print(calc_missing_values(test_data))
print('Mean age of passengers: ', train_data.Age.mean())

print('Median age of passengers: ', train_data.Age.median())
train_data.Age.fillna(29.6, inplace=True)

test_data.Age.fillna(29.6, inplace=True)
train_data.Embarked.fillna(train_data.Embarked.mode().values[0], inplace=True)
test_data.Fare.fillna(test_data.Fare.median(), inplace=True)
print(calc_missing_values(train_data))

print(calc_missing_values(test_data))
# Fare column have some outliers as observed during visualization 

plt.figure(figsize=(10,6))

box_plot = sns.boxplot(x = train_data.Survived, y = train_data.Fare, data = train_data)
train_data[train_data.Fare > 300]
columns_to_drop = ['Cabin','Name','Ticket']

train_data.drop(columns=columns_to_drop, axis=1, inplace=True)

test_data.drop(columns=columns_to_drop, axis=1, inplace=True)
# Training Data

train_data = pd.get_dummies(train_data, columns=['Sex','Embarked'], drop_first=True)



# Test Data

test_data = pd.get_dummies(test_data, columns=['Sex','Embarked'], drop_first=True)
def age_bucketizer(r):

    if r <= 12:

        return 0

    elif r <= 18:

        return 1

    elif r <= 59:

        return 2

    else:

        return 3

    

# Apply the above function on age column for both train data and test data

train_data['age_class'] = train_data.Age.apply(age_bucketizer)

test_data['age_class'] = test_data.Age.apply(age_bucketizer)
first_quartile_fare = train_data.Fare.quantile(0.25)

second_quartile_fare = train_data.Fare.quantile(0.5)

third_quartile_fare = train_data.Fare.quantile(0.75)

def fare_bucketizer(r):

    if r <= first_quartile_fare:

        return 0

    elif r <= second_quartile_fare:

        return 1

    elif r <= third_quartile_fare:

        return 2

    else:

        return 3

    

# Apply above function to fare column of train data and test data

train_data['fare_class'] = train_data.Fare.apply(fare_bucketizer)

test_data['fare_class'] = test_data.Fare.apply(fare_bucketizer)
# drop Age and Fare columns 

train_data.drop(columns=['Age', 'Fare'], axis = 1, inplace = True)

test_data.drop(columns=['Age', 'Fare'], axis = 1, inplace = True)
feature_data = train_data.drop(columns=['Survived'])

target = train_data.Survived

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(feature_data, target, test_size = 0.3, random_state=42)
models = []  # To store all the models

accuracy = []  # To store the accuracy of respective models
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()



# fit the model

log_model.fit(X_train, Y_train)



# predict the data

log_pred = log_model.predict(X_test)



# store the model in models

models.append('Logistic Regression')
from sklearn.metrics import confusion_matrix, accuracy_score



print(confusion_matrix(Y_test, log_pred))

print(round(accuracy_score(Y_test, log_pred),2))



# store the accuracy score

accuracy.append(round(accuracy_score(Y_test, log_pred),2))
from sklearn.neighbors import KNeighborsClassifier



# checking different values of k

for k in range(1, 15):

    knn = KNeighborsClassifier(k)

    knn.fit(X_train, Y_train)

    print(k)

    print(knn.score(X_train, Y_train))

    print(knn.score(X_test, Y_test))
# Fitting model for k = 7

knn = KNeighborsClassifier(7)



# store the model

models.append('KNN')



# fit the model

knn.fit(X_train, Y_train)



# predict for X_test

knn_pred = knn.predict(X_test)



print(confusion_matrix(Y_test, knn_pred))

print(round(accuracy_score(Y_test, knn_pred),2))



# store the accuracy score

accuracy.append(round(accuracy_score(Y_test, knn_pred),2))
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()



# store the model in models

models.append('Decision Tree')



# fit the data

dtree.fit(X_train, Y_train)



# predict the data

dtree_pred = dtree.predict(X_test)





print(confusion_matrix(Y_test, dtree_pred))

print(round(accuracy_score(Y_test, dtree_pred),2))



# store the accuracy score

accuracy.append(round(accuracy_score(Y_test, dtree_pred),2))
model_compare = pd.DataFrame({'models': models, 'accuracy': accuracy})

model_compare
test_data.index
# prediction for test data

test_prediction = knn.predict(test_data)





# creating dataframe for the predicted value with their passenger id

submission = pd.DataFrame({

        "PassengerId": test_data.index,

        "Survived": test_prediction

    })



# type conversion

submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



# Creating a csv file with test prediction

submission.to_csv("titanic_submission.csv", index=False)