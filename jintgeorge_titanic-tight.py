import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Data Plotting and Visualization
import matplotlib.pyplot as plt
# to plot the figures on the page
%matplotlib inline 
import seaborn as sns
sns.set_style("whitegrid")

#Machine Learning and Model Fitting
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

#General rules to follow 
# 1) Do not touch the test data set (Keep it aside for testing your model accuracy)
# 2) If you are only provided the training data, set aside some (around 20%) 
#    of training set as test data 
# 3) Use Cross validation to avoid overfitting (Randomly partition the data for training and testing)

#Load training dataet as pandas dataframe
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#test to check data loaded
train_df.head()
# Give me a 10,000 feet overview on the train data
#DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)

train_df.info()

# Quick Peek Below
# Total Rows/Samples/Datapoints = 891 (Passenger ID counts which is also index)
# There are definitely some Null fields in the data (Age, Cabin, Embarked ) which could be missed due to various reasons
# Comparatively this dataset is pretty clean and formatted  
#Generate various summary statistics, excluding NaN values.
train_df.describe()
#Some of features are not contributing to predict the model 
#eg. PassengerID, Name, Ticket
# Drop these rows from dataframe

#axis = 1 means vertical direction => drop the following columns from all rows
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

##Handle Embarked
#Only 2 missing values for Embarked. Fill those with mode for the same value(most occured)
#which is 'S' (Southampton). Majority of passengers embarked from Southampton (in our trainign dataset)
train_df["Embarked"] = train_df["Embarked"].fillna("S")

#plot
sns.factorplot('Embarked', 'Survived', data = train_df, size = 4, aspect = 3)
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15,5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=train_df, ax=axis2, order=[1,0])
Emb_vs_Surv_df = train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived',data = Emb_vs_Surv_df, order=['S', 'C', 'Q'], ax=axis3)
#Drop the embarked coulum as it does not look useful in predition
train_df.drop('Embarked', axis=1, inplace=True)
test_df.drop('Embarked', axis=1, inplace=True)

fare_not_survived_df = train_df["Fare"][train_df["Survived"] == 0]
fare_survived_df = train_df["Fare"][train_df["Survived"] == 1]
fig, (axis1, axis2) = plt.subplots(2, 1, figsize= (10,9))
sns.distplot(train_df['Fare'], ax=axis1) #seaborn distribution plot which determines appropriate bin size implicitly
train_df["Fare"].hist(ax=axis2, bins=40)
print('------------------Fare Status---------------------')
print('Survived:')
print(fare_survived_df.describe())
print('Not Survived')
print(fare_not_survived_df.describe())
train_df_mean_Age = train_df["Age"].mean()
train_df_std_Age = train_df["Age"].std()
count_train_df_NAN_Age = train_df["Age"].isnull().sum()

test_df_mean_Age = test_df["Age"].mean()
test_df_std_Age = test_df["Age"].std()
count_test_df_NAN_Age = test_df["Age"].isnull().sum()

rand_1 = np.random.randint(train_df_mean_Age - train_df_std_Age, 
                           train_df_mean_Age + train_df_std_Age,
                          count_train_df_NAN_Age)
rand_2 = np.random.randint(test_df_mean_Age - test_df_std_Age,
                          test_df_mean_Age + test_df_std_Age,
                          count_test_df_NAN_Age)
train_df["Age"][train_df.Age.isnull()] = rand_1
test_df["Age"][test_df.Age.isnull()] = rand_2

# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()

fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

#Cabin
#Drop Cabin as it has lot of NAN values and not helpful in predicition
train_df.drop("Cabin", axis=1, inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)
#Family
train_df["Family"] = train_df["Parch"] + train_df["SibSp"]
train_df["Family"].loc[train_df["Family"] > 0] = 1
train_df["Family"].loc[train_df["Family"] == 0] = 0

test_df["Family"] = test_df["Parch"] + test_df["SibSp"]
test_df["Family"].loc[test_df["Family"] > 0] = 1
test_df["Family"].loc[test_df["Family"] == 0] = 0

train_df.drop(["Parch", "SibSp"], axis=1, inplace=True)
test_df.drop(["Parch", "SibSp"], axis=1, inplace=True)
print(train_df["Family"])

fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)

#average plot for Family/Non Family member passengers

family_perc = train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(x = 'Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)
#Sex
#Classify passenger as child or adults
def getPerson(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex

train_df["Person"] = train_df[['Age', 'Sex']].apply(getPerson, axis=1)
test_df["Person"] = test_df[['Age', 'Sex']].apply(getPerson, axis=1)

#drop sex column as we have Person column now
train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)





    
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train_df['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

#person_dummies_test  = pd.get_dummies(test_df['Person'])
#person_dummies_test.columns = ['Child','Female','Male']
#person_dummies_test.drop(['Male'], axis=1, inplace=True)
print(person_dummies_titanic)
