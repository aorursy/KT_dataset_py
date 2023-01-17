# import packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score



# read in training and test data

train_data = pd.read_csv('.../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
fig, ax = plt.subplots(figsize=(10,5))

sns.heatmap(train_data.iloc[:,1:].corr(), annot = True, ax = ax)

plt.show()
# barchart of pclass vs. survival

fig, ax = plt.subplots(figsize = (10,5))



# retrieve relevant data

survived = train_data[train_data['Survived'] == 1]['Pclass'].value_counts()

perished = train_data[train_data['Survived'] == 0]['Pclass'].value_counts()

survived.sort_index(inplace = True)

perished.sort_index(inplace = True)



# plot data

bar_width = 0.35

survived_bar = plt.bar(survived.index.values, survived, bar_width, label = "Survived")

perished_bar = plt.bar(perished.index.values + bar_width, perished, bar_width, label = "Died")



# plot labels

plt.xlabel('Socio-Economic Class')

plt.ylabel('Number of passengers')

plt.title('Survivals for each socio-economic class')

plt.xticks(survived.index.values + 0.175, ('Upperclass', 'Middleclass', 'Lowerclass'))

plt.legend()



# show plot

plt.tight_layout()

plt.show()
# get copy of the main data frame

temp_df = train_data.copy()

temp_df = temp_df[['Survived', 'Sex']]



# get number of who survived and died for the males

men_survived = temp_df[temp_df['Sex'] == 'male']['Survived'].value_counts()[1]

men_perished = temp_df[temp_df['Sex'] == 'male']['Survived'].value_counts()[0]

total_males = men_survived + men_perished

men_data = [men_survived, men_perished]/total_males



# get number of who survived and died for the females

women_survived = temp_df[temp_df['Sex'] == 'female']['Survived'].value_counts()[1]

women_perished = temp_df[temp_df['Sex'] == 'female']['Survived'].value_counts()[0]

total_females = women_survived + women_perished

women_data = [women_survived, women_perished]/total_females



# plot two piecharts for both genders

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,5))

ax1.pie(men_data, labels = ['Survived', 'Died'], colors = ['green', 'red'], autopct = '%1.1f%%')

ax1.set_title('Survival for men on the titanic')

ax2.pie(women_data, labels = ['Survived', 'Died'], colors = ['green', 'red'], autopct = '%1.1f%%')

ax2.set_title('Survival for women on the titanic')



# display plot

plt.tight_layout()

plt.show()
# retrieve all the survivor's ages

temp_df = train_data.copy()

temp_df = temp_df[temp_df['Survived'] == 1]

agedist = temp_df.sort_values('Age')

agedist = agedist['Age']



# display histogram

fig = plt.figure(figsize = (10,5))

plt.hist(agedist[~np.isnan(agedist)], edgecolor='black', linewidth = 1.2, bins = 50)

plt.xlabel('Age')

plt.ylabel('Number Survived')

plt.title('Number of people who survived by age')

plt.show()
# retrieve the appropriate data

temp_df = train_data[['Survived', 'SibSp']].copy()

temp_df = temp_df[temp_df['Survived'] == 1]['SibSp']



# plot histogram of the distribution

fig = plt.figure(figsize = (10,5))

plt.hist(temp_df, bins = np.arange(5) - 0.25, edgecolor='black', linewidth = 1.2, width = 0.5, color = 'green')

plt.xlabel('Number of siblings/spouses for the individual')

plt.ylabel('Number survived')

plt.title('Distribution of the individuals who survived based on number of siblings/spouses')

plt.show()
# retrieve the appropriate data

temp_df = train_data[['Survived', 'Parch']].copy()

temp_df = temp_df[temp_df['Survived'] == 1]['Parch']



# plot histogram of the distribution

fig = plt.figure(figsize = (10,5))

plt.hist(temp_df, bins = np.arange(5) - 0.25, edgecolor='black', linewidth = 1.2, width = 0.5, color = 'green')

plt.xlabel('Number of parents/children for the individual')

plt.ylabel('Number survived')

plt.title('Distribution of the individuals who survived based on number of parents/children')

plt.show()
# retrieve data

temp_df = train_data[['Survived', 'Fare']].copy()

temp_df.sort_values(by = ['Fare'], inplace = True)



# plot scatterplot

fig = plt.figure(figsize = (10,5))

plt.scatter(temp_df['Fare'].values, temp_df['Survived'].values, color = 'orange')

plt.xlabel('Passenger Fare')

plt.ylabel('Survival chance')

plt.title('Survival rate based on the passenger fare to get onboard the Titanic')

plt.show()
# seperate the survived column from the training data as a response vector

# drop features that are not useful such as the Name and Ticket etc...

features = train_data.drop(["Survived", "Name", "Ticket", "Cabin", "Embarked"], axis = 1)

response = train_data["Survived"]



# data wrangling process

# change the male/female terms in the Sex column into 0 and 1 respectively

features.loc[features['Sex'] == 'male', 'Sex'] = 0

features.loc[features['Sex'] == 'female', 'Sex'] = 1



# remove NaN values for each column and replace with the mean of the respective columns values

features['Age'] = features['Age'].fillna(features['Age'].dropna().mean())

features['Fare'] = features['Fare'].fillna(features['Fare'].dropna().mean())



# conduct same cleansing procedure for the test_data

test_data = test_data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis = 1)

test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0

test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1

test_data['Age'] = test_data['Age'].fillna(test_data['Age'].dropna().mean())

test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].dropna().mean())



# build the model

logreg = linear_model.LogisticRegression(solver='lbfgs', max_iter = 5000)

logreg_fit = logreg.fit(features, response)

logreg_predict = logreg.predict(test_data)



# crate new dataframe to store the predicted survival chances along with the passenger id

submission_df = pd.DataFrame({'PassengerID': test_data['PassengerId'],

                             'Survived': logreg_predict})

submission_df.to_csv('submission.csv', index = False)