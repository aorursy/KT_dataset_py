%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

train_data.head(10)
# Getting broad overview of the data



train_data.info()

print('\n\n\n')

test_data.info()
# Total number null values in each column



train_data.isnull().sum()
test_data.isnull().sum()
# percentage of null values in cabin of training data



print((train_data['Cabin'].isnull().sum()/len(train_data['Cabin']))*100)
# Total number of null values in age



pd.isnull(train_data['Age']).sum()
# Distribution of age 



plt.hist(train_data['Age'],bins = 20, range = (0,90))

plt.title('Age distribution with missing values')

plt.show()
# Filling all the missing values in age column with median age



train_data['Age'].fillna(train_data['Age'].median(), inplace = True)

test_data['Age'].fillna(test_data['Age'].median(), inplace = True)



# Conferming if any null in age



train_data['Age'].isnull().sum()
# Age distribution after adjusting the column with median age 



plt.hist(train_data['Age'], bins = 20, range = (0,90))

plt.title('Age distribution without missing values')

plt.show()
# How many passenger have survived



# total_passenge is number of passenger in training data set



total_passenger = train_data['Survived'].count()



# survived_passenger is the count of passenger who have survived in training data set

# since the code for survival is 1 ,its sum will give the required number



survived_passenger = train_data['Survived'].sum()



# percentage of passengers survived



percentage_survived = (survived_passenger/total_passenger)

print('percentage survival : ', str(percentage_survived*100))
# Pclass vs Survived 



groupby_Pclass = train_data[['Survived',  'Pclass']].groupby(['Pclass'], as_index = False)

print(groupby_Pclass.mean())
# Creating a bar chart to see the avg survival across Pclass



y1 = groupby_Pclass['Survived'].get_group(1).mean()

y2 = groupby_Pclass['Survived'].get_group(2).mean()

y3 = groupby_Pclass['Survived'].get_group(3).mean()



x = [1,2,3]

y = [y1, y2, y3]



plt.bar(x,y)

plt.xticks([1.4, 2.4, 3.4], ['Class1', 'Class2', 'Class3'])

plt.ylabel('Mean Survival')

plt.xlabel('Pclass')



plt.plot((0,4), (percentage_survived,percentage_survived), color = 'red', label = 'avg survival')



plt.legend()



#plt.title('Surviveal mean agains Pclass')

plt.show()
# sex vs Survived



groupby_sex = train_data[['Survived', 'Sex']].groupby(['Sex'], as_index = False)

print(groupby_sex.mean())
#Plotting bar chart for avg survival as per sex



y1 = groupby_sex['Survived'].get_group('female').mean()

y2 = groupby_sex['Survived'].get_group('male').mean()



x = [1,2]

y = [y1, y2]



plt.bar(x,y)

plt.xticks([1.4, 2.4,], ['Female', 'Male'])

plt.ylabel('Mean Survival')

plt.xlabel('Sex')



plt.plot((0,4), (percentage_survived,percentage_survived), color = 'red', label = 'avg survival')



plt.legend()



#plt.title('Surviveal mean agains sex')

plt.show()
# Age vs Survived 



groupby_age = train_data[['Survived', 'Age']].groupby(['Survived'], as_index = False)
# Stacked histogram 



x1 = groupby_age['Age'].get_group(0)

x2 = groupby_age['Age'].get_group(1)

plt.hist([x1,x2], stacked = True, bins = 15, range = (0,90), label = ['Not Survived', 'Survived'], color = ['red', 'green'])

plt.legend()



plt.show()
#bins = [0, 16, 35, 60, 90]

#group_names = ['Kid', 'Young' ,'Adult', 'Old']

age_bins = [0, 16, 35, 60, 90]

age_group = [0, 1, 2, 3]

train_data['Age_group'] = pd.cut(train_data['Age'], age_bins, labels=age_group)

train_data.head()
grouped_age = train_data[['Survived', 'Age_group']].groupby(['Age_group'])

grouped_age['Survived'].mean()
# Family size vs Survived



# we are going to crate a column named family size in data



train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1



train_data.head()
groupby_family = train_data[['Survived', 'Family_Size']].groupby(['Family_Size'], as_index = False)

groupby_family['Survived'].mean()
#bins = [0, 1, 4, 11]

#group_names = ['Alone','Small','Big']



family_bins = [0, 1, 4, 11]

family_group = [0, 1, 2]

train_data['Family_group'] = pd.cut(train_data['Family_Size'], family_bins, labels=family_group)

train_data['Family_group'].astype(int)

train_data.head()
grouped_family = train_data[['Survived','Family_group']].groupby(['Family_group'])

grouped_family['Survived'].mean()
# Fair vs Survived



groupby_fare = train_data[['Survived', 'Fare']].groupby(['Survived'])

train_data['Fare'].describe()
y1 = groupby_fare['Fare'].get_group(0)

y2 = groupby_fare['Fare'].get_group(1)



# Plot of fare of people who didn't survive

plt.plot(y1, color = 'red', label = 'Not Survive')



#plot of Fare of people who survived

plt.plot(y2, color = 'green', label = 'Survived')

plt.ylabel('Fare')

plt.legend()

plt.show()
# Lets dig dipper



train_data['Quartile_fare'] = pd.qcut(train_data['Fare'], 4)

train_data.head()
groupby_quartile = train_data[['Survived', 'Quartile_fare']].groupby(['Quartile_fare'], as_index = False)

groupby_quartile.mean()
train_data.head(5)
train_data.isnull().sum()
drop_columns = ['PassengerId', 'Name', 'Age', 'Parch', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Family_Size', 'Quartile_fare']

train_data.drop(drop_columns, axis = 1,inplace = True)
train_data.head()
# Mapping Sex

train_data['Sex'] = train_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Mapping Fare

train_data.loc[ train_data['Fare'] <= 7.91, 'Fare'] = 0

train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare'] = 1

train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare']   = 2

train_data.loc[ train_data['Fare'] > 31, 'Fare']  = 3

train_data['Fare'] = train_data['Fare'].astype(int)



train_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
# Filling missing fare



test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)

test_data['Fare'].isnull().sum()
# With our experience we already know which columns need to be added as which one are going to be removed



# Creating Age_group column

test_data['Age_group'] = pd.cut(test_data['Age'], age_bins, labels=age_group)

test_data['Age_group'].astype(int)



# Creating Family_Size column

test_data['Family_Size'] = test_data['Parch'] + test_data['SibSp'] + 1



# Creating Family_group column

test_data['Family_group'] = pd.cut(test_data['Family_Size'], family_bins, labels=family_group)

test_data['Family_group'].astype(int)



# Mapping Sex

test_data['Sex'] = test_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



# Mapping Fare

test_data.loc[ test_data['Fare'] <= 7.91, 'Fare'] = 0

test_data.loc[(test_data['Fare'] > 7.91) & (test_data['Fare'] <= 14.454), 'Fare'] = 1

test_data.loc[(test_data['Fare'] > 14.454) & (test_data['Fare'] <= 31), 'Fare']   = 2

test_data.loc[ test_data['Fare'] > 31, 'Fare']  = 3

test_data['Fare'] = test_data['Fare'].astype(int)





# Dropping columns

drop_column = ['Name', 'Age', 'Parch', 'SibSp', 'Ticket', 'Cabin', 'Embarked', 'Family_Size']

test_data.drop(drop_column, axis = 1,inplace = True)
test_data.head()
test_data.isnull().sum()
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing
# define training and testing sets



X_train = train_data.drop("Survived",axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId",axis=1)
X_train.shape
Y_train.shape
X_test.shape
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)