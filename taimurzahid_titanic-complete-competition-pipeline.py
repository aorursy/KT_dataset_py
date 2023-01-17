import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/titanic/train.csv')

train.head()
train.info()
test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.head()
test.info()
# Using Matplotlib

ages = train['Age'].dropna().tolist()



plt.figure(figsize=(16,5))

plt.hist(ages, histtype = 'bar', color = 'gray', 

         bins=50, density=True)

plt.xlabel('Age Groups')

plt.ylabel('Percentage of the Population')

plt.title('Age Distribution Among the Population')

plt.grid(True)

plt.show()
# You can leave the following two lines unchanged or make changes to better fit the visualization

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(16,5)})



# Using Seaborn

sns.distplot(a = ages, bins = 80, color = 'gray', vertical = False

            ).set_title('Age Distribution Among the Population')
# Using Matplotlib

sex = train['Sex'].dropna().tolist()

plt.figure(figsize=(7,5))

plt.hist(sex, histtype = 'bar', color = 'gray', bins=2, density=True)

plt.xlabel('Male vs Female')

plt.ylabel('Percentage of the Population')

plt.title('Gender Distribution Among the Population')

plt.grid(True)

plt.show()
sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(7,5)})



# Using Seaborn

sns.countplot(x="Sex", 

              data = train, 

              color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Gender Distribution Among the Population')
# Gender Distribution among the Three Classes of Population

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="Pclass", hue="Sex", 

              data = train, color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Gender Distribution Among the Three Classes of Population')
# Gender distribution among the Survived and the Deceased 

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="Survived", hue="Sex", 

              data = train, color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Gender distribution among the Survived and the Deceased')
# Survival Distribution among the Three Classes of Individuals

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="Survived", hue="Pclass", 

              data = train, color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Survival Distribution among the Three Classes of Individuals')
# Plotting the Survial of the Population with respect to their Gender and Class

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(12,5)})



sns.catplot(x="Pclass", hue="Sex", col="Survived", 

            data=train, kind="count", color = 'gray', 

            edgecolor=sns.color_palette("dark", 1))
# Using Matplotlib

SibSp = train['SibSp'].dropna().tolist()



plt.figure(figsize=(8,5))

plt.hist(SibSp, histtype = 'bar', color = 'gray', bins=8, density=True)

plt.xlabel('Number of Family Members')

plt.ylabel('Population with \'x\' Number of Family Members')

plt.title('Distribution by the Number of Family Members Aboard')

plt.grid(True)

plt.show()
# Using Seaborn

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="SibSp", data = train, 

              color = 'gray', edgecolor=sns.color_palette("dark", 1)

             ).set_title('Distribution by the Number of Family Members Aboard')
# Distribution of the Population with Different Number of Family Members Aboard Grouped by Gender

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="Sex", hue="SibSp", data = train, color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Distribution by the Number of Family Members Aboard, Grouped by Gender')
# Visualizing the Port by which the the Three Classes of the Population came aboard

sns.set(style="darkgrid")

sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(x="Pclass", hue="Embarked", data = train, color = 'gray', 

              edgecolor=sns.color_palette("dark", 1)

             ).set_title('Distribution of the Three Classes of the Population, Grouped by Port')
train.columns.values
train.head()
train.info()
## Uncomment the following line to see a list of all the unique names in the Train File

## This is used to determine the list items to include in the titles list in the next cell.



# train['Name'].unique().tolist()
titles = ['Mrs', 'Mr', 'Don', 'Jonkheer', 'Master', 'Miss', 'Major', 

          'Rev', 'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess']



def get_title(name):

    for title in titles:

        if title in name:

            return title

    return 'None'

        

print(get_title('Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)'))

print(get_title('Williams, Mr. Charles Eugene'))
train['Title'] = train['Name'].apply(lambda x: get_title(x))

train.head()
train['Age'].mean()
train['Age'].fillna(train['Age'].mean(), inplace = True)

train.head()
train.drop('Cabin', axis = 1, inplace = True)

train.head()
train.info()
train.dropna(inplace = True)

train.info()
train.head()
x_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]

y_train = train['Survived']
x_train.head()
y_train.head()
x_train = pd.get_dummies(x_train)
x_train.head()
x_train.info()
# Intializing the Model

clf = RandomForestClassifier(max_depth=75, random_state=0)
# Training the Model

clf.fit(x_train, y_train)
# Preprocessing the Test Set

test['Title'] = test['Name'].apply(lambda x: get_title(x))

test['Age'].fillna(test['Age'].mean(), inplace = True)

test.drop('Cabin', axis = 1, inplace = True)

test.head()
test.info()
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

test.info()
x_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]

x_test.head()
x_test = pd.get_dummies(x_test)
x_test.info()
total_titles = train['Title'].unique().tolist()

test_titles = test['Title'].unique().tolist()



for title in total_titles:

    if title not in test_titles:

        print('Title ' + str(title) + ' not in Test Set')

        x_test['Title_' + str(title)] = 0
x_test.info()
Survived = clf.predict(x_test)

Survived
gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

gender_submission.head()
gender_submission.info()
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': Survived})

output.head()
output.info()
output.to_csv('Submission_RandomForest.csv', index=False)