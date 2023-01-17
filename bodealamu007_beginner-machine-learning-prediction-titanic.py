# Import the libraries

import numpy as np

import pandas as pd

# Data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style="darkgrid")
# import Machine learning 

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# Import the data into a dataframe



titanic = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Explore the dataset

titanic.head(10)
test.head()
# Basic statistics

titanic.describe()
test.describe()
titanic.info()
test.info()
# Check for missing values

titanic.isnull().sum()
test.isnull().sum()


def count(dataframe, column_name):

    """

    Written by Olabode Alamu 15th Jan 2017

    Function calculates the percentage of the different categorical variables.

    The function takes in two parameters- dataframe and column_name and returns 

    the percentage of each category in any particular column.

    

    dataframe = Name of the dataframe under consideration

    column_name = string of column name of the column with categorical variables

    """

    try:

        total = len(dataframe[column_name]) # Counts the number of rows

        for i in dataframe[column_name].unique():

            # Counts the number of rows in each class

            Count = len(dataframe[dataframe[column_name]==i])

            print('Percentage in column '+column_name +' with value '+ str(i)+ ' is' , (Count/total)*100, '%')

    

    except:

        print('Column name not found in dataset or Dataframe doesnt exist.')

            
sns.countplot(x = 'Sex', data = titanic, palette='Set1')
count(dataframe=titanic,column_name='Sex')
# Visualize the proportion of males to females survived

sns.countplot(x='Sex' , hue= 'Survived', data = titanic,  palette='Set1')
# What is the exact number of men that died?

M = titanic[titanic['Sex']=='male']

len(M[M['Survived']== 0])
len(titanic[titanic['Sex']=='male'][titanic[titanic['Sex']=='male']['Survived']== 0])
# What is the exact number of women that died?

F = titanic[titanic['Sex']=='female']

len(F[F['Survived']== 0])
# Visualize the distribution amongst the different classes

sns.countplot(x = 'Pclass', data= titanic , palette='Set1')
# Shows the percentage of passengers in each class

count(titanic,'Pclass')
sns.countplot(x = 'Pclass', data = titanic, hue= 'Survived', palette='Set1')
sns.factorplot(x = 'Pclass', col = 'Survived',hue = 'Sex'

               , data = titanic, kind = 'count',palette='Set1')
# How many women in First class died?

F = titanic[titanic['Sex']=='female'] # Dataframe showing females onboard

Female_died = F[F['Survived']== 0] # Dataframe showing females that died

# Dataframe showing females in first class that died

Female_died[Female_died['Pclass']==1] 
# Visualize the distribution

sns.countplot(x = 'SibSp', data = titanic,palette='Set1')

count(titanic,'SibSp')
sns.countplot(x = 'SibSp', data = titanic, hue= 'Survived', palette='Set1')
# What group had 8 siblings?

titanic[titanic['SibSp']==8]
sns.countplot(x = 'Parch', data = titanic, palette='Set1')
sns.countplot(x = 'Parch', data = titanic, hue= 'Survived', palette='Set1')
# Create a column that shows the size of the family

titanic['Family Size'] = titanic['SibSp']+titanic['Parch']+ 1

# Create a column that shows the size of the family

test['Family Size'] = test['SibSp']+test['Parch']+ 1
titanic.head(3)
sns.countplot(x = 'Family Size', data = titanic, hue= 'Survived', palette='Set1')
titanic['Embarked'].fillna(value = 'S', inplace = True)
# What was the percentage of passengers that embarked at the different ports?

count(dataframe=titanic, column_name='Embarked')
sns.countplot(x = 'Embarked', data = titanic, hue= 'Survived', palette='Set1')
sns.factorplot(col = 'Embarked', x = 'Pclass', data = titanic

               , kind = 'count', palette='Set1')
titanic['Fare per person']= titanic['Fare']/ titanic['Family Size']
test.isnull().sum()
# fill the missing Fare value in the test dataset with the median fare value

test['Fare'].fillna(value = test['Fare'].median(), inplace = True)
test['Fare per person']= test['Fare']/ test['Family Size']
sns.distplot(a= titanic['Fare per person'], bins=50, )
# What was the average fare paid per class?

sns.barplot(data = titanic, x = 'Pclass', y = 'Fare per person', palette='Set1')


sns.factorplot(data = titanic, x = 'Pclass', col = 'Sex'

            ,y='Fare per person',hue='Survived' , palette='Set1', kind = 'bar')
titanic.isnull().sum()
test.isnull().sum()
# Check the distribution of the age before filling the holes

sns.distplot(a = titanic['Age'].dropna(), bins=40)
sns.factorplot(y= 'Age', data = titanic, x = 'Pclass', kind = 'box', col = 'Survived', palette='Set1')
def age_input(dataframe):

    """

    This function fills the missing age values with random numbers generated between two values.

    The first value is the mean - standard deviation while the second value is the mean plus the standard deviation.

    

    Accepts the name of the dataframe as input and returns a dataframe with no missing values in the Age column

    """

    Number_missing_age = dataframe['Age'].isnull().sum() # Counts the number of nan values

    Mean_age = dataframe['Age'].mean() # calculates the mean values

    Std_age = dataframe['Age'].std() # calculates the standard deviation values

    

    # Generates random numbers the size of the missing values

    random_age = np.random.randint(low =(Mean_age-Std_age) , high= (Mean_age+Std_age), size= Number_missing_age)

    dataframe.loc[:,'Age'][np.isnan(dataframe['Age'])]= random_age

    #df.loc[:,'B'][np.isnan(df['B'])]= filll

    
age_input(titanic)
titanic.isnull().sum()
age_input(test)
test.isnull().sum()
sns.factorplot(y= 'Age', data = titanic, x = 'Pclass', kind = 'box', col = 'Survived', palette='Set1')
# Check the distribution of the age after filling the holes

sns.distplot(a = titanic['Age'].dropna(), bins=40, )
sns.factorplot(y= 'Age', data = titanic, x = 'Sex', kind = 'bar', col = 'Survived', palette='Set1')
test.head()
def embarked_dummy(dataframe, label_drop):

    

    """

    Function creates dummy variable for the Embarked column, drops one of the column in the dummy column 

    and joins to the previous dataframe. This function also drops the columns with the names in the list

    column_drop in the passed dataframe.

    

    dataframe = Name of the dataframe

    label_drop = string: String of categorical value in Sex column which you would like to drop

    column_drop = list: list of column labels which you want to be dropped

    

    """

    import pandas as pd

    embarked_dummy = pd.get_dummies(data = dataframe['Embarked'])

    # Drop column in dummy column

    embarked_dummy.drop(labels= label_drop, inplace=True, axis=1)

    

    # Merge to the dataset 

    dataframe= dataframe.join(embarked_dummy)



    return dataframe

    
titanic = embarked_dummy(dataframe=titanic, label_drop='S')
titanic.head()
test = embarked_dummy(dataframe=test, label_drop='S')
test.head()
def gender_dummy(dataframe, label_drop, column_drop):

    

    """

    Function creates dummy variable for the Sex column, drops one of the column in the dummy column 

    and joins to the previous dataframe. This function also drops the columns with the names in the list

    column_drop in the passed dataframe.

    

    dataframe = Name of the dataframe

    label_drop = string: String of categorical value in Sex column which you would like to drop

    column_drop = list: list of column labels which you want to be dropped

    

    """

    import pandas as pd

    gender_dummy = pd.get_dummies(data = dataframe['Sex'])

    # Drop column in dummy column

    gender_dummy.drop(labels= label_drop, inplace=True, axis=1)

    

    # Merge to the dataset 

    dataframe= dataframe.join(gender_dummy)

    # Drop Sex column

    dataframe.drop(labels = column_drop, axis = 1, inplace = True )

    

    return dataframe

    

    
dropped = ['Cabin', 'Sex', 'SibSp', 'Parch', 'Ticket','Fare', 'Embarked','Name']
# apply to train dataset

titanic = gender_dummy(dataframe=titanic, label_drop='male', column_drop=dropped)
titanic.head(3)
# apply to train dataset

test = gender_dummy(dataframe=test, label_drop='male', column_drop=dropped)
test.head(3)
# define training and testing sets



X_train = titanic.drop(["Survived",'PassengerId'],axis=1)

Y_train = titanic["Survived"]

X_test  = test.drop("PassengerId",axis=1).copy()
# Names of classifiers

names = ['LogisticRegression',"Nearest Neighbors", "Linear SVM", "RBF SVM","Random Forest", "Naive Bayes"]
classifiers = [LogisticRegression(),

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    GaussianNB()]
def classifer_iterate(names, classifiers, X_train, Y_train):

    import matplotlib.pyplot as plt

    plt.Figure(figsize=(20,6))

    D = {}

    for name, clf in zip(names, classifiers):

        clf.fit(X = X_train,y = Y_train)

        score = clf.score(X_train, Y_train)

        D[name]= score

        print('Score of ' + name + ' is ', score)

        

    print('-------------------------------------------------------------------------')

    plt.bar(range(len(D)), list(D.values()))

    
classifer_iterate(names=names, classifiers=classifiers,X_train=X_train, Y_train=Y_train)
#  Support Vector Machine

SVM_RBF = SVC(gamma=2, C=1)

SVM_RBF.fit(X_train, Y_train)

Y_pred = SVM_RBF.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })
submission
submission.to_csv('submit.csv', index=False)
print('The end')