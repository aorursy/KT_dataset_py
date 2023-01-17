# Pandas for Data Management

import pandas as pd



# Numpy for Linear Algebra

import numpy as np



# Matplot for Visualization

import matplotlib.pyplot as plt



# Seaborn for Visualization

import seaborn as sns
# To display the exact memory usage in info()

pd.set_option('display.memory_usage' ,'deep')



# To display the float variables with 2 decimal places

pd.set_option('display.precision', 2)



# To display upto 100 columns while displaying a dataframe

pd.set_option('display.max_columns', 100)
# Importing Training data

train = pd.read_csv('/kaggle/input/titanic/train.csv')



# Importing Testing data

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# Printing the dimension of the datasets

print("Dimension of Training Set :", train.shape)

print("Dimension of Testing Set  :", test.shape)
# Creating a reference variable for subsetting the datasets later

train['Set'] = 'Train'

test['Set'] = 'Test'
titanic = pd.concat([train, test], axis = 0, sort = False)



print('Dimension of Titanic', titanic.shape)
titanic.head()
titanic.sample(5)
print('-'*40)



print('Info of Training Set')



print('-'*40)



train.info()



print('-'*40)



print('Info of Testing Set')



print('-'*40)



test.info()
titanic.info()
print('Missing values in Training data : \n', train.isna().sum().sort_values(ascending = False), sep = '')



print('-'*40)



print('Missing values in Testing data : \n', test.isna().sum().sort_values(ascending = False), sep = '')
titanic['Sex'].value_counts()
pd.crosstab(train['Survived'], train['Sex'], normalize = 'columns')
plt.figure(figsize = (7,7))



titanic.groupby('Sex')['Survived'].agg(Survivors = 'sum')['Survivors'].plot(kind = 'bar')



plt.title('Gender wise Survivors', fontsize = 20)



plt.xlabel('Sex', fontsize = 15)



plt.ylabel('Survivors', fontsize = 15)



plt.show()
plt.figure(figsize = (20, 7))



plt.title('Age wise Survival', fontsize = 20)



plt.ylabel('Age', fontsize = 15)



plt.xlabel('Sex', fontsize = 15)



sns.violinplot(x = 'Sex', y = 'Age', 

               hue = 'Survived', data = train, 

               split = True,

               palette = {0: "red", 1: "green"}

              )



plt.show()
list(titanic['Pclass'].sort_values().unique())
plt.figure(figsize = (12,5))



plt.title('Passengers in each Travel Class', fontsize = 20)



plt.xlabel('Passenger Class', fontsize = 15)



plt.ylabel('Number of Passengers', fontsize = 15)



titanic.groupby('Pclass')['Pclass'].count().plot(kind = 'bar');
pd.crosstab(index = train['Survived'], columns = train['Pclass'],

            normalize = 'columns', dropna = False)
titanic['Has_Cabin'] = np.where(titanic['Cabin'].isnull(), 0, 1)

titanic['Has_Cabin'].value_counts(normalize = True)
pd.crosstab(index = titanic.loc[titanic['Set'] == 'Train', 'Survived'],

            columns = titanic.loc[titanic['Set'] == 'Train', 'Has_Cabin'],

            normalize = 'columns', dropna = False)
titanic['Family_Size'] = titanic['SibSp'] + titanic['Parch'] + 1



titanic['Family_Size'].value_counts().sort_index()
pd.crosstab(index = titanic.loc[titanic['Set'] == 'Train', 'Survived'],

            columns = titanic.loc[titanic['Set'] == 'Train', 'Family_Size'],

            normalize = 'columns', dropna = False)
titanic['Alone'] = np.where(titanic['Family_Size'] == 1, 1, 0)



titanic['Alone'].value_counts()
pd.crosstab(index = titanic.loc[titanic['Set'] == 'Train', 'Survived'],

            columns = titanic.loc[titanic['Set'] == 'Train', 'Alone'],

            normalize = 'columns', dropna = False)
titanic['Title'] = titanic['Name'].map(lambda x : x.split(',')[1].split('.')[0].strip())



titanic[['Name','Title']]
print('Average Age of all passengers is %.2f yrs' % titanic['Age'].mean())
titanic.Title.value_counts()
titanic.groupby(['Title'])['Age'].agg(Avg_Age = 'mean').reset_index()
titanic.groupby(['Pclass'])['Age'].agg(Avg_Age = 'mean').reset_index()
mean_ages = titanic.groupby(['Title','Pclass'])['Age'].agg(Avg_Age = 'mean').reset_index()



mean_ages
mean_ages['Avg_Age'] = mean_ages['Avg_Age'].fillna(28.00)



mean_ages
mean_ages['Class_Title'] = mean_ages['Pclass'].map(str) + mean_ages['Title']



mean_ages.head()
titanic['Class_Title'] = titanic['Pclass'].map(str) + titanic['Title']



titanic.head()
titanic = pd.merge(titanic, mean_ages[['Class_Title','Avg_Age']], on = 'Class_Title', how = 'left')



titanic.head()
titanic['Age'] = np.where(titanic['Age'].isnull(),

                          titanic['Avg_Age'],

                          titanic['Age'])



print('Missing Values in Age variable :', titanic['Age'].isnull().sum())
print('Minimum Age of Passengers is %.2f yrs' % titanic['Age'].min())



print('Maximum Age of Passengers is %.2f yrs' % titanic['Age'].max())
titanic['Age_Group'] = np.where(titanic['Age'] < 5,

                                'Baby',

                                np.where((titanic['Age'] >= 5) & (titanic['Age'] < 12),

                                         'Child',

                                         np.where((titanic['Age'] >= 12) & (titanic['Age'] < 18),

                                                  'Teenager',

                                                  np.where((titanic['Age'] >= 18) & (titanic['Age'] < 24),

                                                           'Student',

                                                           np.where((titanic['Age'] >= 24) & (titanic['Age'] < 30),

                                                                    'Young_Adult',

                                                                    np.where((titanic['Age'] >= 30) & (titanic['Age'] < 60),

                                                                             'Adult',

                                                                             np.where(titanic['Age'] >= 60,

                                                                                      'Senior', np.nan)))))))



titanic['Age_Group'].value_counts()
pd.crosstab(index = titanic.loc[titanic['Set'] == 'Train', 'Survived'],

            columns = titanic.loc[titanic['Set'] == 'Train', 'Age_Group'],

            normalize = 'columns', dropna = False)
titanic.loc[titanic['Fare'].isnull(),:]
titanic.groupby(['Pclass','Sex','Title'])['Fare'].agg(Avg_Fare = 'mean').reset_index()
titanic.loc[(titanic['Embarked'] == 'S') &

            (titanic['Pclass'] == 3), 'Fare'].mean()
titanic['Fare'] = titanic['Fare'].fillna(titanic.loc[(titanic['Embarked'] == 'S') &

                                                     (titanic['Pclass'] == 3), 'Fare'].mean())



print('Missing Values in Fare variable :', titanic['Fare'].isnull().sum())
titanic['Embarked'].value_counts(dropna = False)
titanic['Embarked'] = titanic['Embarked'].fillna('S')



print('Missing Values in Embarked variable :', titanic['Embarked'].isnull().sum())
titanic.isnull().sum().sort_values(ascending = False)
titanic = pd.concat([titanic, pd.get_dummies(titanic['Pclass'], prefix = 'Pclass')], axis = 1)



titanic.loc[:,titanic.columns.str.contains('class')].head(10)
titanic['Female'] = titanic['Sex'].map({'female' : 1, 'male' : 0})



titanic.loc[:,['Sex','Female']].head()
titanic = pd.concat([titanic, pd.get_dummies(titanic['Embarked'], prefix = 'Embarked')], axis = 1)



titanic.loc[:,titanic.columns.str.contains('Embarked')].head()
titanic = pd.concat([titanic, pd.get_dummies(titanic['Age_Group'], prefix = 'Age_Group')], axis = 1)



titanic.loc[:,titanic.columns.str.contains('Age_Group')].head()
titanic = titanic.drop(columns = ['Cabin','Class_Title','Ticket','Pclass_3','Embarked_C'])
# For splitting the data into train and test data

from sklearn.model_selection import train_test_split



# For performing Stratified K-Fold Cross Validation

from sklearn.model_selection import StratifiedKFold



# For evaluating Cross Validation results

from sklearn.model_selection import cross_val_score



# For building Logistic Regression Model

from sklearn.linear_model import LogisticRegression



# For making Decision Tree

from sklearn.tree import DecisionTreeClassifier
skfolds = StratifiedKFold(n_splits = 10, random_state = 10)
print(list(titanic.columns))
features = ['Age', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Family_Size', 'Alone', 'Pclass_1', 'Pclass_2',

            'Female', 'Embarked_Q', 'Embarked_S']



x = titanic.loc[titanic['Set'] == 'Train', features]

y = titanic.loc[titanic['Set'] == 'Train', 'Survived']



print('Dimension of Features :', x.shape)

print('Dimension of Outcome  :', y.shape)
results = cross_val_score(estimator = LogisticRegression(solver = 'liblinear'), X = x, y = y, cv = skfolds)



print('Accuracy of Logistic  : %.3f%%' % (results.mean()*100))
results = cross_val_score(estimator = DecisionTreeClassifier(random_state = 1), X = x, y = y, cv = skfolds)



print('Accuracy of Decision Tree : %.3f%%' % (results.mean()*100))
train = titanic[titanic['Set'] == 'Train']

test  = titanic[titanic['Set'] == 'Test']
features = ['Age', 'SibSp', 'Parch', 'Fare', 'Has_Cabin', 'Alone', 'Pclass_1', 'Pclass_2',

            'Female', 'Embarked_Q', 'Embarked_S']



log = LogisticRegression(solver = 'liblinear')
log = log.fit(X = train[features], y = train['Survived'])
submission = pd.DataFrame({'PassengerId' : test['PassengerId'], 'Survived' : log.predict(test[features])}).reset_index(drop = True)

submission
submission.to_csv('./submission.csv', index = False)



print('Submission Successful!!!')