import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Combine both datasets to apply common operations

combine = [df_train, df_test]
# Print the training data information

print('TRAIN INFO \n',df_train.info())

# Print a separator line

print('-'*50)

# Print the test data information

print('TEST INFO \n',df_test.info())
df_train.head()
df_train.describe()
df_train.describe(include=['O'])
sns.barplot(x='Pclass', y='Survived', data=df_train.groupby('Pclass', as_index=False).agg({'Survived':'mean'}))

plt.xlabel(s='Pclass',fontsize=16)

plt.ylabel(s='Survival Rate',fontsize=16)

plt.title('Survival Rate by Pclass', fontsize=18)
sns.barplot(x='Sex', y='Survived', data=df_train.groupby('Sex', as_index=False).agg({'Survived':'mean'}))

plt.xlabel(s='Gender',fontsize=16)

plt.ylabel(s='Survival Rate',fontsize=16)

plt.title('Survival Rate by Gender', fontsize=18)
g = sns.FacetGrid(df_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
df_embarked_gender = df_train.groupby(['Embarked','Sex'], as_index=False).agg({'Survived':'mean'})

sns.barplot(x='Embarked', y='Survived', hue='Sex', data=df_embarked_gender)

plt.xlabel(s='Embarked',fontsize=16)

plt.ylabel(s='Survival Rate',fontsize=16)

plt.title('Survival Rate by Embarked port and Gender', fontsize=18)
# Drop columns Cabin and Ticket on both datasets

df_train = df_train.drop(['Cabin','Ticket'], axis=1)

df_test = df_test.drop(['Cabin','Ticket'], axis=1)



combine = [df_train, df_test]
for dataset in combine: # For each Dataset

    for gender in dataset['Sex'].unique(): # For each gender

        for embarked in dataset.loc[~dataset['Embarked'].isnull(),'Embarked'].unique(): # For each port

            for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class

                

                # Get a Dataframe with not null values to guess the age (using dropna)

                guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass),'Age'].dropna()

                

                # Get the mean

                guessed_age = guess_df.mean()

                

                # Set it to the dataset

                dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass),'Age'] = guessed_age
for dataset in combine: # For each Dataset

    for gender in dataset['Sex'].unique(): # For each gender

        for embarked in dataset.loc[~dataset['Embarked'].isnull(),'Embarked'].unique(): # For each port

            for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class

                for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age

                    

                    # Get a Dataframe with not null values to guess the fare (using dropna)

                    guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Fare'].dropna()

                

                    # Get the median

                    guessed_fare = guess_df.median()

                    

                    # Set it to the dataset

                    dataset.loc[(dataset['Fare'].isnull()) & (dataset['Sex'] == gender) & (dataset['Embarked'] == embarked) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Fare'] = guessed_fare







# Assign the median of the entire dataset for the rows we couldn't guess the fare

fare_median = df_train.loc[~df_train['Fare'].isnull(), 'Fare'].median()



df_train.loc[df_train['Fare'].isnull(), 'Fare'] = fare_median

df_test.loc[df_test['Fare'].isnull(), 'Fare'] = fare_median



combine = [df_train, df_test]
def impute_embarked(use_fare=True):

    for dataset in combine: # For each Dataset

        # Should use fare to impute Embark port?

        if(use_fare):

            for gender in dataset['Sex'].unique(): # For each gender

                for fare in dataset.loc[~dataset['Fare'].isnull(),'Fare'].unique(): # For each fare

                    for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class

                        for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age

                        

                            # Get a Dataframe with not null values to guess the fare (using dropna)

                            guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Fare'] == fare) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'].dropna()

                        

                            # if the dataframe has values

                            if (len(guess_df) > 0):

                                # Get the mode

                                guessed_port = guess_df.mode()[0]

                            

                                # Set it to the dataset

                                dataset.loc[(dataset['Embarked'].isnull()) & (dataset['Sex'] == gender) & (dataset['Fare'] == fare) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'] = guessed_port

        # Dont use fare to impute Embarked port

        else:

            for gender in dataset['Sex'].unique(): # For each gender

                for pclass in dataset.loc[~dataset['Pclass'].isnull(),'Pclass'].unique(): # For each class

                    for age in dataset.loc[~dataset['Age'].isnull(),'Age'].unique(): # For each age

                        

                        # Get a Dataframe with not null values to guess the fare (using dropna)

                        guess_df = dataset.loc[(dataset['Sex'] == gender) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'].dropna()

                    

                        # if the dataframe has values

                        if (len(guess_df) > 0):

                            # Get the mode

                            guessed_port = guess_df.mode()[0]

                        

                            # Set it to the dataset

                            dataset.loc[(dataset['Embarked'].isnull()) & (dataset['Sex'] == gender) & (dataset['Pclass'] == pclass) & (dataset['Age'] == age),'Embarked'] = guessed_port

                            

impute_embarked(use_fare=False)



# Get the most frequent Port from all data for the ones that we couldn't find.

freq_port = df_train.loc[~df_train['Embarked'].isnull(),'Embarked'].mode()[0]

# Set the most frequent port to the null values

df_train.loc[df_train['Embarked'].isnull(),'Embarked'] = freq_port

df_test.loc[df_test['Embarked'].isnull(),'Embarked'] = freq_port



combine = [df_train, df_test]
# Get Dummies for Sex Column

df_sex_dummies = pd.get_dummies(df_train['Sex'], prefix='sex_', drop_first=True)

df_train = pd.concat([df_train, df_sex_dummies], axis=1)   

df_train = df_train.drop('Sex', axis=1)



df_sex_dummies = pd.get_dummies(df_test['Sex'], prefix='sex_', drop_first=True)

df_test = pd.concat([df_test, df_sex_dummies], axis=1)   

df_test = df_test.drop('Sex', axis=1)
# Get Dummies for Sex Column

df_sex_dummies = pd.get_dummies(df_train['Embarked'], prefix='embarked_', drop_first=True)

df_train = pd.concat([df_train, df_sex_dummies], axis=1)   

df_train = df_train.drop('Embarked', axis=1)



df_sex_dummies = pd.get_dummies(df_test['Embarked'], prefix='embarked_', drop_first=True)

df_test = pd.concat([df_test, df_sex_dummies], axis=1)   

df_test = df_test.drop('Embarked', axis=1)
# Get Dummies for Sex Column

df_sex_dummies = pd.get_dummies(df_train['Pclass'], prefix='pclass_', drop_first=True)

df_train = pd.concat([df_train, df_sex_dummies], axis=1)   

df_train = df_train.drop('Pclass', axis=1)



df_sex_dummies = pd.get_dummies(df_test['Pclass'], prefix='pclass_', drop_first=True)

df_test = pd.concat([df_test, df_sex_dummies], axis=1)   

df_test = df_test.drop('Pclass', axis=1)
# Drop the Name column on both datasets

df_train = df_train.drop(['Name'], axis=1)

df_test = df_test.drop(['Name'], axis=1)



combine = [df_train, df_test]
for dataset in combine:

    # Family Size Feature

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



    # Is alone Feature

    dataset['IsAlone'] = 0

    dataset.loc[(dataset['FamilySize'] == 1), 'IsAlone'] = 1





# Display a heatmap to see the correlation between the values.

sns.heatmap(df_train[['FamilySize','Survived','IsAlone']].corr(), annot=True)
df_train = df_train.drop(['FamilySize','SibSp','Parch'], axis=1)

df_test = df_test.drop(['FamilySize','SibSp','Parch'], axis=1)



combine = [df_train, df_test]
# Get the Features without the Label

X_train = df_train.drop(['Survived','PassengerId'], axis=1)

Y_train = df_train['Survived']



# Get the Test values

X_test = df_test.drop(['PassengerId'], axis=1).copy()
from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=50)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

score = random_forest.score(X_train, Y_train)



print('Accuracy',score)
submission = pd.DataFrame(

    {

        'PassengerId': df_test['PassengerId'],

        'Survived': Y_pred

    })



#submission.to_csv('../input/my_submission.csv', index=False)

submission


