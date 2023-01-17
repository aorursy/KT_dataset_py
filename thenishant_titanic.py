#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)



from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



from subprocess import check_output

print(check_output(['ls', '../input/titanic']).decode('utf8')) #check the files available in the directory
#Now let's import and put the train and test datasets in  pandas dataframe



train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#display the first five rows of the train dataset.

train.head(5)
#display the first five rows of the test dataset.

test.head(5)
#check the numbers of samples and features

print(f'The train data size before dropping Id feature is : {train.shape}')

print(f'The test data size before dropping Id feature is : {test.shape} ')



#Save the 'Id' column

train_ID = train['PassengerId']

test_ID = test['PassengerId']



#Now drop the 'Id' and 'Ticket' colum since it's unnecessary for  the prediction process.

train.drop('PassengerId', axis = 1, inplace = True)

test.drop('PassengerId', axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print(f'\nThe train data size after dropping Id feature is : {train.shape}')

print(f'The test data size after dropping Id feature is : {test.shape} ')
ntrain = train.shape[0]

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['Survived'], axis=1, inplace=True)

print(f'all_data size is : {all_data.shape}')
#Find percentage of missing data

missing_data = (all_data.isnull().sum() / len(all_data)) * 100

missing_data = missing_data[missing_data != 0].sort_values(ascending=False)



#Plot the missing data

plt.figure(figsize=(12,6))

sns.barplot(x=missing_data.index, y=missing_data)

plt.xlabel('Features',fontsize=12)

plt.ylabel('Percent of missing values', fontsize=12)

plt.title('Percent missing data by feature', fontsize=15)
# 'Fare' and 'Embarked' have only few missing values. Fill the median for 'Fare' and mode for 'Embarked'.

all_data.fillna({'Fare':train['Fare'].median(), 'Embarked':train['Embarked'].mode()[0]}, inplace = True)



#Since 'Age' is correlated with 'Parch' and 'SibSp', we fill median by grouping the data based on them.

missing_age_index = list(all_data['Age'][all_data["Age"].isnull()].index)

default_age_median = train['Age'].median()

for i in missing_age_index:

    age_median = train['Age'][(train['Parch']==all_data.iloc[i]['Parch']) & (train['SibSp']==all_data.iloc[i]['SibSp'])].median()

    if np.isnan(age_median):

        all_data['Age'].iloc[i] = default_age_median 

    else:

        all_data['Age'].iloc[i] = age_median

    

#Fill Cabin with missing values with 'U' for unknown cabin number and for others take the first letter. 

all_data['Cabin'] = pd.Series(['U' if pd.isnull(i) else i[0] for i in all_data['Cabin']])
#Check remaining missing values if any 

missing_data = (all_data.isnull().sum() / len(all_data)) * 100

missing_data = missing_data[missing_data != 0].sort_values(ascending=False)

missing_data
#Extract 'Title' from 'Name'

title = [i.split(",")[1].split(".")[0].strip() for i in all_data["Name"]]

all_data["Title"] = pd.Series(title)

all_data["Title"].value_counts()
#Drop the 'Name' feature

all_data.drop('Name', axis=1, inplace = True)



# Simplify the 'Title' feature

all_data['Title'] =all_data['Title'].replace('Mlle', 'Miss')

all_data['Title'] = all_data['Title'].replace(['Mme','Lady','Ms'], 'Mrs')

all_data['Title'][(all_data['Title'] !=  'Master') & (all_data['Title'] !=  'Mr') & (all_data['Title'] !=  'Miss')  & 

                      (all_data['Title'] !=  'Mrs')] = 'Others'

all_data["Title"].value_counts()
all_data['Sex'] = all_data['Sex'].map({'male' : 0, 'female' : 1})
#Extract first character from each ticket

all_data['Ticket'] = all_data['Ticket'].map(lambda x : x[0])

all_data['Ticket'].value_counts()
# group tickets from A to 8 in one group 4

all_data['Ticket'] = all_data['Ticket'].replace(['A','W','7','F', '4', '6', 'L', '5', '9', '8'],'4')

all_data['Ticket'].value_counts()
all_data['Cabin'].value_counts()
#Group  'A', 'F', 'G', 'T' into one group 'A'

all_data['Cabin'] = all_data['Cabin'].replace(['A','F','G','T'], 'A')

all_data['Cabin'].value_counts()
all_data = pd.get_dummies(all_data)



#Check that number of rows is preserved

print(f'all_data size is : {all_data.shape}')
#Separate the train and test data

X_train = all_data[:ntrain]

y_train = train['Survived']

X_test = all_data[ntrain:]



#Check whether the datasets have correct size

print(f'train size is : {X_train.shape}')

print(f'target size is: {y_train.shape}')

print(f'test size is : {X_test.shape}')
# We tune max_depth to avoid over fitting

max_depth = [x for x in range(5,10)]

cv_score = [cross_val_score(RandomForestClassifier(max_depth = m, n_jobs=-1, random_state = 0), X_train, y_train, 

                                     scoring = 'accuracy', cv=10, n_jobs=-1).mean() for m in max_depth]

cv_score

plt.figure(figsize = (12,6))

sns.lineplot(x=max_depth, y = cv_score)

plt.xlabel('max_depth')

plt.ylabel('Accuracy')

plt.title('Variation of accuracy with max_depth')

    
forest = RandomForestClassifier(max_depth = 7, random_state=0, n_jobs=-1)

forest.fit(X_train,y_train)



#Find Accuracy

accuracy = cross_val_score(forest, X_train, y_train, scoring='accuracy', cv=10, n_jobs=-1).mean()

print(f'Accuracy on training set is {round(accuracy,5)}')

y_test = forest.predict(X_test)

submission = pd.DataFrame({'PassengerId' : test_ID, 'Survived' : y_test})

submission.to_csv('submission.csv', index=False)