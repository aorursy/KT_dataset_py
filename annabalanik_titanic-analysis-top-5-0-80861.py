# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing modules and libraries for data visualizations, algorithm, data splitting and testing, accuracy score

import seaborn as sns

import matplotlib as plt

%matplotlib inline



import numpy as np

import pandas as pd



from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split 



from sklearn.preprocessing import LabelEncoder
train=pd.read_csv("/kaggle/input/titanic/train.csv")

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head(10)
#counting missing values of the train set

train.isnull().sum()
#counting missing values of the test set

test.isnull().sum()
#creating a function for data visualisations using seaborn

def categorical_summarized(dataframe, x=None, y=None, hue=None, palette='Set1', verbose=True):

    if x == None:

        column_interested = y

    else:

        column_interested = x

    series = dataframe[column_interested]

    print(series.describe())

    print('mode: ', series.mode())

    if verbose:

        print('='*80)

        print(series.value_counts())



    sns.countplot(x=x, y=y, hue=hue, data=dataframe, palette=palette)
#cheking general numbers: how many passengers survived and died

#we can see that there are more passengers who died

c_palette = ['tab:blue', 'tab:red']

categorical_summarized(train, y = 'Survived', palette=c_palette)
#the same thing, but using Sex - how many men and women survived. 

#Female survival chances were much higher than male ones.

categorical_summarized(train, y="Sex", hue = 'Survived', palette=c_palette)
#distribution of dead/survived passengers depending on their embarkation

categorical_summarized(train, y= "Embarked", hue = 'Survived', palette=c_palette)
#The same numbers distributed by Pclass. Predictably, the higher class - the higher survival chances

categorical_summarized(train, x="Pclass", hue = 'Survived', palette=c_palette)
#Correlation map. We can see that the highest correlation with Survived label have Pclass and Fare

#those two features have pretty high correlation.

sns.heatmap((train.loc[:,["Age", "SibSp", 'Parch', 'Fare', 'Pclass', 'Survived']]).corr(), annot=True)


sns.violinplot(x="Pclass", y="Age", hue="Survived", data = train, palette = "Set1")
#concatenating data sets

all_data = train.append(test)
#Filling missing Age values, using created Title feature

import re

all_data['Title'] = all_data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

all_data.replace({'Title': mapping}, inplace=True)

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']

for title in titles:

    age_to_impute = all_data.groupby('Title')['Age'].median()[titles.index(title)]

    all_data.loc[(all_data['Age'].isnull()) & (all_data['Title'] == title), 'Age'] = age_to_impute

    

# Substituting Age values in train and test:

train['Age'] = all_data['Age'][:891]

test['Age'] = all_data['Age'][891:]



sns.countplot(x='Title', data=all_data);

#all_data.drop('Title', axis = 1, inplace = True)

train['Title'] = all_data['Title'][:891]

test['Title'] = all_data['Title'][891:]



train['Title'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr'],[1,2,3,4,5,6],inplace=True)

test['Title'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Rev', 'Dr'],[1,2,3,4,5,6],inplace=True)
all_data['Family_Size'] = all_data['Parch'] + all_data['SibSp']



# Substituting Age values in train and test data frames:

train['Family_Size'] = all_data['Family_Size'][:891]

test['Family_Size'] = all_data['Family_Size'][891:]
all_data['Last_Name'] = all_data['Name'].apply(lambda x: str.split(x, ",")[0])

all_data['Fare'].fillna(all_data['Fare'].mean(), inplace=True)



DEFAULT_SURVIVAL_VALUE = 0.5

all_data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



for grp, grp_df in all_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin==0.0):

                all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0



print("Number of passengers with family survival information:", 

      all_data.loc[all_data['Family_Survival']!=0.5].shape[0])
for _, grp_df in all_data.groupby('Ticket'):

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin==0.0):

                    all_data.loc[all_data['PassengerId'] == passID, 'Family_Survival'] = 0

                        

print("Number of passenger with family/group survival information: " 

      +str(all_data[all_data['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in train and test:

train['Family_Survival'] = all_data['Family_Survival'][:891]

test['Family_Survival'] = all_data['Family_Survival'][891:]
#counting for how many people every ticket was usedm another grouping

all_data['Ticket_Frequency'] = all_data.groupby('Ticket')['Ticket'].transform('count')

train['Ticket_Frequency'] = all_data['Ticket_Frequency'][:891]

test['Ticket_Frequency'] = all_data['Ticket_Frequency'][891:]
#Fare per one

all_data["Fare"] = all_data["Fare"].fillna(test["Fare"].median())

all_data["Fare_per_one"]=all_data['Fare']/all_data['Ticket_Frequency']



# Making 16 Bins for Fare

all_data['FareBin'] = pd.qcut(all_data['Fare_per_one'], 16)



label = LabelEncoder()

all_data['FareBin_Code'] = label.fit_transform(all_data['FareBin'])



train['FareBin_Code'] = all_data['FareBin_Code'][:891]

test['FareBin_Code'] = all_data['FareBin_Code'][891:]



train.drop(['Fare'], 1, inplace=True)

test.drop(['Fare'], 1, inplace=True)
#Making 4 bins for Age

all_data['AgeBin'] = pd.qcut(all_data['Age'], 6)



label = LabelEncoder()

all_data['AgeBin_Code'] = label.fit_transform(all_data['AgeBin'])



train['AgeBin_Code'] = all_data['AgeBin_Code'][:891]

test['AgeBin_Code'] = all_data['AgeBin_Code'][891:]



train.drop(['Age'], 1, inplace=True)

test.drop(['Age'], 1, inplace=True)
sns.set(rc={'figure.figsize':(13,8)})

sns.heatmap((train.loc[:,['SibSp', 'Parch', 'Family_Size','FareBin_Code', 

            'AgeBin_Code','Pclass', 'Ticket_Frequency', 'Family_Survival','Survived']]).corr(), annot=True)
#replacing 'malee' and 'female' with 0 and 1 for the algoritm

train['Sex'].replace(['male','female'],[0,1],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)



#dropping unnecessary features

train.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 

               'Embarked'], axis = 1, inplace = True)

test.drop(['Name','PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin', 

              'Embarked'], axis = 1, inplace = True)
train.head()
y = train['Survived']

train_df = train.drop('Survived', 1)

test_df = test.copy()
#test/train splitting

my_train, my_test, my_y, my_res = train_test_split(train_df, y, test_size=0.1, random_state=42)
#creating trial RFC model

model_try = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)

model_try.fit(my_train, my_y)

preds = model_try.predict(my_test)

print(accuracy_score(my_res, preds))
#feature importance of the algorithm

pd.Series(model_try.feature_importances_, index = my_train.columns).nlargest(12).plot(kind = 'barh',

                            figsize = (10, 10),title = 'Feature importance from Random Forest').invert_yaxis()
test = pd.read_csv('/kaggle/input/titanic/test.csv')
#creating the model, predictions and output

model = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)

model.fit(train_df, y)

predictions = model.predict(test_df)

output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print(predictions)
#feature importance iss a bit different from the above model on train data

pd.Series(model.feature_importances_, index = train_df.columns).nlargest(30).plot(kind = 'barh',

                            figsize = (10, 10),title = 'Feature importance from Random Forest').invert_yaxis()