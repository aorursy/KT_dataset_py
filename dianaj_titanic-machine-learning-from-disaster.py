# Source: https://www.kaggle.com/alankritamishra/titaniceasyway/notebook#Achieving-80%-accuracy-in-easy-way.



# Importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting library

import seaborn as sns # a data visualiozation libray based off of matplot library



# the output of plotting commands is displayed inline

%matplotlib inline



# Importing, reading and showing train data.

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head() # head function only shoes the first five entries.
# Importing, reading and showing test data.

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# Looking at shape of the train data set.

# The x coordinate tell us the amount of rows the dataset contains, and the y coordinate tells us the amount of columns.

train_data.shape
# Now, we are looking at the shape of the test data set.

# We can see that the test data set is missing a column, which is the Survived column. I gathered this information from the test_data table above with the command "test_data.head()"

test_data.shape
# I am now presenting additional information of the train_data dataset. I see that some columns contain null values - Age, Cabin, and Embarked.

train_data.info()
# Doing the same for the test_data dataset. Again, we see null values in columns Age, Fare, and Cabin

test_data.info()
# A more clear way to see the null values for each column is to use the isnull() function, then add them together using the sum() function

train_data.isnull().sum()
test_data.isnull().sum()
# We will use a heatmap to visulize our data. The heatmap is a good way to show missing values because it indicates missing areas in an entire bar

# (which represents the column in question).

sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='spring')
sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='spring')
# We begin by indicating which features are "catagorical," versus numerical. The way we do this is by indicating whether the data type is an object or not.

# If it is an object, then it is a catagorical features, but if it is not an object, it is a numerical feature.

categorical_features = [features for features in train_data.columns if train_data[features].dtypes=='O']

categorical_features
# These are our numerical features

numerical_features = [features for features in train_data.columns if train_data[features].dtypes!='O']

numerical_features
# Relationship between ticket class and passengers survived

sns.barplot(x="Pclass",y="Survived",data=train_data)
# Relationship between sex and passengers survived

sns.barplot(x="Sex",y="Survived",data=train_data)
# Relationship between which port passangers embarked on and passengers survived

sns.barplot(x="Embarked",y="Survived",data=train_data)
# Relationship between which cabin passangers stayed in and the passangers that survived

sns.barplot(x="Cabin",y="Survived",data=train_data)
# Relationship between age of passangers and the passangers that survived

sns.barplot(x="Age",y="Survived",data=train_data)
# Showing our survived data just according to the entry

y_train= train_data['Survived']

y_train
# Data entries foe the train_data dataset

ntrain = train_data.shape[0]

ntrain
# Data entries for the test_data dataset

ntest = test_data.shape[0]

ntest
# Here, we are finally combining our data from the train and test datasets

combined_data = pd.concat((train_data, test_data)).reset_index(drop=True)

combined_data.drop(['Survived'], axis=1, inplace=True)

print("Our combined data is: {}".format(combined_data.shape))
# Checking null values

combined_data.isnull().sum()
# Here, we are getting the median ages of the different subsets of {Sex, Pclass}. 

# These numbers will help us in filling any null values. They will default to the median age.



age_by_pclass_sex = combined_data.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))

print('Median age of all passengers: {}'.format(combined_data['Age'].median()))
# Filling our null values with median, then showing that there are no longer any null values in the Age column

combined_data['Age']= combined_data.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median()))

combined_data['Age'].isnull().sum()
# Doing the same with the embark column. If there is a null value, the mode function will fill that value with the most common embarked variable (S)

mode = combined_data['Embarked'].mode()

combined_data['Embarked']= combined_data['Embarked'].fillna('mode')

mode # this should product 
# Median fare, filling null values for fare

med_fare= combined_data.groupby(['Pclass','Parch','SibSp']).Fare.median()[3][0][0]

combined_data['Fare'] = combined_data['Fare'].fillna(med_fare)

med_fare
# Now, let's take a look at how many null values (should be zero). 

# Note: When a variable has over 50% null values, it is best to just get rid of it completely.

combined_data.isnull().sum()
# Instead of having two catagories, the siblings / spouses (SibSp) and parents / children (Parch), we will simply create one feature called FamilySize,

# which is the sum of both of these catagories

combined_data['FamilySize']= combined_data['SibSp']+combined_data['Parch']+1
# Now instead of full names, we will just use titles of the passenger

combined_data['Title'] = combined_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

combined_data['Title']
# A FarePerPerson column is now needed since we have combined passangers into families. We can achieve the fare per person by dividing the fare by the family size.

combined_data['FarePerPerson']= combined_data['Fare']/combined_data['FamilySize']

combined_data['FarePerPerson']
#dropping columns that are not important

combined_data.drop(['Ticket','SibSp','Name','Parch','Cabin'],axis=1,inplace=True)
# Dropping our fare

combined_data = combined_data.drop(['Fare'],axis=1)
# A look at our new table

combined_data.head()
# These are the catagory feature left to alter

categorical_features=[features for features in combined_data.columns if combined_data[features].dtypes=='O']

categorical_features
from sklearn.preprocessing import LabelEncoder



# Process the columns, then apply the LabelEncoder to categorical features so that we can make them numeric values

lbl= LabelEncoder()



# For titles

lbl.fit(list(combined_data['Title'].values)) 

combined_data['Title'] = lbl.transform(list(combined_data['Title'].values))



# For sex

lbl.fit(list(combined_data['Sex'].values)) 

combined_data['Sex'] = lbl.transform(list(combined_data['Sex'].values))



# For embarked

lbl.fit(list(combined_data['Embarked'].values)) 

combined_data['Embarked'] = lbl.transform(list(combined_data['Embarked'].values))
# Checking our table

combined_data.head()
# Seperating data into the train and test datasets

train_data = combined_data[:ntrain]

test_data = combined_data[ntrain:]
# Seeing table

train_data.corr()
# Creating graph

plt.subplots(figsize=(15,8))



sns.heatmap(train_data.corr(),annot=True,cmap='Oranges')
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
# Our final table with catgorical values turned numeric

x = train_data

x
GBR = GradientBoostingClassifier(n_estimators=100, max_depth=4)

GBR.fit(x,y_train)
#finalMdG is the prediction by GradientBoostingClassifier

finalMdG=GBR.predict(test_data)

finalMdG
ID = test_data['PassengerId']
submission=pd.DataFrame()

submission['PassengerId'] = ID

submission['Survived'] = finalMdG

submission.to_csv('submissiongb.csv',index=False)
rd=RandomForestClassifier()
rd.fit(x,y_train)
#finalMdR is the prediction by RandomForestClassifier

finalMdR=rd.predict(test_data)

finalMdR
# Final submission!

submission=pd.DataFrame()

submission['PassengerId'] = ID

submission['Survived'] = finalMdR

submission.to_csv('submissionrd.csv',index=False)