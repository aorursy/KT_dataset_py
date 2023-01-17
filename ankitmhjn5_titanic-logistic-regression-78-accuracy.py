import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
# reading CSV
train_data = pd.read_csv('../input/titanic/train.csv')
# shape of training data
train_data.shape
train_data.head()
# getting the type of each feature and number of non-null values
train_data.info()
# finding the missing values column
missing_values = 100 * train_data.isna().sum() / len(train_data)
missing_values[missing_values > 0]
# dropping the feature Cabin
train_data = train_data.drop(['Cabin'], axis=1)
# lets calculate the missing value of column Embarked based on mode
train_data['Embarked'].value_counts()
# replacing the missing value of column Embarked with 'S' as occurence of S is more
train_data['Embarked'] = train_data['Embarked'].fillna('S')
# finding the mean age based on gender
train_data.groupby('Sex')['Age'].mean()
# for the age column we gonna replace the null values to mean of gender 
train_data['Age'] = train_data.groupby('Sex')['Age'].apply(lambda row: row.fillna(row.mean()))
# changing the Age column to int
train_data['Age'] = train_data['Age'].astype(int)
# lets retreive title from name column
train_data['Title'] = train_data['Name'].apply(lambda row: row.split(", ")[1].split(".")[0])
train_data['Title'].value_counts()
# let's combine the title Dr, Rev and other with value count < 10 to others
title = ['Mr', 'Miss', 'Mrs', 'Master']
def get_title(row):
    if row in title:
        return row
    else:
        return 'Others'

train_data['Title'] = train_data['Title'].apply(lambda row: get_title(row))
train_data['Title'].value_counts()
# lets see our changed dataframe
train_data.head()
# we can also make the bins for age columns
def get_age_binning(row):
    if row <= 18:
        return 'Children'
    elif row > 18 and row <= 26:
        return 'Youth'
    elif row > 26 and row <= 60:
        return 'Adult'
    elif row > 60:
        return 'Senior Citizen'
    else:
        ""

train_data['Age_Bins'] = train_data['Age'].apply(lambda row: get_age_binning(row))
train_data['Age_Bins'].value_counts()
# we can identify if the person was alone on ship or not by checking sibsp = 0 and sibsp = Parch
def is_alone(sibsp, parch):
    if parch == 0 and sibsp == parch:
        return "Yes"
    else:
        return "No"

train_data['Alone'] = train_data.apply(lambda row: is_alone(row.SibSp, row.Parch), axis=1)
train_data['Alone'].value_counts()
# lets view the distribution of column Fare
sns.distplot(train_data['Fare'])
train_data.describe()
# lets create binning for fare
def get_fare_type(row):
    if row <=100:
        return "Low"
    elif row > 100 and row <=200:
        return "Mid"
    else:
        return "High"

train_data['Fare_Type'] = train_data['Fare'].apply(lambda row : get_fare_type(row))
train_data['Fare_Type'].value_counts()
# we can get rid of below columns 
# PassengerId, Name, Age, Ticket, Fare
train_data = train_data.drop(['PassengerId', 'Name', 'Age', 'Fare', 'Ticket'], axis=1)
# changing the feature Pclass, SibSp, Parch to category  
train_data['Pclass'] = train_data['Pclass'].astype('category')
train_data['SibSp'] = train_data['SibSp'].astype('category')
train_data['Parch'] = train_data['Parch'].astype('category')
# lets view the final features datatype 
train_data.info()
# lets view our final dataframe 
train_data.head()
# count of people survived
sns.countplot(train_data.Survived)
# count of males and females
sns.countplot(train_data.Sex)
# count of people on ship based on fare_type
sns.countplot(train_data.Fare_Type)
# count of people on ship based on pclass
sns.countplot(train_data.Pclass)
# count of people on ship based on SibSp
sns.countplot(train_data.SibSp)
# count of people on ship based on Parch
sns.countplot(train_data.Parch)
sns.countplot(train_data.Alone)
sns.countplot(train_data.Age_Bins)
sns.countplot(train_data.Sex, hue=train_data.Survived)
sns.countplot(train_data.Fare_Type, hue=train_data.Survived)
sns.countplot(train_data.Age_Bins, hue=train_data.Survived)
sns.countplot(train_data.Alone, hue=train_data.Survived)
train_data_copy = train_data.copy()
train_data_copy['Sex'] = train_data_copy['Sex'].apply(lambda row: 1 if row == 'Male' else 0)
train_data_copy['Alone'] = train_data_copy['Alone'].apply(lambda row: 1 if row == 'Yes' else 0)
cols_to_encode = ['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 'Fare_Type']
encoded_data = pd.get_dummies(train_data_copy[cols_to_encode])
encoded_data.head()
train_data_copy = pd.concat([train_data_copy, encoded_data], axis=1)
train_data_copy.columns
train_data_copy = train_data_copy.drop(['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 
                                                  'Fare_Type'], axis=1)
train_data_copy.head()
# lets find the correlation matrix 
plt.figure(figsize=(20,10))
sns.heatmap(train_data_copy.corr(), cmap='YlGnBu', annot=True)
plt.show()
X_train = train_data_copy.drop(['Survived'], axis=1)
Y_train = train_data_copy['Survived']
log_reg_model = LogisticRegression()
rfe = RFE(log_reg_model, 8)
rfe = rfe.fit(X_train, Y_train)
cols = X_train.columns[rfe.support_]
# building the logistic regression model using the 8 best features selected using RFE
log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train[cols], Y_train)
y_pred = log_reg.predict(X_train[cols])
accuracy_score(Y_train, y_pred)
confusion_matrix(Y_train, y_pred)
# reading the test data
test_data = pd.read_csv('../input/titanic/test.csv')
# getting the shape of test data
test_data.shape
# finding missing percentge for test data
missing_data = 100 * test_data.isna().sum() / len(test_data)
missing_data[missing_data > 0]
# finding mean age in test data
test_data.groupby('Sex')['Age'].mean()
# replacing the missing with mean value 
test_data['Age'] = test_data.groupby(['Sex'])['Age'].apply(lambda row: row.fillna(row.mean()))
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

# performing binning for the fare
test_data['Fare_Type'] = test_data['Fare'].apply(lambda row : get_fare_type(row))
test_data['Fare_Type'].value_counts()
# retrieving the title from the name
test_data['Title'] = test_data['Name'].apply(lambda row: row.split(", ")[1].split(".")[0])
test_data['Title'] = test_data['Title'].apply(lambda row: get_title(row))
test_data['Title'].value_counts()
# performing the binning for the age column
test_data['Age_Bins'] = test_data['Age'].apply(lambda row: get_age_binning(row))
test_data['Age_Bins'].value_counts()
# finding if the person was alone or not
test_data['Alone'] = test_data.apply(lambda row: is_alone(row.SibSp, row.Parch), axis=1)
test_data['Alone'].value_counts()
test_data['Pclass'] = test_data['Pclass'].astype('category')
test_data['SibSp'] = test_data['SibSp'].astype('category')
test_data['Parch'] = test_data['Parch'].astype('category')
test_data['Sex'] = test_data['Sex'].apply(lambda row: 1 if row == 'Male' else 0)
test_data['Alone'] = test_data['Alone'].apply(lambda row: 1 if row == 'Yes' else 0)
test_data.head()
cols_to_encode = ['Pclass', 'SibSp', 'Parch', 'Title', 'Embarked', 'Age_Bins', 'Fare_Type']
test_encoded_data = pd.get_dummies(test_data[cols_to_encode])
test_encoded_data.head()
test_data = pd.concat([test_data, test_encoded_data], axis=1)
# saving the passender id to some dataframe as lateron we need to concat the passenger id and final predictions
passenger_id = test_data['PassengerId']
test_data = test_data.drop(['Pclass', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 
                              'Fare_Type', 'Title', 'Age_Bins', 'PassengerId'], axis=1)
y_test_pred = log_reg.predict(test_data[cols])
test_results = pd.concat([passenger_id, pd.DataFrame(y_test_pred)], axis=1)
test_results.columns = ['PassengerId','Survived']
test_results.to_csv('test_results_titanic.csv', index=None)