# Import required libraries
import pandas as pd 
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')
test_df.head()
train_df.head()
print(train_df.columns.values)
train_df.info()
train_df.describe()
train_df.shape
# All numeric (float and int) variables in the dataset
train_df_numeric = train_df.select_dtypes(include=['float64', 'int64'])
train_df_numeric.head()
# Correlation matrix
cor = train_df_numeric.corr()
cor
# Figure size
plt.figure(figsize=(16,8))

# Heatmap
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()
print(train_df.info())
full_dataset = [train_df, test_df]
# Sex is a binary value column.
# Hence making female as 0 and male as 1
sex_dict = {"female":0, "male":1}
for data_df in full_dataset:
    data_df['Sex'] = data_df['Sex'].apply(lambda x:sex_dict[x])
for data_df in full_dataset:
    data_df['Title']  = data_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data_df['Title'] = data_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Royalty')

    data_df['Title'] = data_df['Title'].replace('Mlle', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Mme', 'Mrs')

    
title_dict = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty": 5}
for data_df in full_dataset: 
    data_df['Title'] = data_df['Title'].apply(lambda x: title_dict[x])
    data_df['Title'] = data_df['Title'].fillna(0)
def AgeGroup(age):
    if(age <= 16):
        return 0 
    elif age > 16 and age <= 32:
        return 1
    elif age>32 and age <=48:
        return 2 
    elif age>48 and age <= 64:
        return 3
    else:
        return 4
    
for data_df in full_dataset:
    age_avg = data_df['Age'].mean()
    age_std = data_df['Age'].std()
    age_null_count = data_df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    data_df['Age'][np.isnan(data_df['Age'])] = age_null_random_list
    data_df['Age'] = data_df['Age'].astype(int)
    data_df['AgeGoup'] = data_df['Age'].apply(AgeGroup)
def Alone(familysize):
    if familysize ==1:
        return 1 
    else:
        return 0

for data_df in full_dataset:
    data_df['Family_size'] = data_df['SibSp'] + data_df['Parch'] + 1
    data_df['IsAlone'] = data_df['Family_size'].apply(Alone)
embarked= {'S': 0, 'C': 1, 'Q': 2}
for data_df in full_dataset:
    data_df['Embarked'] = data_df['Embarked'].fillna('S')
    data_df['Embarked'] = data_df['Embarked'].apply(lambda x: embarked[x])
def Cabin(cabin):
    if type(cabin) == str:
        return 1
    else:
        return 0
    
for data_df in full_dataset:
    data_df['HasCabin'] = data_df['Cabin'].apply(Cabin)
def FareGroup(fare):
    if fare <= 7.91:
        return 0;
    elif fare >7.91 and fare <=14.454:
        return 1
    elif fare >14.454 and fare <=31:
        return 2
    else:
        return 3

for data_df in full_dataset:
    data_df['Fare'] = data_df['Fare'].fillna(data_df['Fare'].median())
    data_df['FareGroup'] = data_df['Fare'].apply(FareGroup)
train_df.head()
for data_df in full_dataset:
    data_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'],axis=1, inplace = True)
full_dataset
train_df.head()
# For training and testing purposes, split train dataset into two df
# Importing test_train_split from sklearn library
from sklearn.model_selection import train_test_split
# Putting feature variable to X
X = train_df.drop('Survived',axis=1)

# Putting response variable to y
y = train_df['Survived']

# Splitting the data into train and test
x_train_training, x_train_testing, y_train_training, y_train_testing = train_test_split(X, y, test_size=0.30, random_state=101)
#x_train = train_df.drop("Survived", axis=1)
#y_train = train_df["Survived"]
logreg = LogisticRegression()
logreg.fit(x_train_training, y_train_training)
y_pred = logreg.predict(test_df)
acc_log = round(logreg.score(x_train_training, y_train_training) * 100, 2)
acc_log
# Importing random forest classifier from sklearn library
from sklearn.ensemble import RandomForestClassifier

# Running the random forest with default parameters.
rfc = RandomForestClassifier()
# fit
rfc.fit(x_train_training,y_train_training)
# Making predictions
acc_log = round(rfc.score(x_train_training, y_train_training) * 100, 2)
acc_log
# Making predictions
predictions = rfc.predict(x_train_training)
print( np.mean(predictions == y_train_training))
# Making predictions
predictions = rfc.predict(x_train_testing)
print( np.mean(predictions == y_train_testing))
y_pred = rfc.predict(test_df)
#test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
#submission = pd.DataFrame({
#        "PassengerId": test_data["PassengerId"],
#        "Survived": y_pred
#    })
#submission.to_csv('my_submission.csv', index=False)
#print("Your submission was successfully saved!")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
