import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
# loading the dataset

df_train = pd.read_csv('../input/titanic/train.csv')



df_train.head()
# checking the data types and info

df_train.info()
# shape of the data frame

df_train.shape
# missing value percentage

round(100 * (df_train.isna().sum(axis=0)/len(df_train.index)),2)
# dropping the cabin column as 77% of the data are missing

df_train.drop(columns = ['Cabin'],inplace = True)
df_train.info()
# converting the data type of the columns

df_train['Survived'] = df_train['Survived'].astype('category')

df_train['Pclass'] = df_train['Pclass'].astype('category')

df_train['Sex'] = df_train['Sex'].astype('category')

df_train['Embarked'] = df_train['Embarked'].astype('category')
# printing the list of numerical and non-numerical columns

numerical_columns = df_train.select_dtypes(include = 'number').columns.tolist()

print("Numerical Columns")

print(numerical_columns,end="\n")

categorical_columns = df_train.select_dtypes(exclude = 'number').columns.tolist()

print("Categorical Columns")

print(categorical_columns,end="\n")
# check for duplicate records

any(df_train.duplicated(keep='first'))
# imputing the age with mean values

df_train.Age.fillna(df_train.Age.mean(),inplace = True)
# creating 'Age Groups' category column from Age column

def age_group(x):

    if (x > 0) and (x <=2):

        return 'babies'

    elif (x > 2) and (x <=16):

        return 'children'

    elif (x > 16) and (x <=30):

        return 'yound adults'

    elif (x > 30) and (x <=45):

        return 'middle-aged adults'

    elif (x > 45) and (x <=80):

        return 'old'

    else:

        return 'unknown'



df_train['Age Groups'] = df_train['Age'].apply(age_group)
# plotting the passengers age

print(df_train['Age'].describe())

sns.boxplot(y = df_train['Age'])
# plotting the histogram to see the distribution

sns.distplot(df_train['Age'].dropna())

plt.show()
# plotting count plot to know the number of male and female

sns.countplot(df_train['Sex'])

plt.show()
# filtering only survived passengers in the dataset

df_survived = df_train[df_train['Survived']==1]
# plotting count plot to know the number of male and female in the survived passengers

sns.countplot(df_survived['Sex'])

plt.show()
df_train['Survived'] = df_train['Survived'].astype(int)

by_pclass_segment_group = df_train.pivot_table(values='Survived',index='Pclass',aggfunc='mean')

by_pclass_segment_group.reset_index(inplace=True)

by_pclass_segment_group['Survived'] = 100*by_pclass_segment_group['Survived']

plt.figure(figsize=(8,4))

sns.barplot(x='Pclass',y='Survived', data=by_pclass_segment_group)

plt.xlabel("Pclass")

plt.ylabel("Percentage of passenger survived")

plt.title("% of passenger survived vs Class of the passenger")

plt.show()
by_gender_pclass_segment_group = df_train.pivot_table(values='Survived',index=['Sex','Pclass'])

by_gender_pclass_segment_group.reset_index(inplace=True)

by_gender_pclass_segment_group['Survived'] = 100*by_gender_pclass_segment_group['Survived']

plt.figure(figsize=(8,4))

sns.barplot(x='Pclass',y='Survived',hue='Sex', data=by_gender_pclass_segment_group)

plt.xlabel("Pclass")

plt.ylabel("Percentage of passenger survived")

plt.show()
# bin by age group and analyse which age group survived

plt.figure(figsize=(10,6))

sns.boxplot(x = 'Survived', y = 'Age', data = df_train)

plt.title("Age v/s Survived")

plt.show()
plt.figure(figsize=(10,6))

sns.barplot(y = 'Survived', x = 'Age Groups', data = df_train)

plt.title("Age Groups v/s Survived")

plt.show()
# treating categorical variables

df_train.Sex = df_train.Sex.apply(lambda x : 0 if x == 'male' else 1)

# adding parents and siblings data to get family members data

df_train['Family Members'] = df_train.SibSp + df_train.Parch



#creating dummy variables for passenger class, age groups and embarked variables

pClass = pd.get_dummies(df_train.Pclass, drop_first = True)

embarked = pd.get_dummies(df_train.Embarked, drop_first = True)

age_groups = pd.get_dummies(df_train['Age Groups'], drop_first = True)



df_train = pd.concat([df_train,age_groups,pClass,embarked], axis=1)
# creating X and y training variables

y_train = df_train.pop('Survived')

X_train = df_train.drop(columns = ['PassengerId','Pclass','Name','Ticket','Embarked','Age Groups'])
X_train.head()
# feature scaling of numeric variables

scaler = MinMaxScaler()

cols_to_scale = ['Age','Fare','SibSp','Parch','Family Members']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
# defining logistic regression model and fitting the model with training data set

lr = LogisticRegression()

lr.fit(X_train,y_train)
# predicting the result

y_train_pred = lr.predict(X_train)
# using confusion matrix to find the accuracy

confusion_matrix(y_train,y_train_pred)
round(100* ((480 + 238) / (480+69+104+238)),2)
# now reading the test dataset

df_test = pd.read_csv('../input/titanic/test.csv')

df_test.head()
df_test.info()
# dropping 'cabin' column

df_test.drop(columns = 'Cabin',inplace = True)
# imputting the age and fare values with its mean

df_test.Age.fillna(df_test.Age.mean(),inplace = True)

df_test.Fare.fillna(df_test.Fare.mean(),inplace = True)
# creating the age groups column

df_test['Age Groups'] = df_test['Age'].apply(age_group)
#creating dummy variables for categorical columns

df_test.Sex = df_test.Sex.apply(lambda x : 0 if x == 'male' else 1)

df_test['Family Members'] = df_test.SibSp + df_test.Parch



pClass = pd.get_dummies(df_test.Pclass, drop_first = True)



embarked = pd.get_dummies(df_test.Embarked, drop_first = True)



age_groups = pd.get_dummies(df_test['Age Groups'], drop_first = True)

df_test = pd.concat([df_test,pClass,age_groups,embarked], axis=1)
# creating X test variable

X_test = df_test.drop(columns = ['PassengerId','Pclass','Name','Ticket','Embarked','Age Groups'])
# feature scaling the numerical columns

cols_to_scale = ['Age','Fare','SibSp','Parch','Family Members']

X_test[cols_to_scale] = scaler.fit_transform(X_test[cols_to_scale])
# predicting the test result

y_test_pred = lr.predict(X_test)
#creating the gender submission csv file for submission to kaggle

gender_submission_df = pd.concat([df_test['PassengerId'],pd.Series(y_test_pred)],axis=1)

gender_submission_df.rename(columns={0:'Survived'},inplace = True)

gender_submission_df.to_csv("gender_submissions.csv",index = False)