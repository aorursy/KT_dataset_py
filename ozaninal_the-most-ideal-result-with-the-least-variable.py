# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# # to display all columns:
# pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split, GridSearchCV
#import train and test CSV files
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
#create a copy train
df= train.copy()

#create a combined group of both datasets
combine = [df, test]
#take a look at the training data
df.describe().T
(df.columns)
test.info()
df.isnull().sum()
test.isnull().sum()
print(df['Pclass'].value_counts())
print(df['SibSp'].value_counts())
print(df['Parch'].value_counts())
print(df['Embarked'].value_counts())
print(df['Sex'].value_counts())
print(df['Ticket'].value_counts())
print(df['Cabin'].value_counts())
#draw a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=df);
#draw a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=df);
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=df);
df.groupby('SibSp')['PassengerId'].count()
sns.barplot(x="Parch", y="Survived", data=df)
plt.show()
df.groupby('Parch')['PassengerId'].count()
sns.boxplot(x = df['Fare']);
test.describe(include="all")
df['Parch'] =  df['Parch'].replace([df.loc[df.Parch>1,'Parch'].values], 2)
test['Parch'] =  test['Parch'].replace([df.loc[df.Parch>1,'Parch'].values], 2)
from pandas.api.types import CategoricalDtype 
df['Parch'] = df['Parch'].astype(CategoricalDtype(ordered = True))
test['Parch'] = test['Parch'].astype(CategoricalDtype(ordered = True))
df.groupby(['Parch'])['PassengerId'].count()
test.groupby(['SibSp'])['PassengerId'].count()
df['SibSp'] =  df['SibSp'].replace([df.loc[df.SibSp>1,'SibSp'].values], 2)
test['SibSp'] =  test['SibSp'].replace([df.loc[df.SibSp>1,'SibSp'].values], 2)
df['SibSp'] = df['SibSp'].astype(CategoricalDtype(ordered = True))
test['SibSp'] = test['SibSp'].astype(CategoricalDtype(ordered = True))
sns.barplot(x="SibSp", y="Survived", data=df);
df[df.Cabin.notnull()].groupby('Pclass')['PassengerId'].count()
df = df.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
#we can also drop the Ticket feature since it's unlikely to yield any useful information
df = df.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

df['Embarked'].fillna(df.Embarked.mode()[0], inplace = True)

# It looks like there is a problem in Fare max data. Visualize with boxplot.
sns.boxplot(x = df['Fare']);
Q1 = train['Fare'].quantile(0.25)
Q3 = train['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1- 1.5*IQR
print(lower_limit)

upper_limit = Q3 + 1.5*IQR
print(upper_limit)

print(df[df.Fare<upper_limit]['Fare'].describe())
print(df[df.Fare>upper_limit]['Fare'].describe())
print(df[df.Fare<150]['Fare'].describe())
print(df[df.Fare>150]['Fare'].describe())
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = df[df.Pclass==pclass]['Fare'].median()

        
# I replace values greater than 150 with the median of values greater than 150.

df['Fare'] = df['Fare'].replace(df[df.Fare>150]['Fare'], df[df.Fare>150]['Fare'].median())
test['Fare'] = test['Fare'].replace(test[test.Fare>150]['Fare'], test[test.Fare>150]['Fare'].median())

# df['FareBand'] = pd.qcut(df['Fare'], 4, labels = [1, 2, 3,4])
# test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3,4])


binss = [-1,8, 31, 100,np.inf]
labelss = [ 1,2,3,4]
df['FareBand'] = pd.cut(df["Fare"], binss, labels = labelss)
test['FareBand'] = pd.cut(test["Fare"], binss, labels = labelss)


df['FareBand'] = df['FareBand'].astype(CategoricalDtype(ordered = True))
test['FareBand'] = test['FareBand'].astype(CategoricalDtype(ordered = True))

sns.barplot(x="FareBand", y="Survived", data=df)
plt.show()
df.groupby('FareBand')['Fare'].mean()
#extract a title for each Name in the train and test datasets
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) 

pd.crosstab(df['Title'], df['Sex'])
#replace various titles with more common names

#for df
df['Title'] = df['Title'].replace(['Lady', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Jonkheer', 
                                                 'Dona','Countess', 'Lady', 'Master',
                                                 'Sir'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Ms'],'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
#for test    
test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col', 'Don',
                                                 'Dr', 'Major', 'Rev', 'Jonkheer', 
                                                 'Dona','Countess', 'Lady', 'Master',
                                                 'Sir'], 'Rare')
test['Title'] =test['Title'].replace(['Mlle', 'Ms'],'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')


df.Title = pd.Categorical(df.Title)
test.Title = pd.Categorical(test.Title)    

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#I tried to choose the ones that gave the most optimum result.

age_guess = pd.DataFrame(df.groupby(['Title','SibSp','Pclass','Parch','FareBand','Embarked'], as_index=False)['Age'].median())
df_age_new = df.merge(age_guess, on =['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], how='inner')
df_age_new['AgeErrors'] = (df_age_new['Age_x']-df_age_new['Age_y']).abs()
df_age_new['AgeErrors'].describe()

null_Age = df.drop(df[df.Age.notnull()].index, axis=0)
notnull_Age = df.drop(df[df.Age.isnull()].index, axis=0)

age_guess = pd.DataFrame(df.groupby(['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], as_index=False)['Age'].median())
null_Age = null_Age.merge(age_guess, on =['Title','SibSp','Pclass', 'Parch','FareBand','Embarked'], how='inner')

null_Age['Age_y']= null_Age['Age_y'].fillna(null_Age['Age_y'].median())

null_Age= null_Age.drop('Age_x', axis=1 ).rename(columns= {"Age_y": "Age"})
df = pd.concat([null_Age,notnull_Age], axis=0, ignore_index = True)
null_Age = test.drop(test[test.Age.notnull()].index, axis=0)
notnull_Age = test.drop(test[test.Age.isnull()].index, axis=0)

age_guess = pd.DataFrame(test.groupby(['Title','SibSp','Pclass','Parch' ,'FareBand','Embarked'], as_index=False)['Age'].median())
null_Age = null_Age.merge(age_guess, on =['Title','SibSp','Pclass', 'Parch','FareBand','Embarked'], how='inner')

null_Age['Age_y']= null_Age['Age_y'].fillna(null_Age['Age_y'].median())

null_Age= null_Age.drop('Age_x', axis=1 ).rename(columns= {"Age_y": "Age"})
test = pd.concat([null_Age,notnull_Age], axis=0, ignore_index = True)
#sort the ages into logical categories

bins = [0, 5,  64,  np.inf]
labels = [3,4,5]
df['AgeGroup'] = pd.cut(df["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)


df['AgeGroup'] = df['AgeGroup'].astype(CategoricalDtype(ordered = True))
test['AgeGroup'] = test['AgeGroup'].astype(CategoricalDtype(ordered = True))

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=df)
plt.show()
#drop the name feature since it contains no more useful information.
df = df.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
df = df.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
df = pd.get_dummies(df, columns = ["Pclass"], prefix = ["Pclass"], drop_first= True)
test = pd.get_dummies(test, columns = ["Pclass"], prefix = ["Pclass"], drop_first= True)
df = pd.get_dummies(df, columns = ["Embarked"], prefix = ["Embarked"], drop_first= True)
test = pd.get_dummies(test, columns = ["Embarked"], prefix = ["Embarked"], drop_first= True)
df = pd.get_dummies(df, columns = ["Title"], prefix = ["Title"], drop_first= True)
test = pd.get_dummies(test, columns = ["Title"], prefix = ["Title"], drop_first= True)
df = pd.get_dummies(df, columns = ["Sex"], prefix = ["Sex"], drop_first= True)
test = pd.get_dummies(test, columns = ["Sex"], prefix = ["Sex"], drop_first= True)
from sklearn.model_selection import train_test_split

predictors = df.drop(['Survived', 'PassengerId'], axis=1)
target = df["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Logistic Regression82.74
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines82.23,
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC 82.23, 62.44
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
#MLPClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

scaler.fit(x_train)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_val)

mlpc = MLPClassifier().fit(X_train_scaled, y_train)
y_pred = mlpc.predict(X_test_scaled)
acc_mlpc = round(accuracy_score(y_val, y_pred)*100, 2)
print(acc_mlpc)
models = pd.DataFrame({
    'Model': ['Support Vector Machines',  'Logistic Regression', 
               'Linear SVC',  'MLPClassifier' ,'Stochastic Gradient Descent'],
    'Score': [acc_svc, acc_logreg, 
              acc_linear_svc,acc_mlpc,
              acc_sgd]})
models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = mlpc.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission33.csv', index=False)
