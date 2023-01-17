#Data Analysis libraries
import numpy as np
import pandas as pd
#Data Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Acquiring the data
titanic_train = pd.read_csv("../input/train.csv")
titanic_test = pd.read_csv("../input/test.csv")

titanic_train.info()
print("-"*30)
titanic_test.info()
titanic_train.head()
titanic_train.describe()
titanic_test.describe()
plt.figure(figsize=(10,6))
sns.heatmap(titanic_train.isnull(), yticklabels= False, cbar = False, cmap='viridis' )
#Missing values in test set
titanic_test.isnull().sum()
#Survival of Passengers
sns.set_style("whitegrid")
sns.countplot("Survived", data= titanic_train)
plt.figure(figsize = (8,6))
sns.countplot("Survived", data = titanic_train, hue = "Sex", palette = "coolwarm")
#Distribution of Age
sns.distplot(titanic_train["Age"].dropna(), kde = False, bins = 30, color= "darkgreen")
#Relationship between Age, Sex and Survival of the Passengers
s = sns.FacetGrid(titanic_train, row = "Sex", col = "Survived", margin_titles= True)
s.map(plt.hist, "Age")
#Relationship between the Age and the Fare
sns.jointplot(x = "Age", y = "Fare", data = titanic_train, kind = 'scatter', stat_func= None, dropna=True)

#Correlation between Survival of Gender, Age, and Fare 

g = sns.FacetGrid(data = titanic_train, col = "Sex", hue = "Survived" )
g.map(plt.scatter, "Age", "Fare", alpha = 0.5)
g.add_legend()
#Survival and embarked station
sns.countplot("Embarked", data = titanic_train, hue= "Survived")
# Survival and Pclass
sns.countplot("Survived", data = titanic_train, hue = "Pclass")
#correlation between attributes

corr = titanic_train.corr()

plt.figure(figsize = (10,8))
sns.heatmap(corr, cmap = 'YlGnBu', annot = True)
#Embarked missing data

titanic_train[titanic_train['Embarked'].isnull()]
plt.figure(figsize = (10,10))
sns.boxplot(x = 'Embarked', y = 'Fare', data = titanic_train, hue = "Pclass")
titanic_train['Embarked']= titanic_train['Embarked'].fillna("C")

#Filling the missing Age Values Training data

plt.figure(figsize = (10,14))
sns.boxplot(x = "Pclass", y = "Age", data = titanic_train)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
        
    else:
        return Age
titanic_train['Age'] = titanic_train[['Age', 'Pclass']].apply(impute_age, axis = 1)
plt.figure(figsize = (10,10))
sns.boxplot(x = "Pclass", y = "Age", data = titanic_test)
def impute_agetest(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 42
        elif Pclass == 2:
            return 36
        else:
            return 24
        
    else:
        return Age
    
  
titanic_test['Age'] = titanic_test[['Age', 'Pclass']].apply(impute_agetest, axis = 1)  

titanic_train.drop('Cabin', axis = 1, inplace=True)
titanic_test.drop('Cabin', axis = 1, inplace = True)
titanic_train.head()
titanic_test[titanic_test['Fare'].isnull()]
#Checking distribution of Fare in Pclass = 3

print(titanic_test[titanic_test['Pclass']==3]['Fare'].median())

titanic_test['Fare'] = titanic_test['Fare'].fillna(8)
#Train dataset

sex = pd.get_dummies(titanic_train['Sex'], drop_first=True)
embark = pd.get_dummies(titanic_train['Embarked'], drop_first=True)

titanic_train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace=True)
titanic_train = pd.concat([titanic_train, sex, embark], axis = 1)

titanic_train.head()
#Test Dataset
sex = pd.get_dummies(titanic_test['Sex'], drop_first=True)
embark = pd.get_dummies(titanic_test['Embarked'], drop_first=True)

titanic_test.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace=True)
titanic_test = pd.concat([titanic_test, sex, embark], axis = 1)

titanic_test.head()
#Defining the training and testng datasets

X_train = titanic_train.drop(['PassengerId', 'Survived'], axis = 1)
y_train = titanic_train['Survived']
X_test = titanic_test.drop('PassengerId', axis = 1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
submission = pd.DataFrame({
    'PassengerId':titanic_test['PassengerId'],
    'Survived': predictions
})
submission.to_csv("Titanic.csv", index = False)
