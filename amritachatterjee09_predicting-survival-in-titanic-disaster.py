#Data Analysis & Data wrangling

import numpy as np

import pandas as pd

import missingno as mn

from collections import Counter



#Visualization

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

%matplotlib inline



# setting up plot style 

sns.set_context("paper")

style.use('fivethirtyeight')



# machine learning

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
import warnings

warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the input files

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_train.head()
titanic_test.head()
#Combining the two dataframes for later use



combined=titanic_train.append(titanic_test)
# Database dimension

print("Database dimension : ")

print("Database dimension - titanic_train     :",titanic_train.shape)

print("Database dimension - titanic_test      :",titanic_test.shape)

print("Database dimension - combined          :",combined.shape)



print('\n')



#Database size

print("Database size : ")

print("Database size - titanic_train          :",titanic_train.size)

print("Database size - titanic_test           :",titanic_test.size)

print("Database size - combined               :",combined.size)
#Database column types

print("Titanic Train Dataset Info : ")

print(titanic_train.info())

print("_"* 40)

print("Titanic Test Dataset Info : ")

print(titanic_test.info())
# Checking the numerical variables in train and test data set

titanic_train.describe()
mn.bar(titanic_train)
#Column wise null values in train data set 

#titanic_train.isnull().sum()

null_train_perc = pd.DataFrame((titanic_train.isnull().sum())*100/titanic_train.shape[0]).reset_index()

null_train_perc.columns = ['Column Name', 'Null Values Percentage']

null_train_value = pd.DataFrame(titanic_train.isnull().sum()).reset_index()

null_train_value.columns = ['Column Name', 'Null Values']

null_train = pd.merge(null_train_value, null_train_perc, on='Column Name')

null_train
mn.bar(titanic_test)
#column wise null values in test data set

null_test_perc = pd.DataFrame((titanic_test.isnull().sum())*100/titanic_test.shape[0]).reset_index()

null_test_perc.columns = ['Column Name', 'Null Values Percentage']

null_test_value = pd.DataFrame(titanic_test.isnull().sum()).reset_index()

null_test_value.columns = ['Column Name', 'Null Values']

null_test = pd.merge(null_test_value, null_test_perc, on='Column Name')

null_test
# checking the correlation among the numeric variables

plt.figure(figsize = (8,6))

ax= sns.heatmap(titanic_train.corr(), annot = True, cmap="RdYlGn",linewidth =1)

plt.show()
sns.countplot(titanic_train['Survived'], palette = 'husl')

plt.show()
titanic_train['Survived'].value_counts(normalize=True)
ax = sns.FacetGrid(titanic_train, col='Survived',height = 6, aspect =0.5)

ax.map(sns.distplot, "Age")

plt.show()
ax = sns.kdeplot(titanic_train["Age"][(titanic_train["Survived"] == 0) & (titanic_train["Age"].notnull())], color="Red", shade = True)

ax = sns.kdeplot(titanic_train["Age"][(titanic_train["Survived"] == 1) & (titanic_train["Age"].notnull())], ax =ax, color="Green", shade= True)

ax.set_xlabel("Age")

ax.set_ylabel("Frequency")

ax = ax.legend(["Not Survived","Survived"])
# Overall age distribution of combined test & train database

ax = sns.distplot(combined["Age"], color="purple", label="Skewness : %.2f"%(combined["Age"].skew()))

ax = ax.legend(loc="best")

plt.show()
# Overall Fare distribution of combined test & train database

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

ax = sns.distplot(combined["Fare"], color="green", label="Skewness : %.2f"%(combined["Fare"].skew()))

ax = ax.legend(loc="best")

plt.subplot(1,2,2)

sns.boxplot(combined["Fare"],color="green")

plt.show()
ax = sns.FacetGrid(titanic_train, col='Survived',height = 6, aspect =0.5)

ax.map(sns.distplot, "Fare")

plt.show()
ax = sns.kdeplot(combined["Fare"][(combined["Survived"] == 0) & (combined["Fare"].notnull())], color="Red", shade = True)

ax = sns.kdeplot(combined["Fare"][(combined["Survived"] == 1) & (combined["Fare"].notnull())], ax =ax, color="Green", shade= True)

ax.set_xlabel("Age")

ax.set_ylabel("Frequency")

ax = ax.legend(["Not Survived","Survived"])
plt.figure(figsize = (10,6))

sns.barplot(x="Parch", y="Survived",data = titanic_train,palette="Set2")

plt.ylabel("Survival Probability")

plt.show()
plt.figure(figsize = (10,6))

sns.barplot(x="SibSp", y="Survived",data = titanic_train,palette="husl")

plt.ylabel("Survival Probability")

plt.show()
plt.figure(figsize = (15,6))

plt.subplot(1,2,1)

sns.barplot(x="Sex", y="Survived",data = titanic_train,palette="Set2")

plt.ylabel("Survival Probability")

plt.subplot(1,2,2)

sns.countplot("Sex",data = titanic_train,palette="Set2")

plt.show()
titanic_train[["Sex","Survived"]].groupby('Sex').agg({"mean","count"})
plt.figure(figsize = (18,6))

plt.subplot(1,3,1)

sns.barplot(x="Pclass", y="Survived",data = titanic_train,palette="muted")

plt.ylabel("Survival Probability")

plt.subplot(1,3,2)

sns.countplot("Pclass",data = titanic_train,palette="muted")

plt.subplot(1,3,3)

sns.barplot(x="Pclass", y="Survived",data = titanic_train,hue = "Sex",palette="muted")

plt.ylabel("Survival Probability")

plt.show()
plt.figure(figsize = (18,6))

plt.subplot(1,3,1)

sns.barplot(x="Embarked", y="Survived",data = titanic_train,palette="Accent")

plt.ylabel("Survival Probability")

plt.subplot(1,3,2)

sns.countplot("Embarked",data = titanic_train,palette="Accent")

plt.subplot(1,3,3)

sns.barplot(x="Embarked", y="Survived",data = titanic_train,hue = "Pclass",palette="Accent")

plt.ylabel("Survival Probability")

plt.show()
ax= sns.FacetGrid(data = titanic_train, row = 'Sex', col = 'Pclass', hue = 'Survived',palette = 'husl',height = 4, aspect = 1.4)

ax.map(sns.kdeplot, 'Age', alpha = .75, shade = True)

plt.legend()
##Let us analyze the rows which is missing Fare information

display(combined[combined.Fare.isnull()])
##Let us get fare per person

for df in [titanic_train, titanic_test, combined]:

    df['PeopleInTicket']=df['Ticket'].map(combined['Ticket'].value_counts()) # Getting the unique count of tickets

    df['FarePerPerson']=df['Fare']/df['PeopleInTicket'] 



##Just take the mean fare for the PORT S and the Pclass & fill it. Remember to consider FarePerPerson and not Fare

print('Mean fare for this category: ', titanic_train[(titanic_train.Embarked=='S') & (titanic_train.Pclass==3)]['FarePerPerson'].mean())
#Imputing the fare based on class and embarked location mean

titanic_test.loc[titanic_test.Fare.isnull(), ['Fare','FarePerPerson']] = round(titanic_train[(titanic_train.Embarked=='S')& (titanic_train.Pclass==3)\

                                                                          & (titanic_train.PeopleInTicket==1)]['Fare'].mean(),1)

display(titanic_test[titanic_test.Fare.isnull()]) # As the null value in Fare was only in Test dataset
display(combined[combined.Embarked.isnull()])
##Groupby Embarked and check some statistics

titanic_train[titanic_train.Pclass==1].groupby(['Embarked',"Pclass"]).agg({'FarePerPerson': 'mean', 'Fare': 'mean', 'PassengerId': 'count'})
# Updating the Embarked location for the two missing values

titanic_train.loc[titanic_train.PassengerId==62,'Embarked']="C"

titanic_train.loc[titanic_train.PassengerId==830,'Embarked']="C"

display(titanic_train[titanic_train.Embarked.isnull()])
titanic_train['Title'], titanic_test['Title'] = [df.Name.str.extract(' ([A-Za-z]+)\.', expand=False) for df in [titanic_train, titanic_test]]
#Extracting the statistics of Title on Train dataset

titanic_train.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count'])
TitleDict = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \

             "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \

             "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \

             "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}
titanic_train['Title'], titanic_test['Title'] = [df.Title.map(TitleDict) for df in [titanic_train, titanic_test]]



##Let us now reprint the groups

titanic_train.groupby(['Title', 'Pclass'])['Age'].agg(['mean', 'count'])
# Checking for Master Title with Age missing

display(combined[(combined.Age.isnull()) & (combined.Name.str.contains('Master'))])
print("Average age for Masters in Pclass 3 : ", round(titanic_train[titanic_train.Name.str.contains('Master')]['Age'].mean(),2))

print("Maximum age for Masters in Pclass 3 : ", round(combined[combined.Name.str.contains('Master')]['Age'].max(),2))
#Assigning the max value to the age of passenger with title Master and travelling alone

titanic_test.loc[titanic_test.PassengerId==1231,'Age']=14
for df in [titanic_train, titanic_test]:

    df.loc[(df.Title=='Miss') & (df.Parch!=0) & (df.PeopleInTicket>1), 'Title']="FemaleChild"



# Extracting the statistics

print(titanic_train.groupby(['Pclass','Sex','Title'])['Age'].agg({'mean', 'median', 'count'}))

print("_"*60)

print(titanic_test.groupby(['Pclass','Sex','Title'])['Age'].agg({'mean', 'median', 'count'}))

    
# Checking female child with missing age



display(titanic_train[(titanic_train.Age.isnull()) & (titanic_train.Title=='FemaleChild')])

display(titanic_test[(titanic_test.Age.isnull()) & (titanic_test.Title=='FemaleChild')])
# Creating a lookup table to fill the missing age values

grp = titanic_train.groupby(['Pclass','Sex','Title'])['Age'].mean().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

grp
# Upfating the missing age based on above lookup table

def fill_age(x):

    return grp[(grp.Pclass==x.Pclass)&(grp.Sex==x.Sex)&(grp.Title==x.Title)]['Age'].values[0]

titanic_train['Age'], titanic_test['Age'] = [df.apply(lambda x: fill_age(x) if np.isnan(x['Age']) else x['Age'], axis=1) for df in [titanic_train, titanic_test]]
# Checking to see if any null value exists

print(titanic_train.Age.isnull().sum())

print("_"*50)

print(titanic_test.Age.isnull().sum())
#Function to identify outliers

def outliers(df, n, features):

    outlier_indices = []

    for col in features:

        Q1 = np.percentile(df[col], 25) # First quartile range

        Q3 = np.percentile(df[col],75) # Third quartile range

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers  
# Extracting the outliers IDs

Outliers_id = outliers(titanic_train, 2, ["Age", "SibSp", "Parch", "Fare"])

print(Outliers_id,'\n')

print(titanic_train.loc[Outliers_id])
#Dropping the outliers

titanic_train = titanic_train.drop(Outliers_id, axis = 0).reset_index(drop=True)

titanic_train.shape
plt.figure(figsize = (12,6))

plt.subplot(1,2,1)

sns.countplot(titanic_train['Title'], palette = 'Set2')

plt.subplot(1,2,2)

sns.barplot(x= "Title",y = "Survived", data = titanic_train, palette = "Set2")

plt.ylabel("Survival Probability")

plt.show()
# Checking if test dataset has any null values for Title

display(titanic_test[(titanic_test.Title.isnull())])
titanic_test.loc[titanic_test.PassengerId==1306,'Title']="Royalty"
for dataset in [titanic_train,titanic_test]:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

titanic_train.head(3)    
for dataset in [titanic_train,titanic_test]:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

titanic_train.head(3)  
for dataset in [titanic_train,titanic_test]:

    dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])

titanic_train["Cabin"].value_counts()    
plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.countplot(titanic_train['Cabin'], palette = 'husl',order=['A','B','C','D','E','F','G','T','X'])

plt.subplot(1,2,2)

sns.barplot(x= "Cabin",y = "Survived", data = titanic_train, palette = "husl",order=['A','B','C','D','E','F','G','T','X'])

plt.ylabel("Survival Probability")

plt.show()
for dataset in [titanic_train,titanic_test]:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']=4
# plotting the data based on new age classification

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.countplot(titanic_train['Age'], palette = 'husl')

plt.subplot(1,2,2)

sns.barplot(x= "Age",y = "Survived", data = titanic_train, palette = "husl")

plt.ylabel("Survival Probability")

plt.show()
for dataset in [titanic_train,titanic_test]:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)    
# plotting the data based on new age classification

plt.figure(figsize = (12,5))

plt.subplot(1,2,1)

sns.countplot(titanic_train['Fare'], palette = 'husl')

plt.subplot(1,2,2)

sns.barplot(x= "Fare",y = "Survived", data = titanic_train, palette = "husl")

plt.ylabel("Survival Probability")

plt.show()
label = LabelEncoder()

for dataset in [titanic_train,titanic_test]:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['Cabin_Code'] = label.fit_transform(dataset['Cabin'])

#After Label Encoding

display(titanic_train.head(3))

display(titanic_test.head(3))
Features = ["Survived","Pclass","Age","Fare","FamilySize","Sex_Code","Embarked_Code","Title_Code","Cabin_Code"]

Features_test = ["Pclass","Age","Fare","FamilySize","Sex_Code","Embarked_Code","Title_Code","Cabin_Code"]

train_data = titanic_train[Features]

train_data.head()
# Checking the correlation of the features

plt.figure(figsize = (8,6))

ax= sns.heatmap(train_data.corr(), annot = True, cmap="RdYlGn",linewidth =1)

plt.show()
y_train = train_data["Survived"]

X_train = train_data.drop(['Survived'], axis=1)

X_test = titanic_test[Features_test]

print(X_train.shape, y_train.shape, X_test.shape)
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

print ("Accuracy of the Logistic Regression model : ",acc_log)
#Feature importance

coeff = pd.DataFrame(X_train.columns)

coeff.columns = ['Features_test']

coeff["Correlation"] = pd.Series(logreg.coef_[0])

coeff.sort_values(by='Correlation', ascending=False)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100, random_state=22)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

print ("Accuracy of the Random Forest model : ",acc_random_forest)
feature_imp = pd.Series(random_forest.feature_importances_,index=Features_test).sort_values(ascending=False)

feature_imp
sns.barplot(feature_imp.values,feature_imp.index,palette='dark')
submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": y_pred})
submission_titanic.to_csv("submission.csv",index=False)