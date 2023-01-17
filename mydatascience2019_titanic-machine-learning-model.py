# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # Data Processing
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#import the dataset
df = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df.shape
df_test.shape
df.head()
df_test.head()
df.info()
df_test.info()
df.describe()
a = pd.DataFrame(round(df.isnull().sum().sort_values(ascending = False)*100/df.shape[0],2)).reset_index()
a.rename(columns={"index":"Feature",0:"%"},inplace = True)
b = pd.DataFrame(df.isnull().sum().sort_values(ascending = False).reset_index())
b.rename(columns={"index":"Feature",0:"Actual"},inplace = True)
ab = pd.merge(b,a,how='left',on='Feature')
ab[ab['Actual']>0]
a = pd.DataFrame(round(df_test.isnull().sum().sort_values(ascending = False)*100/df_test.shape[0],2)).reset_index()
a.rename(columns={"index":"Feature",0:"%"},inplace = True)
b = pd.DataFrame(df_test.isnull().sum().sort_values(ascending = False).reset_index())
b.rename(columns={"index":"Feature",0:"Actual"},inplace = True)
ab = pd.merge(b,a,how='left',on='Feature')
ab[ab['Actual']>0]
# Let us take a copy of the dataset before further processing:
df1 = df.copy()
#Let us drop the column Cabin
df1.drop('Cabin',axis=1,inplace = True)
df1.shape
df_test1 = df_test.copy()
#Let us drop the column Cabin
df_test1.drop('Cabin',axis=1,inplace = True)
df_test1.shape
# Let us remove the non-essential columns from the dataset
df1.drop(['PassengerId','Ticket'],axis=1,inplace = True)
df1.head()
df1['Family'] = df1['SibSp']+df1['Parch']
df1.head()
#Let us drop Sibsp and Parch columns from the dataset
df1.drop(['SibSp','Parch'],axis=1,inplace =True)
df1.head()
df1.Family.value_counts()
# Let us group the passengers into two. Those who are travelling soolo are mentioned as 0 and those who are travelling with 
#Partners are marked with 1
df1['Partner']= np.where(df1['Family']>0,1,0)
df1.drop('Family',axis=1,inplace = True)
# from the Title of a person we can categorize a set of people into a single group. Hence let us extract the title from the name
df1['Title']=df1.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
df1.drop('Name',axis=1,inplace = True)
df1.head()
# Let us check the type of titles that we have. If we can categorize those titles in better fashion, it will be better 
df1.Title.value_counts()
# Catgorize the titles:
Title_1 = {
    "Capt":        "Other",
    "Col":         "Other",
    "Major":       "Other",
    "Jonkheer":    "Other",
    "Don":         "Other",
    "Sir" :        "Other",
    "Dr":          "Other",
    "Rev":         "Other",
    "the Countess": "Other",
    "Dona":        "Other",
    "Mme":        "Mrs",  # it means "Mrs",
    "Mlle":       "Miss",  # it means "Miss",
    "Ms":         "Mrs",  # it means "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Other"
}
# Replace the Title with the new titles
df1['Title'] = df1.Title.map(Title_1)
df1.Title.value_counts()
# Now, group the data based on Age,sex and Title
df1_grouped = df1.groupby(['Sex','Pclass', 'Title'])
df1_grouped.Age.median()
# Now apply the grouped median value on the Age whereever it is null
df1.Age = df1_grouped.Age.apply(lambda x: x.fillna(x.median()))

df1.info()
df1.head()
# Embarked
a = df1[df1['Pclass']==1]['Embarked'].value_counts()
a
b = df1[df1['Pclass']==2]['Embarked'].value_counts()
b
c = df1[df1['Pclass']==3]['Embarked'].value_counts()
c
df1['Embarked'] = df1['Embarked'].fillna('S')
df1.info() 
all_numeric = df1.select_dtypes(include=['float64', 'int64'])
cor = all_numeric.corr()
sns.heatmap(cor, cmap="YlGnBu", annot=True)
plt.show()
df_test1['Family'] = df_test1['SibSp']+df_test1['Parch']
df_test1.head()
#Let us drop Sibsp and Parch columns from the dataset
df_test1.drop(['SibSp','Parch','Ticket'],axis=1,inplace =True)
df_test1.Family.value_counts()
# Let us group the passengers into two. Those who are travelling soolo are mentioned as 0 and those who are travelling with 
#Partners are marked with 1
df_test1['Partner']= np.where(df_test1['Family']>0,1,0)
df_test1.drop('Family',axis=1,inplace = True)
# from the Title of a person we can categorize a set of people into a single group. Hence let us extract the title from the name
df_test1['Title']=df_test1.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_test1.drop('Name',axis=1,inplace = True)
df_test1.head()
# Let us check the type of titles that we have. If we can categorize those titles in better fashion, it will be better 
df_test1.Title.value_counts()
# Catgorize the titles:
Title_1 = {
    "Capt":        "Other",
    "Col":         "Other",
    "Major":       "Other",
    "Jonkheer":    "Other",
    "Don":         "Other",
    "Sir" :        "Other",
    "Dr":          "Other",
    "Rev":         "Other",
    "the Countess": "Other",
    "Dona":        "Other",
    "Mme":        "Mrs",  # it means "Mrs",
    "Mlle":       "Miss",  # it means "Miss",
    "Ms":         "Mrs",  # it means "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Other"
}
# Replace the Title with the new titles
df_test1['Title'] = df_test1.Title.map(Title_1)
df_test1.Title.value_counts()
# Now, group the data based on Age,sex and Title
df_test1_grouped = df_test1.groupby(['Sex','Pclass', 'Title'])
df_test1_grouped.Age.median()
# Now apply the grouped median value on the Age whereever it is null
df_test1.Age = df_test1_grouped.Age.apply(lambda x: x.fillna(x.median()))

df_test1.info()
#Since there is only one row which is left blank for fare, let us fill the blank with the median
df_test1['Fare'].fillna((df_test1['Fare'].median()), inplace=True)
df_test1.info()
# Let us see that survival ratio of male and female
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Sex", data=df1)
# Check for Passenger Class
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Pclass", data=df1)
# Check by Passenger title
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Title", data=df1)
df2 = df1[df1['Title']=="Master"]
len(df2)
# Check by children only
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Title", data=df2)
# Check by Partner  only
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Partner", data=df1)
df2 = df1[df1['Partner']==0]
len(df2)
sns.set(style="darkgrid")
sns.countplot(x="Survived",hue="Title", data=df2)
df1.head()
df_test1.head()
#Now convert the male and female to 0 and 1
df_test1.Sex = df1.Sex.map({"male": 0, "female":1})
#Now convert the male and female to 0 and 1
df1.Sex = df1.Sex.map({"male": 0, "female":1})
# Creating a dummy variable for some of the categorical variables and dropping the first one.
embarked_dm = pd.get_dummies(df1[['Embarked']],drop_first = True)
pclass_dm = pd.get_dummies(df1[['Pclass']],drop_first = True)
title_dm = pd.get_dummies(df1[['Title']],drop_first = True)
# Creating a dummy variable for some of the categorical variables and dropping the first one.
embarked_dm_test = pd.get_dummies(df_test1[['Embarked']],drop_first = True)
pclass_dm_test = pd.get_dummies(df_test1[['Pclass']],drop_first = True)
title_dm_test = pd.get_dummies(df_test1[['Title']],drop_first = True)
# Adding the results to the master dataframe
df1 = pd.concat([df1, embarked_dm,pclass_dm,title_dm], axis=1)

# drop the source categorical columns from which we have created dummy variables:
df1.drop(['Embarked','Pclass','Title'],axis = 1,inplace = True)
# Adding the results to the master dataframe
df_test1 = pd.concat([df_test1, embarked_dm_test,pclass_dm_test,title_dm_test], axis=1)

# drop the source categorical columns from which we have created dummy variables:
df_test1.drop(['Embarked','Pclass','Title'],axis = 1,inplace = True)
df1.head()
df_test1.head()
from sklearn.preprocessing import StandardScaler
col = ['Fare', 'Age']
scale = StandardScaler()
df1[col] = scale.fit_transform(df1[col])
df1.head()
col = ['Fare', 'Age']
scale = StandardScaler()
df_test1[col] = scale.fit_transform(df_test1[col])
df_test1.head()
# create X and y for data and target values
X = df1.drop('Survived', axis=1).values
y = df1.Survived.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel.score(X_train,y_train)
test_x = df_test1.drop('PassengerId',axis=1)
predictions = logmodel.predict(test_x)
final_prediction = pd.DataFrame({'PassengerId':df_test1['PassengerId'],'Survived':predictions})
final_prediction.head()
