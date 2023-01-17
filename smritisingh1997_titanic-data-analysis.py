#Required packages

import numpy as np

import pandas as pd

from sklearn import preprocessing

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_rows=None

pd.options.display.max_columns=None
titanic_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
submit_df = pd.DataFrame()

submit_df['PassengerId'] = test_df['PassengerId']
titanic_df.head()
test_df.head()
titanic_df.shape
test_df.shape
titanic_df.info()
test_df.info()
titanic_df.describe()
test_df.describe()
titanic_df.isnull().sum()
test_df.isnull().sum()
titanic_df.dtypes
test_df.dtypes
titanic_df.skew(axis = 0, skipna = True)
test_df.skew(axis = 0, skipna = True)
fig, ax =plt.subplots(2,2, figsize=(10, 8))

sns.distplot(titanic_df["Fare"], ax=ax[0][0])

sns.distplot(titanic_df["Age"], ax=ax[0][1])

sns.distplot(titanic_df["SibSp"], ax=ax[1][0])

sns.distplot(titanic_df["Parch"], kde_kws={'bw':0.1}, ax=ax[1][1])

fig.tight_layout(pad=3.0)

# fig.show()
fig, ax =plt.subplots(2,2, figsize=(10, 8))

sns.countplot("Embarked", data=titanic_df, ax=ax[0][0])

sns.countplot("Survived", data=titanic_df, ax=ax[0][1])

sns.countplot("Pclass", data=titanic_df, ax=ax[1][0])

sns.countplot("Sex", data=titanic_df, ax=ax[1][1])

fig.tight_layout(pad=3.0)

# fig.show()
fig, ax =plt.subplots(2,2, squeeze=False, figsize=(12, 8))

sns.countplot("Embarked", hue='Survived', data=titanic_df, ax=ax[0][0])

sns.countplot("Sex", hue='Survived', data=titanic_df, ax=ax[0][1])

sns.countplot("Pclass", hue='Survived', data=titanic_df, ax=ax[1][0])

fig.tight_layout(pad=3.0)

fig.delaxes(ax[1][1])

# fig.show()
fig, ax =plt.subplots(2, 2, squeeze=False, figsize=(10,8))

sns.boxplot(x='Survived', y='Age', data=titanic_df, ax=ax[0][0])

sns.boxplot(x='Survived', y='Fare', data=titanic_df, ax=ax[0][1])

sns.boxplot(x='Survived', y='SibSp', data=titanic_df, ax=ax[1][0])

sns.boxplot(x='Survived', y='Parch', data=titanic_df, ax=ax[1][1])

fig.tight_layout(pad=3.0)

# fig.show()
sns.catplot(x="Parch", y="Age", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="SibSp", y="Age", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="SibSp", y="Fare", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Parch", y="Fare", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Sex", y="Age", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Sex", y="Fare", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Pclass", y="Age", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Pclass", y="Fare", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Embarked", y="Age", col="Survived", data=titanic_df, height=5, aspect=.8)
sns.catplot(x="Embarked", y="Fare", col="Survived", data=titanic_df, height=5, aspect=.8)
fig, ax =plt.subplots(3, 2, squeeze=False, figsize=(10,8))

sns.scatterplot(x='Age', y='Fare', hue='Survived', data=titanic_df, ax=ax[0][0])

sns.scatterplot(x='SibSp', y='Age', hue='Survived', data=titanic_df, ax=ax[0][1])

sns.scatterplot(x='Parch', y='Age', hue='Survived', data=titanic_df, ax=ax[1][0])

sns.scatterplot(x='SibSp', y='Fare', hue='Survived', data=titanic_df, ax=ax[1][1])

sns.scatterplot(x='Parch', y='Fare', hue='Survived', data=titanic_df, ax=ax[2][0])

fig.tight_layout(pad=3.0)

fig.delaxes(ax[2][1])

# fig.show()
fig, ax =plt.subplots(2, 2, squeeze=False, figsize=(10,8))

sns.boxplot(titanic_df['Age'], ax=ax[0][0])

sns.boxplot(titanic_df['Fare'], ax=ax[0][1])

sns.boxplot(titanic_df['SibSp'], ax=ax[1][0])

sns.boxplot(titanic_df['Parch'], ax=ax[1][1])

fig.tight_layout(pad=3.0)

# fig.show()
for i in (test_df.select_dtypes(include ='object').columns):

    if(i != 'Survived'):

        data_crosstab = pd.crosstab(titanic_df[i], titanic_df['Survived'], margins = False)

        stat, p, dof, expected = stats.chi2_contingency(data_crosstab)

        prob=0.95

        alpha = 1.0 - prob

        if p <= alpha:

            print(i, ' : Dependent (reject H0)')

        else:

            print(i, ' : Independent (fail to reject H0)')
corr_train = titanic_df.corr()

corr_train
titanic_df['Survived'].value_counts()
sum(titanic_df['Survived'] == 1) / len(titanic_df)
#code for doing oversampling to balance the unbalanced dataset here



# from imblearn.over_sampling import SMOTE

# X_train = titanic_df.loc[:, titanic_df.columns != 'Survived']

# y_train = titanic_df.loc[:, titanic_df.columns == 'Survived']

# sm = SMOTE(random_state = 2) 

# X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
print('For Training Data: ')

for i in titanic_df.columns:

    if((titanic_df[i].isnull().sum()) > 0):

        p = ((titanic_df[i].isnull().sum()) / len(titanic_df[i])) * 100

        print('Proportion of null values in ', i, ' column is ', p)
data = titanic_df.copy()

data.drop(data[(data['Age'].isna()) & (data['Survived']==1)].index,axis=0,inplace=True)

#Test data appended so there is no need to process it separately

data = data.append(test_df)

data["Embarked"]=data['Embarked'].fillna(data['Embarked'].mode()[0])

#Imputing Age column with the mean value of Age column in the dataset

data['Age']= data['Age'].fillna(data['Age'].mean())

#Imputing Cabin column with 0 value and rest all values are replcaed by 1

data["Cabin"] = data['Cabin'].fillna(0)

data.loc[data['Cabin']!=0,'Cabin'] = 1

data["Cabin"]=data["Cabin"].astype('int64')
df_corr=data.loc[:,["Survived","Cabin"]]

sns.heatmap(df_corr.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Creating feature_final dataframe to store all the final features which we'll input to the model

feature_final=pd.DataFrame()

feature_final['Survived']=data['Survived']

feature_final['PassengerId']=data['PassengerId']

feature_final['Cabin']=data['Cabin']

feature_final['Fare']=data['Fare']

feature_final['SibSp']=data['SibSp']

feature_final['Parch']=data['Parch']

feature_final['Sex']=data['Sex']

#For sex column, I am replacing male with 0 and female with 1, and finally converted sex column to integer data type (to remove dummy variable trap)

feature_final.loc[feature_final['Sex']=='male','Sex'] = 0

feature_final.loc[feature_final['Sex'] == 'female','Sex'] = 1

feature_final['Sex']=feature_final['Sex'].astype('int64')

#Pclass column is converted to integer data type

data['Pclass'] = data['Pclass'].astype('int64')

feature_final['Pclass']= data['Pclass']

#Creating dummy for Embarked column

data['Embarked'] = pd.Categorical(data['Embarked'])

dfdummies = pd.get_dummies(data['Embarked'], prefix = 'Embarked')

feature_final = pd.concat([feature_final, dfdummies], axis=1)
#As age column contains lot of outliers, so it is needed to remove it, binning is one of the way to treat outlier

bins = [0,18,35,45,62,200]

#The above values are decided based on the plot for Age column

labels = ['Child','Teen','Youth','Old','Very_Old']

data['Age_binned'] = pd.cut(data['Age'], bins=bins, labels=labels)

#converting Age to Dummies and Adding to newly created DataFrame

data['Age_binned'] = pd.Categorical(data['Age_binned'])

feature_final['Age']= data['Age_binned']
#Creating a new column Is_Alone based on whether we have 0 family members or more

feature_final.loc[feature_final['SibSp']+feature_final['Parch'] == 0,'Is_Alone'] = 0

feature_final['Is_Alone']=feature_final['Is_Alone'].fillna(1)

feature_final['Is_Alone']=feature_final['Is_Alone'].astype('int64')
#Name column contains title, which is an important information to feed to model

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset = data.copy()

# extract titles

dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# replace titles with a more common title or as Rare

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                        'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# convert titles into numbers

dataset['Title'] = dataset['Title'].map(titles)

# filling NaN with 0, to get safe

dataset['Title'] = dataset['Title'].fillna(0)

data['Title'] = dataset['Title']

feature_final['Title']=dataset['Title']
#From ticket column, we can extract information which kind of ticket is bought 

data['Deck'] = data['Ticket'].apply(lambda x: x.split(' ')[0].split('/')[0].split('.')[0])

data['Deck'] = data['Deck'].apply(lambda x: 'Gen' if x.isnumeric()  else x)

data['Deck'] = pd.Categorical(data['Deck'])

dfdummies = pd.get_dummies(data['Deck'], prefix = 'Deck')

feature_final = pd.concat([feature_final, dfdummies], axis=1)
AgeList = { 'Child':1,'Teen':2, 'Youth':3, 'Old':4,'Very_Old':5}

feature_final['Age']= feature_final['Age'].map(AgeList)
train_df = feature_final.loc[feature_final['Survived'].notnull()]

test_df = feature_final.loc[feature_final['Survived'].isna()]
logreg = LogisticRegression(max_iter=1000)

Y_train = train_df['Survived']

X_train = train_df.drop(['Survived','PassengerId'],axis=1)

logreg.fit(X_train,Y_train)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# Imputing Fare column with mean value

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())

# Dropping unnecessary columns from test dataset

test= test_df.drop(['Survived','PassengerId'],axis=1)
##For running grid search making parameter dictionary

# param_grid = {

#               "n_estimators": [10, 18, 22],

#               "min_samples_split": [1, 2, 4, 10, 15, 20],

#               "criterion" : ['gini', 'entropy'],

#               "min_samples_leaf": [1, 5, 10, 20, 50, 70, 100],

#               "n_estimators": [100, 400, 700, 1000, 1500]}
##Performing grid search to find best parameters

# rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)

# clf.fit(X_train, Y_train)

##Printing the best parameters

# clf.bestparams
#Fitting the model

random_forest = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 1, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features='auto', 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)



random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)
#Predicting Result for Test dataset

Y_prediction = random_forest.predict(test).astype(int)
#Storing the prediction in Survived column of submit_df dataframe

submit_df['Survived'] = Y_prediction
# For saving the result to submit.csv file

submit_df.to_csv("submit.csv", index=False)