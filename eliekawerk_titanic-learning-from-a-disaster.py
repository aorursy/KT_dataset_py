import numpy as np  # linear algebra

import pandas as pd # data wrangling

import matplotlib.pyplot as plt # plotting

import seaborn as sns # statistical plots and aesethics

import re # regular expression



######### Preprocessing #######

from sklearn.preprocessing import (LabelEncoder, Imputer, StandardScaler) # data preparation



##### Machine learning models ##############

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import  (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier)

from xgboost import XGBClassifier



##### Model evaluation and hyperparameter tuning ##############

from sklearn.model_selection import (cross_val_score, GridSearchCV, StratifiedKFold,\

                                   RandomizedSearchCV, train_test_split,\

                                   learning_curve, validation_curve)

from sklearn.pipeline import Pipeline

from sklearn.metrics import (f1_score, classification_report, roc_auc_score, roc_curve)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



df_train.head(3)
df_test.head()
df_train.info()
df_test.info()
df_train.shape
df_test.shape
pd.set_option('display.max_rows', 500)

print(df_train.dtypes)
df_train.drop(['PassengerId', 'Survived','Pclass'], axis=1).describe()
categorical_variables = ['Survived', 'Pclass', 'Sex','Embarked']

for cat_var in categorical_variables:

    print("----------------------------")

    print("Distribiton of %s" %(cat_var))

    print(df_train[cat_var].value_counts())

    print("----------------------------")
sns.set_style('whitegrid')

sns.countplot('Pclass', data=df_train, hue="Survived")
sns.countplot('Sex', data=df_train, hue="Survived")
g = sns.FacetGrid(df_train, row = 'Survived', hue='Sex', size=4, aspect=2)

g = (g.map(sns.kdeplot,'Age', shade='True')).add_legend()
g = sns.FacetGrid(df_train, row= "Pclass" , col = 'Survived', hue='Sex', size=4, aspect=1)

g = (g.map(sns.kdeplot,'Age', shade='True')).add_legend()
sns.countplot('Embarked', data=df_train, hue='Survived')
g = sns.FacetGrid(df_train, col= "Pclass", hue ='Survived' , size=4, aspect=1)

g = (g.map(sns.countplot,'Embarked')).add_legend()
g = sns.FacetGrid(df_train, hue='Survived', size=4, aspect=2)

g = (g.map(sns.kdeplot, 'Fare', shade=True)).add_legend()

plt.xlim(-10, 125)

plt.show()
sns.countplot('SibSp', data=df_train, hue='Survived')

plt.show()
sns.countplot('Parch', data=df_train, hue='Survived')

plt.legend(loc='center')

plt.show()
g = sns.FacetGrid(data=df_train, col='Survived', hue='Sex', size=4, aspect=1)

g = (g.map(sns.countplot, 'Parch')).add_legend()

plt.show()
sns.heatmap(df_train.isnull(),  yticklabels=False, cbar=False, cmap='viridis')

plt.suptitle('Missing values in the training set')

plt.show()
sns.heatmap(df_test.isnull(),  yticklabels=False, cbar=False, cmap='viridis')

plt.suptitle('Missing values in the test set')

plt.show()
for feature in ['Pclass', 'Embarked','Sex', 'Survived', 'SibSp', 'Parch']:

    plt.suptitle('Age distribution by %s' %(feature))

    sns.boxplot(x=feature, y='Age', data=df_train)

    plt.show()
medians_by_parch = []



for i in df_train['Parch'].unique().tolist():

    medians_by_parch.append(df_train[df_train['Parch'] == i]['Age'].median())



for i, median_age in enumerate(medians_by_parch):

    print('For a number of Parents/Children of %d, the median age is %f' %(i,median_age))
def impute_age(cols, medians_by_parch):

    Parch = cols['Parch']

    Age = cols['Age']

    

    if pd.isnull(Age):

        return medians_by_parch[Parch]

    else:

        return Age

    

df_train['Age'] =  df_train.apply(impute_age, args =(medians_by_parch,) , axis=1)

df_test['Age']  =  df_train.apply(impute_age, args =(medians_by_parch,) , axis=1)
df_train[pd.isnull(df_train['Embarked'])]
cond = (df_train['Sex']=='female') & (df_train['Survived']==1) & (df_train['Pclass']== 1)

sns.countplot(df_train[cond]['Embarked'])
cond = pd.isnull(df_train["Embarked"])

df_train.loc[cond,'Embarked'] = 'S'
sum(pd.isnull(df_test['Fare']))
df_test[pd.isnull(df_test['Fare'])]
df_test[pd.isnull(df_test['Fare'])] = df_train[df_train['Pclass'] == 3]['Fare'].median()
Cabin_dist = df_train["Cabin"].dropna().apply(lambda x: x[0])



sns.countplot(Cabin_dist, palette='coolwarm')

plt.show()
del Cabin_dist 



df_train.drop('Cabin', axis=1, inplace=True)

df_test.drop('Cabin', axis=1, inplace=True)
corr = df_train.drop("PassengerId",axis=1).corr()

print(corr)



plt.figure(figsize=(12,12))

sns.heatmap(corr, annot=True, cbar=True, square=True, fmt='.2f', cmap='coolwarm')

plt.show()
plt.figure(figsize=(12,12))

sns.pairplot(df_train[['Age','SibSp','Parch','Fare']])

plt.show()
def is_alone(passenger):

    var = passenger['SibSp'] + passenger['Parch']

    # if var = 0 then passenger was alone 

    # Otherwise passenger was with siblings or family or both

    if var == 0:

        return 1

    else:

        return 0

    

df_train['Alone'] = df_train.apply(is_alone, axis=1)

df_test["Alone"] = df_test.apply(is_alone, axis=1)
sns.countplot('Alone', data=df_train, hue='Survived' )
def is_minor(age):

    if age < 18.0:

        return 1

    else:

        return 0 



df_train['Minor'] = df_train["Age"].apply(is_minor)

df_test['Minor'] = df_test["Age"].apply(is_minor)
sns.countplot('Minor', data=df_train, hue='Survived')
def get_title(name, title_Regex):

    if type(name) == str:

        return title_Regex.search(name).groups()[0]

    else:

        return 'Mr'



title_Regex = re.compile(r',\s(\w+\s?\w*)\.\s', re.I)

    

df_train["Title"] =  df_train["Name"].apply(get_title, args=(title_Regex,))

# There s a floating number in the test set at index 152, I created a function  (get_title) to surpass this

# and replace it with 'Mr'

df_test["Title"] =  df_test["Name"].apply(get_title, args = (title_Regex,))



plt.figure(figsize=(14,7))

g = sns.countplot('Title', data=df_train)

plt.xticks(rotation=50)
print(df_train["Title"].unique())
dict_title = {

    'Mr': 'Mr',

    'Miss': 'Miss',

    'Mlle': 'Miss',

    'Mrs': 'Mrs',

    'Mme': 'Mrs',

    'Dona': 'Nobility',

    'Lady': 'Nobility', 

    'the Countess': 'Nobility',

    'Capt': 'Nobility',

    'Col': 'Nobility',

    'Don': 'Nobility',

    'Dr': 'Nobility',

    'Major': 'Nobility',

    'Rev': 'Nobility', 

    'Sir': 'Nobility',

    'Jonkheer': 'Nobility',    

  }



df_train["Title"] =  df_train["Title"].map(dict_title)



plt.figure(figsize=(14,7))

sns.countplot('Title', data=df_train)

plt.show()

