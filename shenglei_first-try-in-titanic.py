import pandas as pd

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
test.shape
train.describe()
test.describe()
train.info()
train.isnull().sum()
test.isnull().sum()
# any的意思是其中有null的，all的意思是这一列中全是null

train_null_columns = train.columns[train.isnull().any()]
test_null_columns = test.columns[test.isnull().any()]
train[train['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train);
train['Embarked'] = train['Embarked'].fillna('C')
test[test['Fare'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test);
fare_median = test[(test['Pclass'] == 3) & (test['Embarked'] == 'S')]['Fare'].median()
test['Fare'] = test['Fare'].fillna(fare_median)
train['Deck'] = train['Cabin'].str[0]

test['Deck'] = test['Cabin'].str[0]
train['Deck'].unique() #查看船舱
train[train['Deck'].isnull()]['Pclass'].describe()
test['Deck'].unique() #查看船舱
test[test['Deck'].isnull()]['Pclass'].describe()
train.loc[train['Pclass'] == 3]['Deck'].describe()
test.loc[test['Pclass'] == 3]['Deck'].describe()
# train['Deck'] = train['Deck'].fillna('F')

# test['Deck'] = test['Deck'].fillna('F')



train['Deck'] = train['Deck'].fillna('Z')

test['Deck'] = test['Deck'].fillna('Z')
train['Deck'].unique() #查看船舱
test['Deck'].unique() #查看船舱
train['Family'] = train['SibSp'] + train['Parch'] + 1

test['Family'] = test['SibSp'] + test['Parch'] + 1
sns.countplot(x="Family", hue="Survived", data=train);
#train[train['Family'] == 1]['FamilyType'] = 'singleton'

#train[(train['Family'] > 1) & (train['Family'] < 5)]['FamilyType'] = 'small'

#train[train['Family'] > 4]['FamilyType'] = 'large'



#要使用loc才行，一部分不能作为左值

train.loc[train['Family'] == 1, 'FamilyType'] = 'singleton'

train.loc[(train['Family'] > 1) & (train['Family'] < 5), 'FamilyType'] = 'small'

train.loc[train['Family'] > 4, 'FamilyType'] = 'large'
test.loc[test['Family'] == 1, 'FamilyType'] = 'singleton'

test.loc[(test['Family'] > 1) & (test['Family'] < 5), 'FamilyType'] = 'small'

test.loc[test['Family'] > 4, 'FamilyType'] = 'large'
train['Name']
import re



def get_title(name):

    title = re.compile('(.*, )|(\\..*)').sub('',name)

    return title



titles = train['Name'].apply(get_title)

print(pd.value_counts(titles))

train['Title'] = titles



titles = test['Name'].apply(get_title)

test['Title'] = titles
rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

train.loc[train["Title"] == "Mlle", "Title"] = 'Miss'

train.loc[train["Title"] == "Ms", "Title"] = 'Miss'

train.loc[train["Title"] == "Mme", "Title"] = 'Mrs'

train.loc[train["Title"] == "Dona", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Lady", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Countess", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Capt", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Col", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Don", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Major", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Rev", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Sir", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Jonkheer", "Title"] = 'Rare Title'

train.loc[train["Title"] == "Dr", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Mlle", "Title"] = 'Miss'

test.loc[test["Title"] == "Ms", "Title"] = 'Miss'

test.loc[test["Title"] == "Mme", "Title"] = 'Mrs'

test.loc[test["Title"] == "Dona", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Lady", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Countess", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Capt", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Col", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Don", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Major", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Rev", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Sir", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

test.loc[test["Title"] == "Dr", "Title"] = 'Rare Title'
test['Title'].value_counts()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex',"Title","FamilyType",'Deck']

for col in cat_vars:

    train[col]=labelEnc.fit_transform(train[col])

    test[col]=labelEnc.fit_transform(test[col])



train.head()
from sklearn.ensemble import RandomForestRegressor



def fill_missing_age(data):

    

    #Feature set

    features = data[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'Title','Pclass','Family',

                 'FamilyType', 'Deck']]

    # Split sets into train and prediction

    train  = features.loc[ (data.Age.notnull()) ]# known Age values

    prediction = features.loc[ (data.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(prediction.values[:, 1::])

    

    # Assign those predictions to the full data set

    data.loc[ (data.Age.isnull()), 'Age' ] = predictedAges 

    

    return data



train=fill_missing_age(train)

test=fill_missing_age(test)
train['Age'].describe()
train['Fare'].describe()
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(train[['Age', 'Fare']])

train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(test[['Age', 'Fare']])

test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])
correlation=train.corr()

plt.figure(figsize=(10, 10))



sns.heatmap(correlation, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
train.corr()['Survived']
print(train.isnull().sum())

print(test.isnull().sum())
features = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",

             "Embarked", "FamilyType", "Title","Deck"]

#features = ["Pclass", "Sex", "Age", "Fare", "Family",

              #"Embarked", "FamilyType", "Title","Deck"]

target="Survived"



X_train = train[features]



y_train = train[target]



X_test = test[features]
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold



lr = LinearRegression()



lr.fit(X_train, y_train)



y_test = lr.predict(X_test)



y_test[y_test > .5] = 1

y_test[y_test <=.5] = 0



y_test = y_test.astype(int)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test

    })

submission.to_csv('titanic_linear.csv', index=False)
from sklearn import tree



dtc = tree.DecisionTreeClassifier()

dtc = dtc.fit(X_train, y_train)



y_test = dtc.predict(X_test)

y_test = y_test.astype(int)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test

    })

submission.to_csv('titanic_tree.csv', index=False)
from sklearn import ensemble



rf = ensemble.RandomForestClassifier()

rf = rf.fit(X_train, y_train)



y_test = rf.predict(X_test)

y_test = y_test.astype(int)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test

    })

submission.to_csv('titanic_forest.csv', index=False)
import xgboost as xgb



model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)



y_test = model.predict(X_test)

y_test = y_test.astype(int)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_test

    })

submission.to_csv('titanic_xgboost.csv', index=False)