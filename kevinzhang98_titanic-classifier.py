import numpy as np

import pandas as pd 

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

unchanged_data = test_data
train_data.head()
sns.countplot(x = "Sex", hue ="Survived",data = train_data, palette = "Blues")
sns.countplot(x = "Pclass", hue ="Survived",data = train_data, palette = "Blues");
sns.countplot(x = "Pclass", hue ="Sex",data = train_data, palette = "Blues")
sns.countplot(x = "Parch", hue ="Survived",data = train_data, palette = "Blues")
sns.countplot(x = "Embarked", hue="Survived", data = train_data)
train_data.describe()
train_data.isnull().sum()
Pclass_1_avg_age = train_data[train_data['Pclass']==1]['Age'].median()

Pclass_2_avg_age = train_data[train_data['Pclass']==2]['Age'].median()

Pclass_3_avg_age = train_data[train_data['Pclass']==3]['Age'].median()



def fill_age(age):

    if str(age[5]).lower()=='nan':

        if age[2]==1:

            return Pclass_1_avg_age

        elif age[2]==2:

            return Pclass_2_avg_age

        else:

            return Pclass_3_avg_age

    else:

        return age[5]



train_data['Age']=train_data.apply(fill_age,axis=1)
train_data['Embarked'].fillna(train_data['Embarked'].value_counts().index[0], inplace=True)
train_data.isnull().sum()
train_data["Sex"].value_counts()
train_data["Family_Size"] = train_data["SibSp"] + train_data["Parch"]

y = train_data["Survived"]
p = {1:'1st',2:'2nd',3:'3rd'} 

train_data['Pclass'] = train_data['Pclass'].map(p)
titles = set()

for name in train_data["Name"]:

    titles.add(name.split(",")[1].split(".")[0].strip())

titles
dict_of_title = {

    "Dr": "Officer",

    "Rev": "Officer",

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Jonkheer": "Royal",

    "Lady" : "Royal",

    "Don": "Royal",

    "Sir" : "Royal",

    "the Countess":"Royal"

}



def make_titles(data):

    data["Title"] = data["Name"].map(lambda s:s.split(",")[1].split(".")[0].strip())

    data["Title"] = data["Title"].map(dict_of_title)

    return data



train_data = make_titles(train_data)



train_data
features = ["Pclass","Age","Sex","Fare","Title"]

train_data = train_data[features]

categorical_df = train_data[['Pclass',"Sex","Title"]]

one_hot_encode = pd.get_dummies(categorical_df,drop_first=True) 

train_data = train_data.drop(['Pclass',"Sex","Title"],axis=1)

train_data = pd.concat([train_data,one_hot_encode],axis=1)
#X = train_data

#train_X, test_X, train_y, test_y = train_test_split(X,y,random_state = 0)
# clf = RandomForestClassifier(random_state = 0)

# clf.fit(train_X,train_y)
#pred = clf.predict(test_X)
#metrics.accuracy_score(test_y, pred)
#confusion_matrix(test_y,pred)
test_data.head()
test_data.isnull().sum()
Pclass_1_avg_age_test = test_data[test_data['Pclass']==1]['Age'].median()

Pclass_2_avg_age_test = test_data[test_data['Pclass']==2]['Age'].median()

Pclass_3_avg_age_test = test_data[test_data['Pclass']==3]['Age'].median()



def fill_age_kaggle(age):

    if str(age[4]).lower()=='nan':

        if age[1]==1:

            return Pclass_1_avg_age_test

        elif age[1]==2:

            return Pclass_2_avg_age_test

        else:

            return Pclass_3_avg_age_test

    else:

        return age[4]

#test_data['Age'].fillna(test_data['Age'].median(),inplace=True)

test_data['Age']=test_data.apply(fill_age_kaggle,axis=1)

test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)

test_data["Family_Size"] = test_data["SibSp"] + test_data["Parch"]

p = {1:'1st',2:'2nd',3:'3rd'} 

test_data['Pclass'] = test_data['Pclass'].map(p)

test_data = make_titles(test_data)

test_data = test_data[features]

categorical_df = test_data[['Pclass',"Sex","Title"]]

one_hot_encode = pd.get_dummies(categorical_df,drop_first=True) 

test_data = test_data.drop(['Pclass',"Sex","Title"],axis=1)

test_data = pd.concat([test_data,one_hot_encode],axis=1)
X = train_data

#test_data = make_titles(test_data)

royal = np.zeros(418,dtype=int)

test_data["Title_Royal"] = royal

test_data
train_data
classifier = RandomForestClassifier(n_estimators=300, random_state=0)  

grid_param = {  

    'n_estimators': [100, 300, 500, 800, 1000],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False]

}
gd_sr = GridSearchCV(estimator=classifier,  

                     param_grid=grid_param,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)
#gd_sr.fit(X, y) 
#best_parameters = gd_sr.best_params_  

#print(best_parameters) 
parameters = {

    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "min_samples_split": np.linspace(0.1, 0.5, 12),

    "min_samples_leaf": np.linspace(0.1, 0.5, 12),

    "max_depth":[3,5,8],

    }

#clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1)

#clf.fit(X,y)

#print(clf.best_params_)
kaggle_clf = GradientBoostingClassifier(learning_rate= 0.1, max_depth= 5, min_samples_leaf=0.1, min_samples_split = 0.1).fit(X,y)

kaggle_pred = kaggle_clf.predict(test_data)
kaggle_pred
my_submission = pd.DataFrame({'PassengerId': unchanged_data.PassengerId, 'Survived': kaggle_pred})
my_submission.to_csv('submission.csv', index=False)