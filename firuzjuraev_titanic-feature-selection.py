from IPython.display import Image
Image(url= "https://cms.qz.com/wp-content/uploads/2014/01/titanic-paramout-pics-web.jpg?quality=75&strip=all&w=1100&h=619")
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
def side_by_side(*objs, **kwds):
    from pandas.io.formats.printing import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print (adjoin(space, *reprs))
    print()
    return
titanic_data = pd.read_csv("../input/titanic/train.csv", encoding = "UTF-8") 
titanic_data.columns
side_by_side(titanic_data.isnull().sum(), titanic_data.count())
titanic_data.drop(["Cabin"], axis=1, inplace = True) 
corr = titanic_data.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
titanic_data.columns
titanic_data["Embarked"].fillna('S', inplace=True)
fig,ax = plt.subplots(figsize=(20,8))
fig.suptitle("Titanic Data", fontsize=30)
corrcoef = titanic_data[['Survived', 'Pclass', 'Sex', 'Embarked', 
                         'Age', 'SibSp', 'Parch', 'Fare']].corr()
mask = np.array(corrcoef)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)
plt.show();
titanic_data['Title'] = titanic_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

titanic_data['Title'] = titanic_data['Title'].map(title_mapping)
#titanic_data["Embarked"].fillna('S', inplace=True)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
titanic_data['Embarked'] = titanic_data['Embarked'].map(embarked_mapping)
titanic_data.loc[ titanic_data['Age'] <= 16, 'Age'] = 0,
titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <= 26), 'Age'] = 1,
titanic_data.loc[(titanic_data['Age'] > 26) & (titanic_data['Age'] <= 36), 'Age'] = 2,
titanic_data.loc[(titanic_data['Age'] > 36) & (titanic_data['Age'] <= 62), 'Age'] = 3,
titanic_data.loc[ titanic_data['Age'] > 62, 'Age'] = 4
titanic_data["Age"].fillna(2, inplace=True)
side_by_side(titanic_data.isnull().sum(), titanic_data.count())
titanic_data["Gender"] = 0

for i in range(0, len(titanic_data)): 
    if titanic_data["Sex"].iloc[i] == 'female':
        titanic_data["Gender"].iloc[i] = 1         
titanic_data.head(10)
fig,ax = plt.subplots(figsize=(20,8))
fig.suptitle("Titanic Data", fontsize=30)
corrcoef = titanic_data[['Survived', 'Pclass', 'Gender', 
                         'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']].corr()
mask = np.array(corrcoef)
mask[np.tril_indices_from(mask)] = False
sns.heatmap(corrcoef, mask=mask, vmax=.8, annot=True, ax=ax)
plt.show();
women = titanic_data.loc[titanic_data.Gender == 1]["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)
men = titanic_data.loc[titanic_data.Gender == 0]["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)
def bar_chart(train_data, feature): 
    survived = train_data[train_data['Survived']==1][feature].value_counts()
    dead = train_data[train_data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead'] 
    df.plot(kind='bar', stacked=True, figsize=(10, 5)) 
bar_chart(titanic_data, 'Sex')
bar_chart(titanic_data, 'Pclass')
Features = ["Gender", "Parch", "Fare", "Pclass", "SibSp", "Title", "Embarked"]

Y = titanic_data["Survived"]
X = pd.get_dummies(titanic_data[Features])

#model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
#model = SVC() 
#model = KNeighborsClassifier(n_neighbors = 13) 
model.fit(X, Y)
 
Features = ["Gender", "Parch", "Fare", "Pclass", "SibSp", "Title", "Embarked"]
titanic_test_data = pd.read_csv("../input/titanic/test.csv", encoding = 'UTF-8')  

# Name 
titanic_test_data['Title'] = titanic_test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

titanic_test_data['Title'] = titanic_test_data['Title'].map(title_mapping)

# Embarked
titanic_test_data["Embarked"].fillna('S', inplace=True)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
titanic_test_data['Embarked'] = titanic_test_data['Embarked'].map(embarked_mapping)

# Gender 
titanic_test_data["Gender"] = 0

for i in range(0, len(titanic_test_data)): 
    if titanic_test_data["Sex"].iloc[i] == 'female':
        titanic_test_data["Gender"].iloc[i] = 1  

titanic_test_data.fillna(0, inplace = True)
side_by_side(titanic_test_data.isnull().sum(), titanic_test_data.count())
X_test = pd.get_dummies(titanic_test_data[Features]) 
predictions = model.predict(X_test) 

output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
