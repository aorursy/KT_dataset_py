# import pandas
import pandas as pd
from pandas import Series,DataFrame

# Create DataFrame
titanic_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

titanic_df.head()
# check ingo
titanic_df.info()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# check gender(sex)
sns.countplot('Sex',data=titanic_df)
# plot
sns.countplot('Sex',data=titanic_df,hue='Pclass')
# 
sns.countplot('Pclass',data=titanic_df,hue='Sex')
# define under 16 yrs are child

def male_female_child(passenger):
    # get age and sex
    age,sex = passenger
    # if under 16-> child, if not-> return sex(male or female)
    if age < 16:
        return 'child'
    else:
        return sex
    
# add person column
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)
test_df['person'] = test_df[['Age','Sex']].apply(male_female_child,axis=1)
titanic_df[0:10]

sns.countplot('Pclass',data=titanic_df,hue='person')

titanic_df['Age'].hist(bins=70)
titanic_df['Age'].mean()
titanic_df['person'].value_counts()

fig = sns.FacetGrid(titanic_df, hue="Sex",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue="person",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()
fig = sns.FacetGrid(titanic_df, hue="Pclass",aspect=4)
fig.map(sns.kdeplot,'Age',shade= True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

titanic_df.head()
deck = titanic_df['Cabin'].dropna()
deck = test_df['Cabin'].dropna()
# Quick preview of the decks
deck.head()

levels = []
for level in deck:
    levels.append(level[0])    

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d',order=sorted(set(levels)))
cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.countplot('Cabin',data=cabin_df,palette='summer',order=sorted(set(cabin_df['Cabin'])))
# 
titanic_df.head()
test_df.head()
sns.countplot('Embarked',data=titanic_df,hue='Pclass')
from collections import Counter
Counter(titanic_df.Embarked)
titanic_df.Embarked.value_counts()
titanic_df.head()
titanic_df['Alone'] =  titanic_df.Parch + titanic_df.SibSp
titanic_df['Alone']

test_df['Alone'] =  test_df.Parch + test_df.SibSp
test_df['Alone']

titanic_df['Alone'].loc[titanic_df['Alone'] >0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


test_df['Alone'].loc[test_df['Alone'] >0] = 'With Family'
test_df['Alone'].loc[test_df['Alone'] == 0] = 'Alone'
titanic_df.head()
test_df.head()
sns.countplot('Alone',data=titanic_df,palette='Blues')
titanic_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})

sns.countplot('Survivor',data=titanic_df,palette='Set1')
test_df["Survivor"] = titanic_df.Survived.map({0: "no", 1: "yes"})
sns.factorplot('Pclass','Survived',data=titanic_df, order=[1,2,3])
sns.factorplot('Pclass','Survived',hue='person',data=titanic_df, order=[1,2,3], aspect=2)
sns.lmplot('Age','Survived',data=titanic_df)
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter', hue_order=[1,2,3])
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations,hue_order=[1,2,3])
sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
# Predict with "Random Forest"
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
titanic_df.head()
test_df.head()
titanic_df["Sex"][titanic_df["Sex"] == "male"] = 0
titanic_df["Sex"][titanic_df["Sex"] == "female"] = 1

titanic_df["Embarked"][titanic_df["Embarked"] == "S"] = 0
titanic_df["Embarked"][titanic_df["Embarked"] == "C"] = 1
titanic_df["Embarked"][titanic_df["Embarked"] == "Q"] = 2

titanic_df["person"][titanic_df["person"] == "male"] = 0
titanic_df["person"][titanic_df["person"] == "female"] = 1
titanic_df["person"][titanic_df["person"] == "child"] = 2

titanic_df["Alone"][titanic_df["Alone"] == "With Family"] = 0
titanic_df["Alone"][titanic_df["Alone"] == "Alone"] = 1

titanic_df["Survivor"][titanic_df["Survivor"] == "yes"] = 0
titanic_df["Survivor"][titanic_df["Survivor"] == "no"] = 1

#test
test_df["Sex"][test_df["Sex"] == "male"] = 0
test_df["Sex"][test_df["Sex"] == "female"] = 1

test_df["Embarked"][test_df["Embarked"] == "S"] = 0
test_df["Embarked"][test_df["Embarked"] == "C"] = 1
test_df["Embarked"][test_df["Embarked"] == "Q"] = 2

test_df["person"][test_df["person"] == "male"] = 0
test_df["person"][test_df["person"] == "female"] = 1
test_df["person"][test_df["person"] == "child"] = 2

test_df["Alone"][test_df["Alone"] == "With Family"] = 0
test_df["Alone"][test_df["Alone"] == "Alone"] = 1

test_df["Survivor"][test_df["Survivor"] == "yes"] = 0
test_df["Survivor"][test_df["Survivor"] == "no"] = 1
titanic_df.info()
titanic_df["Age"] = titanic_df["Age"].fillna(round(titanic_df["Age"].median()))
titanic_df["Embarked"] = titanic_df["Embarked"].fillna(titanic_df["Embarked"].median())

#test
test_df["Age"] = test_df["Age"].fillna(round(test_df["Age"].median()))
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].median())
test_df["Fare"] = test_df["Fare"].fillna(round(test_df["Fare"].median()))



titanic_df.head()
test_df.info()
#evaluation
target = titanic_df["Survived"].values
features_one = titanic_df[["Pclass", "Sex", "Age", "Fare","Embarked","Alone","person"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
test_features = test_df[["Pclass", "Sex", "Age", "Fare","Embarked","Alone","person"]].values
my_prediction = my_tree_one.predict(test_features)
my_prediction.shape
print(my_prediction)
PassengerId = np.array(test_df["PassengerId"]).astype(int)
 
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])
