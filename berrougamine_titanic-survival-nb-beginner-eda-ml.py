from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/5095eabce4b06cb305058603/5095eabce4b02d37bef4c24c/1352002236895/100_anniversary_titanic_sinking_by_esai8mellows-d4xbme8.jpg?format=1000w")
import pandas as pd

import numpy as np

from pandas.plotting  import scatter_matrix

import matplotlib.pyplot as plt

from sklearn import model_selection 

from sklearn.metrics import classification_report 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.neighbors import KNeighborsClassifier

from collections import Counter

import seaborn as sns

import warnings#ignore alertes

warnings.filterwarnings('ignore')


train_df = pd.read_csv("../input/train.csv")



test_df = pd.read_csv("../input/test.csv")



train_df.head(10)


test_df.head(10)


train_df.describe()
train_df.info()
test_df.info()
#Detection of possible correlations between class and other attributes

g = sns.heatmap(train_df[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#Defined a function that represents the Sex, Pclass in barplot

def bar_chart(feature):

    survécu = train_df[train_df['Survived']==1][feature].value_counts()

    mort = train_df[train_df['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survécu,mort])

    df.index = ['Survived','dead']

    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('Sex')
#Explanation

train_df[["Sex","Survived"]].groupby('Sex').mean()
#display data by seaborn barplot

g = sns.barplot(x="Sex",y="Survived",data=train_df)

g = g.set_ylabel("Survival Probability")
bar_chart('Pclass')


g = sns.factorplot(x="Pclass",y="Survived",data=train_df,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")

#more details

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train_df,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("probabilité de survie par class ticket et Sex")
bar_chart('SibSp')
g  = sns.factorplot(x="SibSp",y="Survived",data=train_df,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
bar_chart('Parch')
g  = sns.factorplot(x="Parch",y="Survived",data=train_df,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
g = sns.FacetGrid(train_df, col='Survived')

g = g.map(sns.distplot, "Age")
# C = Cherbourg(France),Q = Queenstown(New-Zelande),S = Southampton(England)

bar_chart('Embarked')
g = sns.factorplot(x="Embarked", y="Survived",  data=train_df,size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
g = sns.factorplot("Pclass", col="Embarked",  data=train_df,size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")
from collections import Counter

def detect_outliers(df,n,features):



    outlier_indices = []

    



    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        



        outlier_step = 1.5 * IQR



        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index



        outlier_indices.extend(outlier_list_col)



    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



Outliers_to_drop = detect_outliers(train_df,2,["Age","SibSp","Parch","Fare"])
train_df.loc[Outliers_to_drop] 
# I defined outliers as being above of 99% percentile here

# get lists of people above 99% percentile for each feature

import pprint

highest = {}

for column in train_df.columns:

    if train_df[column].dtypes != "object": # exclude string data typed columns

        highest[column]=[]

        q = train_df[column].quantile(0.99)

        highest[column] = train_df[train_df[column] > q].index.tolist()

    

pprint.pprint(highest)
# delete 'PassengerId' from dictionary highest

highest.pop('PassengerId', 0)
# summarize the previous dictionary, highest

# create a dictionary of outliers and the frequency of being outlier

highest_count = {}

for feature in highest:

    for person in highest[feature]:

        if person not in highest_count:

            highest_count[person] = 1

        else:

            highest_count[person] += 1

             

highest_count
# This time, I defined outliers as being below of 1% percentile here

# get lists of people below 1% percentile for each feature

lowest = {}

for column in train_df.columns:

    if train_df[column].dtypes != "object": # exception string 

        lowest[column]=[]

        q = train_df[column].quantile(0.01)

        lowest[column] = train_df[train_df[column] < q].index.tolist()



# supp 'PassengerId' 

lowest.pop('PassengerId', 0)



pprint.pprint(lowest)
for person in lowest['Age']:

    if person not in highest_count:

        highest_count[person] = 1

    else:

        highest_count[person] += 1

 

highest_count
# fare >99%

train_df.loc[highest['Fare'],['Fare', 'Survived']]
# age above 99% percentile

train_df.loc[highest['Age'],['Age', 'Survived']]
# age below 1% percentile

train_df.loc[lowest['Age'],['Age','Survived']]


train_df.isnull().sum()


test_df.isnull().sum()


test_df["Fare"].isnull().sum()


test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())


test_df["Fare"].isnull().sum()


train_df["Embarked"].isnull().sum()


Pclass1 = train_df[train_df['Pclass']==1]['Embarked'].value_counts()

Pclass2 = train_df[train_df['Pclass']==2]['Embarked'].value_counts()

Pclass3 = train_df[train_df['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class','2nd class', '3rd class']

df.plot(kind='bar',stacked=True, figsize=(10,5));
train_test_data = [train_df, test_df]

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train_df["Embarked"].isnull().sum()
train_df.head()
train_df["Cabin"] = (train_df["Cabin"].notnull().astype('int'))

test_df["Cabin"] = (test_df["Cabin"].notnull().astype('int'))





print("survival % in cabins  = 1 :", train_df["Survived"][train_df["Cabin"] == 1].value_counts(normalize = True)[1]*100)



print("survival % in cabins  = 0 :", train_df["Survived"][train_df["Cabin"] == 0].value_counts(normalize = True)[1]*100)



sns.barplot(x="Cabin", y="Survived", data=train_df)

plt.show()


train_df = train_df.drop(['Cabin'], axis = 1)

test_df = test_df.drop(['Cabin'], axis = 1)
#train

print('pourcentage of missing values in "Age" is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))

train_df["Age"].isnull().sum()
#test

print('pourc of missing values in "Age" is %.2f%%' %((test_df['Age'].isnull().sum()/test_df.shape[0])*100))

test_df["Age"].isnull().sum()
g = sns.factorplot(y="Age",x="Sex",data=train_df,kind="box")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=train_df,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=train_df,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=train_df,kind="box")


g = sns.heatmap(train_df[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
dataset =  train_df 



# NaN values

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()#Median of age

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
g = sns.factorplot(x="Survived", y = "Age",data = train_df, kind="box")


train_df["Age"].isnull().sum()




dataset =  test_df 



# Nan values

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()#Median of age

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
test_df["Age"].isnull().sum()
train_df.isnull().sum()


test_df.isnull().sum()
train_df["Name"].head()
train_test_data = [train_df, test_df]



for dataset in train_test_data:

    dataset['Titre'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


train_df['Titre'].value_counts()
train_len = len(train_df)

dataset =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)



g = sns.countplot(x="Titre",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45) 


dataset[['Titre', 'Survived']].groupby(['Titre'], as_index=False).mean()


bar_chart('Titre')
#mapping list

mapping_nom = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,

                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

#application au dataset(train+test)

for dataset in train_test_data:

    dataset['Titre'] = dataset['Titre'].map(mapping_nom)

train_df.head()


test_df.head()
# M:0 et F:1

sexe_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sexe_mapping)
bar_chart('Sex')


train_df.drop('Name', axis=1, inplace=True)

test_df.drop('Name', axis=1, inplace=True)


embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train_df.head()
train_df["taillefamille"] = train_df["SibSp"] + train_df["Parch"] + 1

test_df["taillefamille"] = test_df["SibSp"] + test_df["Parch"] + 1


g = sns.factorplot(x="taillefamille",y="Survived",data = train_df)

g = g.set_ylabels("Probabilité de survie")
#solo= alone / pfamille=smallfamily / Mfamille=Medium / Gfamille=Bigfamily

train_df['Solo'] = train_df['taillefamille'].map(lambda s: 1 if s == 1 else 0)

train_df['Pfamille'] = train_df['taillefamille'].map(lambda s: 1 if s == 2  else 0)

train_df['Mfamille'] = train_df['taillefamille'].map(lambda s: 1 if 3 <= s <= 4 else 0)

train_df['Gfamille'] = train_df['taillefamille'].map(lambda s: 1 if s >= 5 else 0)


g = sns.factorplot(x="Solo",y="Survived",data=train_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="Pfamille",y="Survived",data=train_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="Mfamille",y="Survived",data=train_df,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="Gfamille",y="Survived",data=train_df,kind="bar")

g = g.set_ylabels("Survival Probability")
train_df.head()
#for test data

test_df['Solo'] = test_df['taillefamille'].map(lambda s: 1 if s == 1 else 0)

test_df['Pfamille'] = test_df['taillefamille'].map(lambda s: 1 if s == 2  else 0)

test_df['Mfamille'] = test_df['taillefamille'].map(lambda s: 1 if 3 <= s <= 4 else 0)

test_df['Gfamille'] = test_df['taillefamille'].map(lambda s: 1 if s >= 5 else 0)
test_df.head()


features_drop = ['Ticket', 'SibSp', 'Parch']



train_df = train_df.drop(features_drop, axis=1)

test_df = test_df.drop(features_drop, axis=1)


train_df.head(10)


test_df.head(5)
train_df.info()
from sklearn.model_selection import train_test_split



X_all = train_df.drop(['Survived', 'PassengerId'], axis=1)

Y_all = train_df['Survived']



num_test = 0.20 #20% for test

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=num_test, random_state=23)
#KNN



from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

knn = KNeighborsClassifier()



# Choose some parameter combinations to try

parameters = {'n_neighbors':[3, 5, 7],

              'weights':['uniform'], 

              'algorithm':['auto'], 

              'leaf_size':[30]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(knn, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))


# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

# Choose the type of classifier. 

GBC = GradientBoostingClassifier()



# Choose some parameter combinations to try

parameters = {

              'max_depth': [1, 2, 3, 4, 5],

              'max_features': [1, 2, 3, 4]}

             



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(GBC, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier

# Choose the type of classifier. 

RFC = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {

              'max_depth': [1, 2, 3, 4, 5],

              'max_features': [1, 2, 3, 4]}

             



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(RFC, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#CART



# Choose the type of classifier. 

CART = DecisionTreeClassifier()



# Choose some parameter combinations to try

parameters = {'max_depth': [1, 2, 3, 4, 5],

              'max_features': [1, 2, 3, 4]}

             



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(CART, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, Y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))




predictions = clf.predict(X_test)

print(accuracy_score(Y_test, predictions))
#Confusion matrix



predictions = clf.predict(X_test)

print(confusion_matrix(Y_test, predictions))
predictions = clf.predict(X_test)

print(classification_report(Y_test, predictions))
ids = test_df['PassengerId']

predictions = clf.predict(test_df.drop('PassengerId', axis=1))



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('test_result.csv', index = False)

output.head()
from IPython.display import Image

Image(url= "https://static1.squarespace.com/static/5006453fe4b09ef2252ba068/t/5090b249e4b047ba54dfd258/1351660113175/TItanic-Survival-Infographic.jpg?format=1500w")