import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

cmap = sns.diverging_palette(250, 10, as_cmap=True)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
datasets = [train, test]
data = pd.concat([train, test])
for dataset in datasets:
    dataset["Embarked"].fillna("S", inplace = True)
for dataset in datasets:
    dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, "S": 2})
train.head()
#Ticket
data = pd.concat([train, test])
df = data["Ticket"].value_counts()

family_tickets = df.loc[df > 1].index
family_tickets.tolist()

single_tickets = df.loc[df == 1].index
single_tickets.tolist()

data['TicketCat'] = data['Ticket'].copy()

for ticket in single_tickets:
    data.loc[data["Ticket"] == ticket, "TicketCategory"] = 0

i = 1
for ticket in family_tickets:
    data.loc[data["Ticket"] == ticket, "TicketCategory"] = i
    i+=1
    
train["TicketCategory"] = data["TicketCategory"][:891].astype(int)
test["TicketCategory"] = data["TicketCategory"][891:].astype(int)
for dataset in datasets:
    dataset["FamilySize"] = dataset["SibSp"]+dataset["Parch"]+1
train.head()
for dataset in datasets:
    dataset["FamilySize"] = np.where((dataset["FamilySize"]) == 1 , 'Solo',
                           np.where((dataset["FamilySize"]) <= 4,'Medium', 'Big'))
    
    dataset["FamilySize"] = dataset["FamilySize"].map({"Solo":0, "Medium":1, "Big":2})
for dataset in datasets:
    dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})
train.head()
for dataset in datasets:
    dataset["Title"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()
for dataset in datasets:
    dataset["Title"] = dataset["Title"].replace(["Dr", "Rev", "Major", "Col", "Mlle", "Don", "Jonkheer", "Lady", "Mme", "Countess", "Ms", "Sir", "Capt"], "Other")
    dataset["Title"] = dataset["Title"].map({"Mr":0, "Miss":1, "Mrs":2 , "Master": 3, "Other" :4})
for dataset in datasets:
    dataset["Title"].fillna(0, inplace = True)
for dataset in datasets:
        df = train.groupby(['Title', 'Pclass'])['Age']
        dataset['Age'] = df.transform(lambda x: x.fillna(x.mean()))
for dataset in datasets:
    dataset['Age_band']=0
    dataset.loc[dataset['Age']<=16,'Age_band']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age_band']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age_band']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age_band']=3
    dataset.loc[dataset['Age']>64,'Age_band']=4
for dataset in datasets:
    dataset["Fare"].fillna(dataset["Fare"].mean(), inplace = True)
for dataset in datasets:
    dataset['Fare_cat']=0
    dataset.loc[dataset['Fare']<=7.91,'Fare_cat']=0
    dataset.loc[(dataset['Fare']>7.91)&(dataset['Fare']<=14.454),'Fare_cat']=1
    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare_cat']=2
    dataset.loc[(dataset['Fare']>31)&(dataset['Fare']<=513),'Fare_cat']=3
for dataset in datasets:
    dataset.drop(["SibSp"], axis = 1, inplace = True)
    dataset.drop(["Parch"], axis = 1, inplace = True)
    dataset.drop(["Cabin"], axis = 1, inplace = True)
    dataset.drop(["Name"], axis = 1, inplace = True)
    dataset.drop(["PassengerId"], axis = 1, inplace = True)
    dataset.drop(["Ticket"], axis = 1, inplace = True)
    dataset.drop(["FamilySize"], axis = 1, inplace = True)
    dataset.drop(["Age"], axis = 1, inplace = True)
    dataset.drop(["Fare"], axis = 1, inplace = True)
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(["Survived"], axis = 1), train["Survived"])
train.head(50)
# Fitting Random Forest Classification to the Training set
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_features='auto', 
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

# Creating the Grid Search Parameter list
parameters = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1, 5, 10],
             "min_samples_split" : [12, 16, 20, 24],
             "n_estimators": [100, 400, 700]}

# Setting up the gridSearch to find the optimal parameters
gridSearch = GridSearchCV(estimator=classifier,
                  param_grid=parameters,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

# Getting the optimal grid search parameters
gridSearch = gridSearch.fit(X_train, y_train)

# Printing the out of bag score and the best parameters values
print(gridSearch.best_score_)
print(gridSearch.best_params_)

# building the random forrest classifier
classifier = RandomForestClassifier(criterion='entropy', 
                             n_estimators=100,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
classifier.fit(X_train, y_train)
print("%.5f" % classifier.oob_score_)

# Creating the list of important features
pd.concat((pd.DataFrame(X_train.columns, columns = ['variable']), 
           pd.DataFrame(classifier.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:9]

prediction = classifier.predict(test)

temp = pd.DataFrame(pd.read_csv("../input/test.csv")['PassengerId'])
temp['Survived'] = prediction
temp.to_csv("../working/submission.csv", index = False)
