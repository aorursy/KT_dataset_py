# Import of libraries

import pandas as pd

import numpy as np

import matplotlib as mtl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy as sc

import sklearn

import warnings

warnings.filterwarnings('ignore') 

plt.rcParams['figure.figsize'] = (10, 10)



#Import of data

prediction = pd.read_csv("../input/test.csv", sep = ",")

train = pd.read_csv("../input/train.csv", sep = ",")

example = pd.read_csv("../input/gender_submission.csv", sep = ",")



import sys

print ("Python version: {}".format(sys.version))

print ("Numpy version: {}".format(np.__version__))

print ("Pandas version: {}".format(pd.__version__))

print ("Matplotlib version: {}".format(mtl.__version__))

print ("Seaborn version: {}".format(sns.__version__))

print ("Scipy version: {}".format(sc.__version__))

print ("Scikit version: {}".format(sklearn.__version__))
# OUR TRAINING SET

train.head()
# OUR SET TO MAKE PREDICTIONS AND TO SUBMIT TO KAGGLE 

prediction.head()
# THAT'S HOW OUR PREDICTION SHOULD LOOK LIKE

example.head()
# STORING indexes

prediction_indexes = prediction["PassengerId"]

train_indexes = train["PassengerId"]



# REINDEXING

prediction.set_index("PassengerId", inplace = True)

train.set_index("PassengerId", inplace = True)
# Concatenation

frame = [train, prediction]

data = pd.concat(frame)
del train, prediction, example
data.dtypes
pd.value_counts(data["Survived"][train_indexes])
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Calculating the percentage of missing values in 'data'

percentage = (data.isnull().sum()/len(data))*100

percentage.sort_values(inplace=True)

percentage
# Imputing missing values in 'Age'

for i in range(len(data.Age)):

    if data.Age.isnull()[i+1]: 

        data.Age[i+1] = data[(data["Pclass"] == data["Pclass"][i+1]) & (data["Sex"] == data["Sex"][i+1])]["Age"].median()

data["Age"].isnull().any()
pd.value_counts(data[data['Cabin'].isnull()==True]['Survived'])
data["Deck"]=data.Cabin.str[0]

data["Deck"].unique() # 0 is for null values
data.Deck.fillna('Z', inplace=True)

data["Deck"].unique() # Z is for null values
# Deleting 'Cabin'

del data['Cabin']
data.Embarked.fillna('S', inplace=True)
data['Fare'] = data['Fare'].fillna(data['Fare'].median())
g = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=data[data.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8);
pd.value_counts(data['Deck'])
# Tranforming strings into numerical format

mapper = {'A': 1, 'B': 2, 'C': 3,'D': 4,'E': 5,'F': 6,'G': 7,'T': 8,'Z': 0,}

data["Deck"] = data["Deck"].map(mapper)
data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                             'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



data['Title'] = data['Title'].replace('Mlle', 'Miss')

data['Title'] = data['Title'].replace('Ms', 'Miss')

data['Title'] = data['Title'].replace('Mme', 'Mrs')
pd.value_counts(data['Title'])
sns.factorplot("Survived", col="Title", col_wrap=4,

                    data=data[data.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8);
data["NameLength"] = data["Name"].apply(lambda x: len(x))



bins = [0, 20, 40, 57, 85]

group_names = ['short', 'okay', 'good', 'long']

data['NlengthD'] = pd.cut(data['NameLength'], bins, labels=group_names)



sns.factorplot(x="NlengthD", y="Survived", data=data)

print(data["NlengthD"].unique())
del data['Name']

del data['NameLength']
data['Ticket'].head()
del data['Ticket']
#Relative values

cross = pd.crosstab(data["Pclass"][train_indexes], data["Survived"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
#Relative values

cross = pd.crosstab(data["Sex"][train_indexes], data["Survived"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
#Relative values

cross = pd.crosstab(data["SibSp"][train_indexes], data["Survived"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
#Relative values

cross = pd.crosstab(data["SibSp"][train_indexes], data["Pclass"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
experiment = data[data["Pclass"]== 3]

#Relative values

cross = pd.crosstab(experiment["Survived"][train_indexes], experiment["SibSp"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
del experiment
#Relative values

cross = pd.crosstab(data["Parch"][train_indexes], data["Survived"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
experiment = data[data["Pclass"]== 3]

#Relative values

cross = pd.crosstab(experiment["Survived"][train_indexes], experiment["Parch"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
#Relative values

cross = pd.crosstab(data["Parch"][train_indexes], data["Pclass"][train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
del experiment
sns.kdeplot(data["Fare"][data.Survived == 1])

sns.kdeplot(data["Fare"][data.Survived == 0])

plt.legend(['Survived', 'Died'])

plt.xlim(-20,200)

plt.show()
data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0

data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1

data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2

data.loc[ data['Fare'] > 31, 'Fare'] = 3

data['Fare'] = data['Fare'].astype(int)
cross = pd.crosstab(data.Embarked[train_indexes], data.Survived[train_indexes])

cross.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#Relative values

cross = pd.crosstab(data.Embarked[train_indexes], data.Survived[train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
cross = pd.crosstab(data.Embarked[train_indexes], data.Pclass[train_indexes])

cross
#Relative values

cross = pd.crosstab(data.Embarked[train_indexes], data.Pclass[train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
data["Age"].hist()
# Categorization of numerical variable Age

data["Cat_Age"] = np.nan

data.loc[data['Age'] <= 5, 'Cat_Age'] = "Infant"

data.loc[(data['Age'] > 5) & (data['Age'] <= 18), 'Cat_Age'] = "Child"

data.loc[(data['Age'] > 18) & (data['Age'] <= 55), 'Cat_Age'] = "Adult"

data.loc[data['Age'] > 55, 'Cat_Age'] = "Senior"

del data['Age']
data.head()
#Relative values

cross = pd.crosstab(data.Cat_Age, data.Survived)

cross = cross.reindex(['Infant', 'Child', 'Adult', 'Senior'])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
#Relative values

cross = pd.crosstab(data.Cat_Age[train_indexes], data.Pclass[train_indexes])

cross = cross.div(cross.sum(1).astype(float), axis=0)

cross.plot(kind='bar', stacked=True,  grid=False)
mapper_1 = {'male': 0, 'female': 1}

data["Sex"] = data["Sex"].map(mapper_1)



mapper_2 = {'Infant': 0, 'Child': 1, 'Adult':2, 'Senior': 3}

data["Cat_Age"] = data["Cat_Age"].map(mapper_2)



mapper_3 = {'S': 0, 'Q': 1, 'C': 2}

data["Embarked"] = data["Embarked"].map(mapper_3)



mapper_4 = {'short': 0, 'okay': 1, 'good':2, 'long':3 }

data['NlengthD'] = data['NlengthD'].map(mapper_4)



mapper_5 = {'Master': 0, 'Miss': 1, 'Mr':2, 'Mrs':3,'Rare':4 }

data['Title'] = data['Title'].map(mapper_5)
data.head()
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['IsAlone'] = 0

data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

del data['SibSp']

del data['Parch']

del data['FamilySize'] 
data = pd.get_dummies(data, columns = ["Embarked", "Fare", "Pclass", "Title", "Cat_Age", "Deck", "NlengthD"])
data.head()
# Separating data into predictors (X) and target (Y)

X = data.loc[train_indexes][data.columns.difference(['Survived'])]

Y = data['Survived'][train_indexes]
# Feature Selection with RFE



from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression



model = LogisticRegression()

rfe = RFE(model, 20)

fit = rfe.fit(X, Y)

selected_feature = fit.support_



print("Num Features: %d", fit.n_features_) 

print("Selected Features: %s",  fit.support_)

print("Feature Ranking: %s", fit.ranking_) 
# Dropping unimportant features

col_to_drop=[]

for i in range(len(X.columns)-1):

    if selected_feature[i] == False:

        col_to_drop.append(i)



X.drop(X.iloc[:, col_to_drop], axis=1, inplace = True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()

svc_lin = SVC(kernel='linear')

knn = KNeighborsClassifier(n_neighbors = 3)

decision_tree = DecisionTreeClassifier(max_depth=6 )

random_forest = RandomForestClassifier(max_depth=6, n_estimators=10)

extra_trees = ExtraTreesClassifier(max_depth=6,n_estimators=10)

gbc = GradientBoostingClassifier()

ada = AdaBoostClassifier()



models = [logreg, svc_lin, knn, 

              decision_tree, random_forest, extra_trees, gbc, ada]
# Folding

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
np.set_printoptions(suppress=True)



number_models = len(models)

folds = 5



Chart_1 = np.zeros((len(train_indexes),number_models))

Chart_2 = np.zeros((len(prediction_indexes), folds))

Chart_3 = np.zeros((len(prediction_indexes), number_models))



X_pred = data.loc[prediction_indexes][data.columns.difference(['Survived'])]

X_pred.drop(X_pred.iloc[:, col_to_drop], axis=1, inplace = True)
from sklearn.metrics import accuracy_score



scores = pd.DataFrame(columns= ['Model','Accuracy_valid', 'Accuracy_train'])



for i, model in enumerate(models):

    for j, (train_index, test_index) in enumerate(sss.split(X, Y)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        

        # if I had a continuous feature that I needed to scale, I would place it here.

        # sc_1 = StandardScaler().fit(X_train["Fare"])

        # X_train["Fare"] = sc_1.transform(X_train["Fare"])

        #X_test["Fare"] = sc_1.transform(X_test["Fare"]) 

        #data["Fare"][prediction_indexes]=sc_1.transform(data["Fare"][prediction_indexes])

        

    

        model.fit(X_train, Y_train)

        Chart_1[test_index, i-1] = model.predict(X_test)

        

        Chart_2[ :, j-1] = model.predict(X_pred)

        Chart_3[ :, i-1] = Chart_2.mean(axis= 1)

        name = str(model).rsplit('(', 1)[0]

        accuracy_val = np.average(cross_val_score(model, X_test, Y_test, scoring= "accuracy"))

        accuracy_train = np.average(cross_val_score(model, X_train, Y_train, scoring= "accuracy"))

        scores = scores.append({'Model': name,'Accuracy_valid': accuracy_val, 'Accuracy_train': accuracy_train}, ignore_index=True)



scores.set_index("Model") 

scores.plot(x= 'Model', y = 'Accuracy_valid', kind='bar', title='Scores' )

scores.plot(x= 'Model', y = 'Accuracy_train', kind='bar', title='Scores' )
stacking = LogisticRegression().fit(Chart_1, Y)

Y_submit = stacking.predict(Chart_3)
Y_submit = Y_submit.astype(int)

submission = pd.DataFrame({ 'PassengerId': prediction_indexes,

                            'Survived': Y_submit })

submission.to_csv("submission__stack.csv", index=False)
