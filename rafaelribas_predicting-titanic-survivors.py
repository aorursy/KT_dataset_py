##Chapter 1 - Data Explore##

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier



from sklearn.metrics import accuracy_score



from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import GridSearchCV







train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')





train.info()

print ("------------------------")

test.info()



train.describe()

train.describe(include=['O'])



##Chapter 2 - Descriptive Analysis##

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train.corr(), vmax=.8, square=True);



sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)



sns.factorplot('Pclass','Survived',order=[1,2,3], data=train,size=8,aspect=3)



facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()



##Chapter 3 - Feature Engineer##

train['Cabin'] = train['Cabin'].fillna('M')

train['Cabin'] = train.Cabin.str.extract('([A-Za-z])', expand=False)

train['Cabin'] = train['Cabin'].map( {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E':0, 

                                            'F':0, 'G':0, 'T':0, 'M':1} ).astype(int)





most_freq = train.Embarked.dropna().mode()[0]

train['Embarked'] = train['Embarked'].fillna(most_freq)



train['Age'] = train['Age'].fillna(-1) 

train.loc[(train.Pclass == 1) & (train.Age == -1),'Age'] = 37

train.loc[(train.Pclass == 2) & (train.Age == -1),'Age'] = 29

train.loc[(train.Pclass == 3) & (train.Age == -1),'Age'] = 24



train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'] = train['Title'].replace(['Lady', 'Countess', 'Dona'],'Royalty')

train['Title'] = train['Title'].replace(['Mme'], 'Mrs')

train['Title'] = train['Title'].replace(['Mlle','Ms'], 'Miss')

train['Title'] = train['Title'].replace(['Capt', 'Col', 'Major','Rev'], 'Officer')

train['Title'] = train['Title'].replace(['Jonkheer', 'Don','Sir'], 'Royalty')

train.loc[(train.Sex == 'male')   & (train.Title == 'Dr'),'Title'] = 'Mr'

train.loc[(train.Sex == 'female') & (train.Title == 'Dr'),'Title'] = 'Mrs'



train.loc[(train.Age < 16),'Sex'] = 'Child'



train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



train.loc[(train['FamilySize'] == 1), 'IsAlone'] = 1

train.loc[(train['FamilySize'] > 1), 'IsAlone'] = 0



train['Name_Len'] = train['Name'].apply(lambda x: len(x))



train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))



Embark_dummies_train  = pd.get_dummies(train['Embarked'])

train2 = train.join(Embark_dummies_train)



Sex_dummies_train  = pd.get_dummies(train2['Sex'])

train2 = train2.join(Sex_dummies_train)



Title_dummies_train  = pd.get_dummies(train2['Title'])

train2 = train2.join(Title_dummies_train)



train2['Age*Class'] = train2.Age * train2.Pclass



##Chapter 4 - Modelling##

train_features = train.drop(["Survived","PassengerId","SibSp","Parch"], axis=1)

train_label = train["Survived"]



cv = ShuffleSplit(n_splits=20, test_size=0.30, random_state=0)

def grid_search_model(feature, label, model, parameters, cv):

    CV_model = GridSearchCV(estimator=model, param_grid=parameters, cv=cv)

    CV_model.fit(feature, label)

    CV_model.cv_results_

    print("Best Score:", CV_model.best_score_," / Best parameters:", CV_model.best_params_)

    

def predict_model(feature, label, model, Xtest, submit_name):

    model.fit(feature, label)

    Y_pred  = model.predict(Xtest)

    score   = cross_val_score(model, feature, label, cv=cv)



    submission = pd.DataFrame({

            "PassengerId": test["PassengerId"],

            "Survived": Y_pred

        })

    submission.to_csv(submit_name, index=False)

    

    return score



## Random Forest ##

param_range = (np.linspace(50, 400, 6)).astype(int)

param_leaf = (np.linspace(4, 20, 6)).astype(int)

param_split = (np.linspace(10, 30, 6)).astype(int)

param_grid = {'n_estimators':param_range, 'min_samples_leaf':param_leaf, 'min_samples_split':param_split}



grid_search_model(train_features, train_label, RandomForestClassifier(random_state=0), param_grid, cv)



Random_Forest = RandomForestClassifier(n_estimators=300, random_state =0, min_samples_split = 10, min_samples_leaf = 4)

Random_Forest.fit(train_features,train_label)

Random_Forest_Pred = Random_Forest.predict(test)



RF_submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Random_Forest_Pred

    })



RF_submission.to_csv("RF_titanic_submission.csv", index=False)



###########################################################