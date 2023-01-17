import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import preprocessing
from xgboost import XGBClassifier
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier

def extractDeck(x):
    if str(x) != "nan":
        return str(x)[0]
    else :
        return

#Import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combine = train.drop(["Survived"], axis=1).append(test).drop(["PassengerId", "Ticket"], axis=1)
target = train['Survived']

#Feature preprocessing
combine["hasParents"] = combine["Parch"].apply(lambda x : (x>0)*1)
combine["hasSibs"] = combine["SibSp"].apply(lambda x : (x>0)*1)
combine["title"] = combine['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combine["Deck"] = combine['Cabin'].apply(extractDeck)
combine.drop(["Parch", "SibSp", "Cabin", "Name"], axis=1)

#Turning categorical to integer
combine['Sex'] = combine['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
#One hot encoding on Embarked
combine['Embarked'].fillna('S', inplace = True)
#combine = pd.get_dummies(combine)#, columns = ['Embarked'])


#Fill the blank
combine['Age'].fillna(combine['Age'].dropna().median(), inplace = True)

#Turning age to ranges
combine.loc[(combine['Age'] <= 16), 'Age'] = 0 
combine.loc[(combine['Age'] > 16) & (combine['Age'] <= 32), 'Age'] = 1 
combine.loc[(combine['Age'] > 32) & (combine['Age'] <= 48), 'Age'] = 2 
combine.loc[(combine['Age'] > 48) & (combine['Age'] <= 64), 'Age'] = 3 
combine.loc[(combine['Age'] > 64), 'Age'] 


#Filling the blank
combine['Fare'].fillna(combine['Fare'].dropna().median(), inplace=True)

#Turning fare to ranges
combine.loc[ combine['Fare'] <= 7.91, 'Fare'] = 0
combine.loc[(combine['Fare'] > 7.91) & (combine['Fare'] <= 14.454), 'Fare'] = 1
combine.loc[(combine['Fare'] > 14.454) & (combine['Fare'] <= 31), 'Fare']   = 2
combine.loc[ combine['Fare'] > 31, 'Fare'] = 3
combine['Fare'] = combine['Fare'].astype(int)

combine["Pclass"]=combine["Pclass"].astype("str")
combine = pd.get_dummies(combine)

#Defining learning vectors
nb = train.shape[0]
X = combine[:nb]
y = target

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, train_size=0.9)

#XGBoost model tuning
model = XGBClassifier(booster='gbtree', silent=1, seed=0, base_score=0.5, subsample=0.75)
parameters = {'n_estimators':[75], #50,100
            'max_depth':[4],#1,10
            'gamma':[4],#0,6
            'max_delta_step':[1],#0,2
            'min_child_weight':[1], #3,5 
            'colsample_bytree':[0.55,0.6,0.65], #0.5,
            'learning_rate': [0.001,0.01,0.1]
            }
tune_model =  GridSearchCV(model, parameters, cv=3, scoring='accuracy')
tune_model.fit(X_train,y_train)
print('Best parameters :', tune_model.best_params_)
print('Results :', format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))


#Learn on the whole data
tune_model.fit(X, y)
Y_pred = tune_model.predict(combine[nb:])

#Submit the prediction
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

