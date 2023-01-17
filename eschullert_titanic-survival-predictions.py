import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
test=pd.read_csv('../input/test.csv').set_index('PassengerId')

train=pd.read_csv('../input/train.csv').set_index('PassengerId')
train.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True, errors='ignore')

test.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True, errors='ignore')
train['Sex']=train['Sex'].apply(lambda d: 1 if d=='female' else 0)

test['Sex']=test['Sex'].apply(lambda d: 1 if d=='female' else 0)
train['Embarked'].value_counts()
train['EmbarkedS']=train['Embarked'].apply(lambda d: 1 if d=='S' else 0)

train['EmbarkedC']=train['Embarked'].apply(lambda d: 1 if d=='C' else 0)

train['EmbarkedQ']=train['Embarked'].apply(lambda d: 1 if d=='Q' else 0)

train.drop(columns='Embarked', inplace=True)
test['EmbarkedS']=test['Embarked'].apply(lambda d: 1 if d=='S' else 0)

test['EmbarkedC']=test['Embarked'].apply(lambda d: 1 if d=='C' else 0)

test['EmbarkedQ']=test['Embarked'].apply(lambda d: 1 if d=='Q' else 0)

test.drop(columns='Embarked', inplace=True)
train.fillna(train.mean(), inplace=True)

test.fillna(test.mean(), inplace=True)
X_train=train.drop(columns='Survived')

y_train=train['Survived']
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
modelos = [('neigh',KNeighborsClassifier()),

           ('svc',SVC()),

           ('Gauss',GaussianProcessClassifier()),

           ('dTree',DecisionTreeClassifier()),

           ('forest',RandomForestClassifier(n_estimators=100)),

           ('adaBoost',AdaBoostClassifier()),

           ('gaussNB',GaussianNB()),

           ('QDA',QuadraticDiscriminantAnalysis()),

           ('neural',MLPClassifier(activation='tanh', solver='lbfgs', alpha=0.001, learning_rate='adaptive', learning_rate_init=0.01))]
model_scores = {}
from sklearn.model_selection import cross_val_score
for modelo in modelos:

    score = abs(cross_val_score(modelo[1], X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)).mean()

    model_scores[modelo[0]]=score
%matplotlib inline

lst = sorted(model_scores.items())

x, y = zip(*lst)

plt.plot(x,y)

plt.show()
forest=RandomForestClassifier(n_estimators=100)

adaBoost=AdaBoostClassifier()

neural=MLPClassifier(activation='tanh', solver='lbfgs', alpha=0.001, learning_rate='adaptive', learning_rate_init=0.01)
forest.fit(X_train, y_train);

adaBoost.fit(X_train, y_train);

neural.fit(X_train, y_train);
forestp=forest.predict(test)

adap=adaBoost.predict(test)

neuralp=neural.predict(test)
predicciones=pd.DataFrame(data={'PassengerId':test.index, 'forest':forestp, 'ada':adap, 'neural':neuralp}).set_index('PassengerId')
predicciones['Survived']=predicciones.apply(lambda s: round(s.mean()), axis=1)
predicciones.drop(columns=['forest', 'ada', 'neural'], inplace=True)
importance_features = pd.DataFrame({'Ada Boost':adaBoost.feature_importances_,

                                    'Random Forest':forest.feature_importances_}, index=X_train.columns)
plt.barh(importance_features.sort_values(by='Ada Boost').index, importance_features['Ada Boost'].sort_values())

plt.title('Ada Boost')

plt.show()
plt.barh(importance_features.sort_values(by='Random Forest').index, importance_features['Random Forest'].sort_values())

plt.title('Random Forest')

plt.show()
predicciones.to_csv('pred.csv')