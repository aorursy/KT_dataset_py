#library

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

import seaborn as sn

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import plotly.graph_objects as go
#dataset

data_load = pd.read_csv('/kaggle/input/glaucoma-dataset/GlaucomaM.csv')
data_load.head()
data_load.isnull().sum()
le = LabelEncoder()
data_load.Class = le.fit_transform(data_load.Class)
data_load['Class']
model_params = {

    'svm': {

        'model': svm.SVC(gamma='auto'),

        'params' : {

            'C': [1,10,20],

            'kernel': ['rbf','linear']

        }  

    },

    'random_forest': {

        'model': RandomForestClassifier(),

        'params' : {

            'n_estimators': [1,5,10]

        }

    },

    'logistic_regression' : {

        'model': LogisticRegression(solver='liblinear',multi_class='auto'),

        'params': {

            'C': [1,5,10]

        }

    }

}   

pd.DataFrame(model_params)
scores = []



for model_name, mp in model_params.items():

    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)

    clf.fit(data_load.drop('Class',axis='columns'), data_load.Class)

    scores.append({

        'model': model_name,

        'best_score': clf.best_score_,

        'best_params': clf.best_params_

    })

    

df = pd.DataFrame(scores,columns=['model','best_score','best_params'])

df
from sklearn.model_selection import train_test_split
X = data_load.drop('Class', axis='columns')

y = data_load.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)


model = SVC(C=1.0,kernel='linear')
model.fit(X_train, y_train)
model.score(X_test, y_test)
classes1 = {

    0:'Normal',

    1:'Gulcoma',

}
y_predicted = model.predict(X_test)
y_predicted
classes1[y_predicted[3]]
cm = confusion_matrix(y_test, y_predicted)

cm
fig = go.Figure(data=go.Heatmap(

                   z=cm,

                   x=['Normal','Glucoma'],

                   y=['Normal','Glucoma'],

                   hoverongaps = False))

fig.show()