from pandas import read_csv
from numpy import array
from collections import Counter

from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import preprocessing

import plotly as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)
train_data = read_csv('../input/train.csv')

train_len = len(train_data)
print("{0} Data Has Been Loaded!".format(train_len))
print(train_data.columns)
gender_of_passenger = train_data['Sex']
colors = ['rgb(8,48,107)', 'rgb(158,202,225)']

gender_count_dict = Counter(gender_of_passenger)

gender = list(gender_count_dict.keys())
gender_count = list(gender_count_dict.values())

layout = {
    'title' : 'Gender Of Passengers',
}

trace = go.Pie(
    labels=gender, 
    values=gender_count,
    textinfo='value',
    textfont=dict(
        size=20, 
        color='rgb(255,255,255)'),
    marker=dict(colors=colors, 
        line=dict(color='rgb(0,0,0)', 
        width=2)),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
age_of_passengers = train_data['Age']
age_of_passengers = age_of_passengers.dropna(axis=0)

age_count_dict = Counter(age_of_passengers)

age = list(age_count_dict.keys())
age_count = list(age_count_dict.values())

layout = {
    'title' : 'Age Of Passengers',
    'xaxis' : {
        'title' : 'Age',
    },
    'yaxis' : {
        'title' : 'Sum Of Age',
    },
}

trace = go.Bar(
    x = age,
    y = age_count,
    marker = dict(
      color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=2,
        )  
    ),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
class_of_passengers = train_data['Pclass']

class_count_dict = Counter(class_of_passengers)

clss = list(class_count_dict.keys())
clss_count = list(class_count_dict.values())

layout = {
    'title' : 'Class Of Passengers',
    'xaxis' : {
        'title' : 'Class',
    },
    'yaxis' : {
        'title' : 'Sum Of Passengers',
    },
}

trace = go.Bar(
    x = clss,
    y = clss_count,
    marker = dict(
      color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )  
    ),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
embarked_point = train_data["Embarked"]
embarked_point = embarked_point.dropna(axis=0)
colors = ['rgb(8,48,107)', 'rgb(158,202,225)', 'rgb(0,102,204)']

embarked_point_dict = Counter(embarked_point)

embarked = list(embarked_point_dict.keys())
embarked_count = list(embarked_point_dict.values())


for i in range(len(embarked)):
    if embarked[i] == 'S':
        embarked[i] = 'Southampton'
    elif embarked[i] == 'C':
        embarked[i] = 'Cherbourg'
    elif embarked[i] == 'Q':
        embarked[i] = 'Queenstown'

layout = {
    'title' : 'Gender Of Passengers',
}

trace = go.Pie(
    labels=embarked, 
    values=embarked_count,
    textinfo='value',
    textfont=dict(
        size=20, 
        color='rgb(255,255,255)'),
    marker=dict(colors=colors, 
        line=dict(color='rgb(0,0,0)', 
        width=2)),
)

fig = go.Figure(data = [trace], layout = layout)
py.offline.iplot(fig)
algoritms = ["Logistic Regression", "Decision Tree Classifier", "SVC", "KNeighbors Classifier"]
scores = []
predictions_list = []

y = train_data["Survived"]

parameters = ["Pclass", "Sex", "Age"]

x = train_data[parameters]

le = preprocessing.LabelEncoder()
x = x.apply(le.fit_transform)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=23)

Logistic_Regression = LR().fit(x_train,y_train)
Decision_Tree_Classifier = DTC().fit(x_train,y_train)
SVC = SVC().fit(x_train,y_train)
K_Neighbors_Classifier = KNC(3).fit(x_train,y_train)

scores.append(Logistic_Regression.score(x_train,y_train))
scores.append(Decision_Tree_Classifier.score(x_train,y_train))
scores.append(SVC.score(x_train,y_train))
scores.append(K_Neighbors_Classifier.score(x_train,y_train))

predictions_LR = Logistic_Regression.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_LR))
predictions_DTC = Decision_Tree_Classifier.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_DTC))
predictions_SVC = SVC.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_SVC))
predictions_KNC = K_Neighbors_Classifier.predict(x_test)
predictions_list.append(accuracy_score(y_test, predictions_KNC))

layout = {
    'title' : 'Percentage Of Success Of Algorithms And Test Result',
    'xaxis' : {
        'title' : 'Algoritms',
    },
    'yaxis' : {
        'title' : '%',
    },
}

trace1 = go.Bar(
    name = 'Train Set',
    x = algoritms,
    y = scores,
    marker = dict(
      color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=2,
        )  
    ),
)

trace2 = go.Bar(
    name = 'Test Set',
    x = algoritms,
    y = predictions_list,
    marker = dict(
      color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=2,
        )  
    ),
)

fig = go.Figure(data = [trace1, trace2], layout = layout)
py.offline.iplot(fig)