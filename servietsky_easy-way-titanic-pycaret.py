!pip install pycaret



from pycaret.classification import *

import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport 



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
report_train = ProfileReport(train)

report_train
report_test = ProfileReport(test)

report_test
train_tmp = train.copy()

test_tmp = test.copy()

train_tmp['type'] = 'train'

train_tmp.drop('Survived', axis = 1, inplace = True)

test_tmp['type'] = 'test'

data = pd.concat([train_tmp, test_tmp], ignore_index = True)



dfplot = data.groupby(['type','Pclass']).count()['PassengerId'].to_frame().reset_index()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Pclass'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),

              1, 1)

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Pclass'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),

              1, 2)

fig.update_layout(

    title_text="Pclass")

fig.show()
dfplot = data.groupby(['type','Sex']).count()['PassengerId'].to_frame().reset_index()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Sex'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),

              1, 1)

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Sex'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),

              1, 2)

fig.update_layout(

    title_text="Sex")

fig.show()
dfplot = data.groupby(['type','SibSp']).count()['PassengerId'].to_frame().reset_index()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['SibSp'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),

              1, 1)

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['SibSp'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),

              1, 2)

fig.update_layout(

    title_text="SibSp")

fig.show()
dfplot = data.groupby(['type','Embarked']).count()['PassengerId'].to_frame().reset_index()

fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Embarked'], values=dfplot[dfplot['type'] == 'test']['PassengerId'], name="test"),

              1, 1)

fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Embarked'], values=dfplot[dfplot['type'] == 'train']['PassengerId'], name="train"),

              1, 2)

fig.update_layout(

    title_text="Embarked")

fig.show()
fig = px.violin(train, y="Age", x="Survived", color="Sex", box=True, points="all", hover_data=['Age', 'Survived', 'Sex'])

fig.show()
fig = px.violin(train, y="Fare", x="Survived", color="Sex", box=True, points="all", hover_data=['Fare', 'Survived', 'Sex'])

fig.show()
env = setup(data = train, 

             target = 'Survived',

             numeric_imputation = 'mean',

             categorical_features = ['Sex','Embarked'], 

             ignore_features = ['Name','Ticket','Cabin'],

             silent = True,

            remove_outliers = True,

            normalize = True)
compare_models()
xgb = create_model('xgboost')
tuned_xgb = tune_model('xgboost')
plot_model(estimator = xgb, plot = 'learning')
plot_model(estimator = xgb, plot = 'feature')
plot_model(estimator = xgb, plot = 'auc')
plot_model(estimator = xgb, plot = 'confusion_matrix')
interpret_model(xgb)
predictions = predict_model(xgb, data=test)

predictions.head()
submission['Survived'] = round(predictions['Score']).astype(int)

submission.to_csv('submission.csv',index=False)

submission.head(10)