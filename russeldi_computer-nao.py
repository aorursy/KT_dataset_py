!pip install pycaret==1.0.0

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
        
test = pd.read_csv("../input/comnao/test_nao.csv")
train = pd.read_csv("../input/comnao/train_nao.csv")
report_train = ProfileReport(train)
report_train
report_test = ProfileReport(test)
report_test
train_tmp = train.copy()
test_tmp = test.copy()
train_tmp['type'] = 'train'
train_tmp.drop('已知标签', axis = 1, inplace = True)
test_tmp['type'] = 'test'
data = pd.concat([train_tmp, test_tmp], ignore_index = True)

dfplot = data.groupby(['type','已知标签']).count()['ID'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['已知标签'], values=dfplot[dfplot['type'] == 'test']['ID'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['已知标签'], values=dfplot[dfplot['type'] == 'train']['ID'], name="train"),
              1, 2)
fig.update_layout(
    title_text="已知标签")
fig.show()
dfplot = data.groupby(['type','Alpha']).count()['ID'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Alpha'], values=dfplot[dfplot['type'] == 'test']['ID'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Alpha'], values=dfplot[dfplot['type'] == 'train']['ID'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Alpha")
fig.show()
dfplot = data.groupby(['type','Beta']).count()['ID'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Beta'], values=dfplot[dfplot['type'] == 'test']['ID'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Beta'], values=dfplot[dfplot['type'] == 'train']['ID'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Beta")
fig.show()
dfplot = data.groupby(['type','Theta']).count()['ID'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Theta'], values=dfplot[dfplot['type'] == 'test']['ID'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Theta'], values=dfplot[dfplot['type'] == 'train']['ID'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Theta")
fig.show()
dfplot = data.groupby(['type','Delta']).count()['ID'].to_frame().reset_index()
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Test', 'Train'])
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'test']['Delta'], values=dfplot[dfplot['type'] == 'test']['ID'], name="test"),
              1, 1)
fig.add_trace(go.Pie(labels=dfplot[dfplot['type'] == 'train']['Delta'], values=dfplot[dfplot['type'] == 'train']['ID'], name="train"),
              1, 2)
fig.update_layout(
    title_text="Delta")
fig.show()
fig = px.violin(train, y="Theta", x="Alpha", color="已知标签", box=True, points="all", hover_data=['Theta', 'Alpha', '已知标签'])
fig.show()
fig = px.violin(train, y="Theta", x="Delta", color="已知标签", box=True, points="all", hover_data=['Theta', 'Delta', '已知标签'])
fig.show()
fig = px.violin(train, y="Alpha", x="已知标签", color="已知标签", box=True, points="all", hover_data=['Alpha', '已知标签', '已知标签'])
fig.show()
fig = px.violin(train, y="Beta", x="已知标签", color="已知标签", box=True, points="all", hover_data=['Beta', '已知标签', '已知标签'])
fig.show()
fig = px.violin(train, y="Theta", x="已知标签", color="已知标签", box=True, points="all", hover_data=['Theta', '已知标签', '已知标签'])
fig.show()
fig = px.violin(train, y="Delta", x="已知标签", color="已知标签", box=True, points="all", hover_data=['Delta', '已知标签', '已知标签'])
fig.show()
env = setup(data = train, 
             target = '已知标签',
             numeric_imputation = 'mean',
             categorical_features = ['Alpha','Beta','Theta','Delta'], 
            ignore_features = ['ID'],
            silent = True,
            remove_outliers = True,
            normalize = True)
compare_models()
xgb = create_model('xgboost')
tuned_rf = tune_model('rf')
tuned_xgb = tune_model('xgboost')
plot_model(estimator = xgb, plot = 'learning')
plot_model(estimator = xgb, plot = 'feature')
plot_model(estimator = xgb, plot = 'auc')
plot_model(estimator = xgb, plot = 'confusion_matrix')
interpret_model(xgb)
predictions = predict_model(xgb, data=test)
predictions.head()
submission['已知标签'] = round(predictions['Score']).astype(int)
submission.to_csv('submission.csv',index=False)
submission.head(100)