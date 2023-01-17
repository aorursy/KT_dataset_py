# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.manifold import TSNE

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.decomposition import PCA
from itertools import chain
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
train.shape,test.shape
train.head().T
train.isnull().values.any(),test.isnull().values.any()
train.vid_id.nunique(),test.vid_id.nunique()
hist_data = [train['start_time_seconds_youtube_clip'].values]
group_labels = ['distplot']

fig = ff.create_distplot(hist_data, group_labels,bin_size=20)
py.offline.iplot(fig, filename='Distplot of start time')
hist_data = [train['end_time_seconds_youtube_clip'].values]
group_labels = ['distplot']

fig = ff.create_distplot(hist_data, group_labels, bin_size=20)
py.offline.iplot(fig, filename='Distplot of end time')
train['length_audio']=pd.DataFrame(train['end_time_seconds_youtube_clip']-train['start_time_seconds_youtube_clip'])
train['length_audio'].value_counts()
train.loc[train.length_audio<10]['is_turkey'].value_counts()
train['is_turkey'].value_counts().keys()
data = [go.Bar(
            x=train['is_turkey'].value_counts().keys(),
            y=train['is_turkey'].value_counts().values
    )]
layout=go.Layout(title='Distribution of target variable')

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(data, filename='Distribution of target variable')
embeddings=train.audio_embedding
embeddings=embeddings.apply(lambda x:list(chain.from_iterable(x)))
embeddings_df=pd.DataFrame(list(embeddings))
embeddings_df.head()
embeddings_df=embeddings_df.fillna(-1)
pca = PCA(n_components=50)
pca_result = pca.fit_transform(embeddings_df.values)
var_exp=pca.explained_variance_ratio_
cum_var_exp=np.cumsum(pca.explained_variance_ratio_)
trace1 = go.Bar(
        x=['PC %s' %i for i in range(1,20)],
        y=var_exp,
        showlegend=False)

trace2 = go.Scatter(
        x=['PC %s' %i for i in range(1,20)], 
        y=cum_var_exp,
        name='cumulative explained variance')

data = [trace1, trace2]

layout=go.Layout(
        yaxis=go.layout.YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
trace = go.Scatter(
    x = pca_result[:,0],
    y = pca_result[:,1],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = train.is_turkey.values,
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
    )
)
data = [trace]

layout = go.Layout(title = 'PCA (Principal Component Analysis)',
              hovermode= 'closest',
              yaxis=go.layout.YAxis(title='Principal Component 2',zeroline=False),
              xaxis=go.layout.XAxis(title='Principal Component 1',zeroline=False),
              showlegend= True
             )

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='PCA')
tsne = TSNE()
tsne_results = tsne.fit_transform(embeddings_df.values) 
trace = go.Scatter(
    x = tsne_results[:,0],
    y = tsne_results[:,1],
    mode = 'markers',
    marker = dict(
        size = 8,
        color = train.is_turkey.values,
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
    )
)
data = [trace]

layout = go.Layout(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',
              hovermode= 'closest',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False),
              showlegend= True
             )

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='TSNE')
import lightgbm as lgb
lgb_params = {
    'learning_rate': 0.01,
    'max_depth': 7,
    'num_leaves': 40, 
    'objective': 'binary',
    'num_class':1,
    'tree_learner':'voting',
    'metric':'auc',
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'max_bin': 100
}
dtrain_lgb = lgb.Dataset(embeddings_df, label=train.is_turkey)
cv_result_lgb = lgb.cv(lgb_params, 
                       dtrain_lgb, 
                       num_boost_round=3000, 
                       nfold=5, 
                       stratified=True, 
                       early_stopping_rounds=50, 
                       verbose_eval=100, 
                       show_stdv=True)
num_boosting_round=len(list(cv_result_lgb.values())[0])
model_lgb = lgb.train(lgb_params, dtrain_lgb, num_boost_round=num_boosting_round)
test_embeddings=test.audio_embedding.apply(lambda x:list(chain.from_iterable(x)))
test_embeddings_df=pd.DataFrame(list(test_embeddings))
test_embeddings_df=test_embeddings_df.fillna(-1)
test_preds=model_lgb.predict(test_embeddings_df)
sub=pd.read_csv('../input/sample_submission.csv')
sub.is_turkey=test_preds
sub.to_csv('submission.csv',index=False)
