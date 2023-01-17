!pip install -U vega_datasets notebook vega

from __future__ import print_function, division

import numpy as np
import pandas as pd


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook
pd.options.display.precision = 15

import time
import datetime
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from IPython.display import HTML
import json
import altair as alt

import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py 
import plotly.graph_objs as go 
py.init_notebook_mode(connected=True)

alt.renderers.enable('notebook')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(15)

import altair as alt
from altair.vega import v5
from IPython.display import HTML

# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey
def prepare_altair():
    """
    Helper function to prepare altair for working.
    """

    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION
    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
    noext = "?noext"
    
    paths = {
        'vega': vega_url + noext,
        'vega-lib': vega_lib_url + noext,
        'vega-lite': vega_lite_url + noext,
        'vega-embed': vega_embed_url + noext
    }
    
    workaround = f"""    requirejs.config({{
        baseUrl: 'https://cdn.jsdelivr.net/npm/',
        paths: {paths}
    }});
    """
    
    return workaround
    

def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped
           

@add_autoincrement
def render(chart, id="vega-chart"):
    """
    Helper function to plot altair visualizations.
    """
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )
df_classes = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
df_features = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
df_edgelist = pd.read_csv('/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
df_classes['class'].value_counts()
df_features.head()
# renaming columns
df_features.columns = ['id', 'time step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in range(72)]
df_features.head()
df_features['time step'].value_counts().sort_index().plot();
plt.title('Number of transactions in each time step');
# merge with classes
df_features = pd.merge(df_features, df_classes, left_on='id', right_on='txId', how='left')
plt.figure(figsize=(12, 8))
grouped = df_features.groupby(['time step', 'class'])['id'].count().reset_index().rename(columns={'id': 'count'})
sns.lineplot(x='time step', y='count', hue='class', data=grouped);
plt.legend(loc=(1.0, 0.8));
plt.title('Number of transactions in each time step by class');
bad_ids = df_features.loc[(df_features['time step'] == 37) & (df_features['class'] == '1'), 'id']
short_edges = df_edgelist.loc[df_edgelist['txId1'].isin(bad_ids)]
graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)
nx.draw(graph, cmap = plt.get_cmap('rainbow'), with_labels=True, pos=pos)
graph1 = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.Graph())
pos1 = nx.spring_layout(graph1)
nx.draw(graph1, cmap = plt.get_cmap('rainbow'), with_labels=False, pos=pos1)
# grouped = df_features.groupby(['time step', 'class'])['trans_feat_0'].mean().reset_index()
# chart = alt.Chart(grouped).mark_line().encode(
#     x=alt.X("time step:N", axis=alt.Axis(title='Time step', labelAngle=315)),
#     y=alt.Y('trans_feat_0:Q', axis=alt.Axis(title='Mean of trans_feat_0')),
#     color = 'class:N',
#     tooltip=['time step:O', 'trans_feat_0:Q', 'class:N']
# ).properties(title="Average trans_feat_0 in each time step by type", width=600).interactive()
# chart
plt.figure(figsize=(12, 8))
grouped = df_features.groupby(['time step', 'class'])['trans_feat_0'].mean().reset_index()
sns.lineplot(x='time step', y='trans_feat_0', hue='class', data=grouped);
plt.legend(loc=(1.0, 0.8));
plt.title('Average trans_feat_0 in each time step by type');
edges = pd.read_csv("/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
features = pd.read_csv("/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_features.csv",header=None)
classes = pd.read_csv("/kaggle/input/elliptic-data-set/elliptic_bitcoin_dataset/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")

len(edges),len(features),len(classes)
display(edges.head(5),features.head(5),classes.head(5))

tx_features = ["tx_feat_"+str(i) for i in range(2,95)]
agg_features = ["agg_feat_"+str(i) for i in range(1,73)]
features.columns = ["txId","time_step"] + tx_features + agg_features
features = pd.merge(features,classes,left_on="txId",right_on="txId",how='left')
features['class'] = features['class'].apply(lambda x: '0' if x == "unknown" else x)
features.groupby('class').size()
count_by_class = features[["time_step",'class']].groupby(['time_step','class']).size().to_frame().reset_index()
illicit_count = count_by_class[count_by_class['class'] == '1']
licit_count = count_by_class[count_by_class['class'] == '2']
unknown_count = count_by_class[count_by_class['class'] == "0"]
x_list = list(range(1,50))
fig = go.Figure(data = [
    go.Bar(name="Unknown",x=x_list,y=unknown_count[0],marker = dict(color = 'rgba(120, 100, 180, 0.6)',
        line = dict(
            color = 'rgba(120, 100, 180, 1.0)',width=1))),
    go.Bar(name="Licit",x=x_list,y=licit_count[0],marker = dict(color = 'rgba(246, 78, 139, 0.6)',
        line = dict(
            color = 'rgba(246, 78, 139, 1.0)',width=1))),
    go.Bar(name="Illicit",x=x_list,y=illicit_count[0],marker = dict(color = 'rgba(58, 190, 120, 0.6)',
        line = dict(
            color = 'rgba(58, 190, 120, 1.0)',width=1)))

])
fig.update_layout(barmode='stack')
py.iplot(fig)
bad_ids = features[(features['time_step'] == 32) & ((features['class'] == '1'))]['txId']
short_edges = edges[edges['txId1'].isin(bad_ids)]
graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)

edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='blue'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text=[]
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Transaction Type',
            xanchor='left',
            titleside='right',
            tickmode='array',
            tickvals=[0,1,2],
            ticktext=['Unknown','Illicit','Licit']
        ),
        line_width=2))
node_trace.text=node_text
node_trace.marker.color = pd.to_numeric(features[features['txId'].isin(list(graph.nodes()))]['class'])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title="Illicit Transactions",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
good_ids = features[(features['time_step'] == 32) & ((features['class'] == '2'))]['txId']
short_edges = edges[edges['txId1'].isin(good_ids)]
graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                 create_using = nx.DiGraph())
pos = nx.spring_layout(graph)

edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='blue'),
    hoverinfo='none',
    mode='lines')

node_x = []
node_y = []
node_text=[]
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            title='Transaction Type',
            xanchor='left',
            titleside='right',
            tickmode='array',
            tickvals=[0,1,2],
            ticktext=['Unknown','Illicit','Licit']
        ),
        line_width=2))
node_trace.text=node_text
node_trace.marker.color = pd.to_numeric(features[features['txId'].isin(graph.nodes())]['class'])

fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title="Licit Transactions",
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=True,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
fig.show()
data = features[(features['class']=='1') | (features['class']=='2')]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(15)
X = data[tx_features+agg_features]
y = data['class']
y = y.apply(lambda x: 0 if x == '2' else 1 )
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=15,shuffle=False)
reg = LogisticRegression().fit(X_train,y_train)
preds = reg.predict(X_test)
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("Logistic Regression")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
clf = RandomForestClassifier(n_estimators=50, max_depth=100,random_state=15).fit(X_train,y_train)
preds = clf.predict(X_test)
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("Random Forest Classifier")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
from torch.utils.data import Dataset, DataLoader
class LoadData(Dataset):
    
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.X.iloc[idx]
        features = np.array([features])
        label = y.iloc[idx]

        return features,label
traindata = LoadData(X_train,y_train)
train_loader = DataLoader(traindata,batch_size=128,shuffle=True)  
testdata = LoadData(X_test,y_test)
test_loader = DataLoader(testdata,batch_size=128,shuffle=False) 
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(165,50 )

        self.output = nn.Linear(50,1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):

        x = F.relu(self.hidden(x))

        x = self.out(self.output(x))
        
        return x

model = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion=nn.BCELoss()
n_epochs=10
for epoch in range(n_epochs):
        model.to('cuda')
        model.train()
        running_loss = 0.
        for data in train_loader:
            x,label=data
            x,label=x.cuda(),label.cuda()
            output = model.forward(x.float())
            output = output.squeeze()
            loss = criterion(output.float(), label.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_loader)}")
all_preds = []
for data in test_loader:
    x,labels = data
    x,labels = x.cuda(),labels.cuda()
    preds = model.forward(x.float())
    all_preds.extend(preds.squeeze().detach().cpu().numpy())

preds = pd.Series(all_preds).apply(lambda x: round(x))
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("MLP")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
embed_names = ["emb_"+str(i) for i in range(1,51)]
embeddings = pd.read_csv('/kaggle/input/ellipticemb50d/elliptic.emb',delimiter=" ",skiprows=1,header=None)
embeddings.columns = ['txId'] + ["emb_"+str(i) for i in range(1,51)]
data = features[(features['class']=='1') | (features['class']=='2')]
data = pd.merge(data,embeddings,how='inner')
X = data[tx_features+agg_features+embed_names]
y = data['class']
y = y.apply(lambda x: 0 if x == '2' else 1 )
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=15,shuffle=False)

reg = LogisticRegression().fit(X_train,y_train)
preds = reg.predict(X_test)
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("Logistic Regression")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
clf = RandomForestClassifier(n_estimators=50, max_depth=100,random_state=15).fit(X_train,y_train)
preds = clf.predict(X_test)
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("Random Forest Classifier")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
traindata = LoadData(X_train,y_train)
train_loader = DataLoader(traindata,batch_size=128,shuffle=True)  

testdata = LoadData(X_test,y_test)
test_loader = DataLoader(testdata,batch_size=128,shuffle=False)  

import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(215,50 )

        self.output = nn.Linear(50,1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):

        x = F.relu(self.hidden(x))

        x = self.out(self.output(x))
        
        return x

model = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion=nn.BCELoss()
n_epochs=10

for epoch in range(n_epochs):
        model.to('cuda')
        model.train()
        running_loss = 0.
        for data in train_loader:
            x,label=data
            x,label=x.cuda(),label.cuda()
            output = model.forward(x.float())
            output = output.squeeze()
            loss = criterion(output.float(), label.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(train_loader)}")
            
all_preds = []
for data in test_loader:
    x,labels = data
    x,labels = x.cuda(),labels.cuda()
    preds = model.forward(x.float())
    all_preds.extend(preds.squeeze().detach().cpu().numpy())

preds = pd.Series(all_preds).apply(lambda x: round(x))
prec,rec,f1,num = precision_recall_fscore_support(y_test,preds, average=None)
print("MLP")
print("Precision:%.3f \nRecall:%.3f \nF1 Score:%.3f"%(prec[1],rec[1],f1[1]))
micro_f1 = f1_score(y_test,preds,average='micro')
print("Micro-Average F1 Score:",micro_f1)
