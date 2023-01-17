from __future__ import print_function, division

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import networkx as nx

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py 

import plotly.graph_objs as go 

py.init_notebook_mode(connected=True)



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