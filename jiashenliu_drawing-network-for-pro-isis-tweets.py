import sklearn

import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plt

%matplotlib inline

import re
df = pd.read_csv('../input/tweets.csv')
print(df.shape)
index = []

for i in range(len(df['tweets'])):

    if '@' in df['tweets'][i]:

        index.append(i)

        

With_mention = df.iloc[index,:].reset_index(drop=True)

With_mention['Tagged_User'] = With_mention['tweets'].apply(lambda x: re.findall(r'@([A-Za-z0-9_]+)',str(x)))

User = With_mention['username'].unique()

With_mention['Tagged_User_Co'] = With_mention['Tagged_User'].apply(lambda x: list(set(x).intersection(User)) )

With_mention['Co_length'] = With_mention['Tagged_User_Co'].apply(lambda x: len(x))

With_mention_2 = With_mention[With_mention['Co_length']>0].reset_index(drop=True)
for i in range(len(With_mention['tweets'])):

    frame = With_mention.iloc[i,:]

    for j in range(len(frame['Tagged_User'])):

        tmp = pd.DataFrame({'User':[frame['username']],'Mentions':[frame['Tagged_User'][j]],'Time': [frame['time']],'User_numberstatuses':[frame['numberstatuses']],'User_followers':[frame['followers']],'Weight': [1]})

        if i==0 and j==0:

            Mention_net = tmp

        else:

            Mention_net = Mention_net.append(tmp, ignore_index=True)

    
for i in range(len(With_mention_2['tweets'])):

    frame = With_mention_2.iloc[i,:]

    for j in range(len(frame['Tagged_User_Co'])):

        Mentioned_follower = list(df[df['username']==frame['Tagged_User_Co'][j]]['followers'])[0]

        Mehtioned_statuses = list(df[df['username']==frame['Tagged_User_Co'][j]]['numberstatuses'])[0] 

        tmp = pd.DataFrame({'Mentioned_statuses':[Mehtioned_statuses],'Mentioned_followers':[Mentioned_follower],'User':[frame['username']],'Mentions':[frame['Tagged_User_Co'][j]],'Time': [frame['time']],'User_numberstatuses':[frame['numberstatuses']],'User_followers':[frame['followers']],'Weight': [1]})

        if i==0 and j==0:

            Mention_net_2 = tmp

        else:

            Mention_net_2 = Mention_net_2.append(tmp, ignore_index=True)

    
Mention_net = Mention_net[Mention_net['User']!=Mention_net['Mentions']]

Mention_net_2 = Mention_net_2[Mention_net_2['User']!=Mention_net_2['Mentions']]
Mention_net = Mention_net.reset_index(drop=True)

Mention_net_2 = Mention_net_2.reset_index(drop=True)
Mention_net.head(5)
In_degree = Mention_net.groupby(by=['Mentions'],as_index=False)['Weight'].sum()

Out_degree= Mention_net.groupby(by=['User'],as_index=False)['Weight'].sum()
In_degree = pd.DataFrame(In_degree).sort_values(by='Weight',ascending=False).reset_index(drop=True)

Out_degree = pd.DataFrame(Out_degree).sort_values(by='Weight',ascending=False).reset_index(drop=True)
print( 'Most mentioned user is '+str(In_degree['Mentions'][0])+' with ' + str(In_degree['Weight'][0])+' times mentioned by the other users.')

print( 'Most active user is '+str(Out_degree['User'][0])+' with ' + str(Out_degree['Weight'][0])+' times mentioning other users.')
In_degree_2 = Mention_net_2.groupby(by=['Mentions'],as_index=False)['Weight'].sum()

Out_degree_2= Mention_net_2.groupby(by=['User'],as_index=False)['Weight'].sum()
In_degree_2 = pd.DataFrame(In_degree_2).sort_values(by='Weight',ascending=False).reset_index(drop=True)

Out_degree_2 = pd.DataFrame(Out_degree_2).sort_values(by='Weight',ascending=False).reset_index(drop=True)
print( 'Most mentioned user is '+str(In_degree_2['Mentions'][0])+' with ' + str(In_degree['Weight'][0])+' times mentioned by the other users.')

print( 'Most active user is '+str(Out_degree_2['User'][0])+' with ' + str(Out_degree['Weight'][0])+' times mentioning other users.')
network1 = Mention_net.iloc[:,[0,2,5]]

network1 = network1.groupby(by=['Mentions','User'],as_index='False')['Weight'].sum().reset_index(name='Weight')

network1= pd.DataFrame(network1).sort_values(by='Weight',ascending=False).reset_index(drop=True)

network1 = network1[network1['Weight']>20]

import networkx as nx
print('The 10 most frequent tagged user pairs are ')

network1.iloc[0:9,:]
G = nx.Graph()

for i in range(len(network1['User'])):

    G.add_edge(network1['User'][i],network1['Mentions'][i],weight=network1['Weight'][i])

plt.figure(1)

plt.figure(figsize=(14,14))

d=d = nx.degree(G)

nx.draw_circular(G,node_color='g', edge_color='#909090', node_size=[v*100 for v in d.values()],with_labels=True)

plt.axis('equal')
network2 = Mention_net_2.iloc[:,[2,4,7]]

network2 = network2.groupby(by=['Mentions','User'],as_index='False')['Weight'].sum().reset_index(name='Weight')

network2= pd.DataFrame(network2).sort_values(by='Weight',ascending=False).reset_index(drop=True)

network2 = network2[network2['Weight']>10]

import networkx as nx
print('The 10 most frequent tagged user pairs within the ISIS pro are ')

network2.iloc[0:9,:]
G = nx.Graph()

for i in range(len(network2['User'])):

    G.add_edge(network2['User'][i],network2['Mentions'][i],weight=network2['Weight'][i])

plt.figure(2)

plt.figure(figsize=(14,14))

d=d = nx.degree(G)

nx.draw_circular(G,node_color='g', edge_color='#909090', node_size=[v*100 for v in d.values()],with_labels=True)

plt.axis('equal')