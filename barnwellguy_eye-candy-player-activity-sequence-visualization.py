%matplotlib notebook

import matplotlib.pyplot as plt

import matplotlib



import numpy as np

import scipy as sp

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)

pd.set_option('max_colwidth', 120)

   

from IPython.display import display

from tqdm.notebook import tqdm





from collections import defaultdict

import networkx as nx
def gb(df,key):

    groupby = df.groupby(key)

    return tqdm(groupby,total=groupby.ngroups)
print('numpy  :',np.__version__)

print('pandas :',pd.__version__)
dataTemp =  r'../input/data-science-bowl-2019/{}.csv'



train             = pd.read_csv(dataTemp.format('train'))

train_labels      = pd.read_csv(dataTemp.format('train_labels'))
assessments = [x for x in train.title.unique() if 'Assessment' in x]

worlds      = train.world.unique()

titles      = train.title.unique()
tupleCounts = defaultdict(lambda:0)

for insId,df in gb(train,'installation_id'):

    sequence  = df.groupby('game_session',sort=False)['title'].first().values

    for i in range(len(sequence)-1):

        tupleCounts[(sequence[i],sequence[i+1])] += 1            
worldSeqs = {}



worldSeqs['NONE'] = [train[train.world == 'NONE'].title.iloc[0]]



for world in np.setdiff1d(worlds,['NONE']):



    worldTitles      = train[train.world == world].title.unique()

    worldTupleCounts = {key:tupleCounts[key] for key in tupleCounts if key[0] in worldTitles and key[1] in worldTitles}

    seq              = [x for x in worldTitles if 'Level 1' in x]



    while len(seq) < len(worldTitles):

        

        # cheating start

        

        if seq[-1] == 'Fireworks (Activity)':

            seq.append('12 Monkeys')

            continue



        if seq[-1] == 'Air Show':

            seq.append('Treasure Map')

            continue

        

        otherTitles = np.setdiff1d(worldTitles,seq)

        if 'Level' in seq[-1]:

            otherTitles = [x for x in otherTitles if 'Level' not in x]

            

        # cheating end



            

        counts = [worldTupleCounts[(seq[-1],title)] for title in otherTitles]

        seq.append(otherTitles[np.argmax(counts)])

    

    assert len(seq) == len(np.unique(seq))

    

    print(world)

    print(seq,'\n')

    

    worldSeqs[world] = seq
from matplotlib.patches import Ellipse



title2type = train.groupby('title')['type'].first().to_dict()

type2color = {'Clip':'y','Game':'g','Assessment':'r','Activity':'c'}



#decide location of nodes

worldLocs = {'NONE'        :(-30,15),

             'TREETOPCITY' :(-30,0),

             'MAGMAPEAK'   :(0  ,15),

             'CRYSTALCAVES':(20 ,0),

             }

titleLocs = {}

for world in worlds:

    level = 0

    x     = 0

    for title in worldSeqs[world]:

        if 'Level' in title:

            level += 1

            x      = 0

        titleLocs[title] = (worldLocs[world][0]+x*6 + 1.0*level,worldLocs[world][1]-level*4)

        x += 1

            

def draw_nodes():

    

    plt.figure(figsize=(19,10))

    

    G = nx.Graph()

    G.add_nodes_from(titleLocs.keys())

    colors = []

    for node in G:

        colors.append(type2color[title2type[node]]) 

    nx.draw_networkx_nodes(G,pos=titleLocs,node_color=colors)    

    

    text = nx.draw_networkx_labels(G,{x:(y[0]+0.7,y[1]) for x,y in titleLocs.items()},{x:x for x in titleLocs})



    for _,t in text.items():

        t.set_rotation('20')   # rotation makes perfect

        t.set_ha('left')

        t.set_va('bottom')

        

    plt.xlim(-40,60)

    plt.ylim(-17,20)

    plt.tight_layout()

        



def draw_edges(tupleCounts,connectionstyle="arc3,rad=0.5",widthFunc=np.sqrt,width=5,cmap=plt.cm.Blues,alpha=0.9):

    

    DG = nx.DiGraph()

    DG.add_weighted_edges_from([(key[0],key[1],tupleCounts[key]) for key in tupleCounts])

    edges,weights = zip(*nx.get_edge_attributes(DG,'weight').items())

    weights       = np.array(weights)   

    nx.draw_networkx_edges(DG,pos=titleLocs,

                           connectionstyle=connectionstyle,               # curvy looks better

                           edgelist=edges,

                           edge_color=weights,                           # deeper color bigger count

                           horizontalalignment='left',

                           arrowsize=20,

                           width=widthFunc(weights/np.max(weights))*width,     # width is proportional to square root of count

                           edge_cmap=cmap,                       

                           alpha=alpha)

    

  

    r = 1.5

    for edge,weight in zip(edges,weights):

        if edge[0] == edge[1]:

            x,y = titleLocs[edge[0]]

            plt.gca().add_patch(Ellipse((x,y+0.5*r),

                                        width=1.5*r,height=r,

                                        lw    = widthFunc(weight/np.max(weights))*width,

                                        color = cmap(weight/np.max(weights)),

                                        fill= False,

                                        alpha=alpha))

    

    

    

    

    plt.tight_layout()

    

        

def draw(tupleCounts,plotTitle):

    draw_nodes()

    draw_edges(tupleCounts)

    plt.title(plotTitle)

    plt.tight_layout()

    

    from matplotlib.patches import FancyArrowPatch,Circle
topTupleCounts = {}

for title in titles:

    dests  = np.array([key[0] for key in tupleCounts if key[1]==title])

    counts = np.array([tupleCounts[(title,dest)] for dest in dests])



    argsort      = np.argsort(counts)

    sortedCounts = counts[argsort]

    

    n = np.where(np.cumsum(np.flip(sortedCounts)) > np.sum(counts)*0.7)[0][0]

    for i in argsort[~n:]:

        topTupleCounts[(title,dests[i])] = tupleCounts[(title,dests[i])]    

draw(topTupleCounts,'top 70% outgoing from each nodes')        
topTupleCounts = {}

for title in titles:

    if 'Assess' in title:

        origins  = np.array([key[0] for key in tupleCounts if key[1]==title])

        counts   = np.array([tupleCounts[(o,title)] for o in origins])



        argsort      = np.argsort(counts)

        sortedCounts = counts[argsort]



        n = np.where(np.cumsum(np.flip(sortedCounts)) > np.sum(counts)*0.99)[0][0]

        for i in argsort[~n:]:

            topTupleCounts[(origins[i],title)] = tupleCounts[(origins[i],title)]    

draw(topTupleCounts,'99% incoming to assessments')            
def getSeqLength(length):

    seq2assessment = defaultdict(lambda:0)

    for insId,df in gb(train,'installation_id'):

        sequence  = df.groupby('game_session',sort=False)['title'].first().values

        for i in [j for j in range(len(sequence)) if 'Assessment' in sequence[j]]:

            if i >= length:

                seq2assessment[tuple([sequence[i-length+j] for j in range(0,length+1)])] += 1    

    return seq2assessment
seq2assessment = getSeqLength(5)



keys    = list(seq2assessment.keys())

counts  = np.array([seq2assessment[key] for key in keys])

argsort = np.argsort(counts)



tupleCounts5 = {}

for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    for j in range(len(key)-1):

        tupleCounts5[(key[j],key[j+1])] = count

draw(tupleCounts5,'most common length-5 path')   



for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    print('{:30s} {:4d}/{:4d}'.format(key[-1],count,train[train.title==key[-1]].game_session.nunique()))
seq2assessment = getSeqLength(2)



keys    = list(seq2assessment.keys())

counts  = np.array([seq2assessment[key] for key in keys])

argsort = np.argsort(counts)



tupleCounts5 = {}

for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    for j in range(len(key)-1):

        tupleCounts5[(key[j],key[j+1])] = count

draw(tupleCounts5,'most common length-2 path')   



for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    print('{:30s} {:4d}/{:4d}'.format(key[-1],count,train[train.title==key[-1]].game_session.nunique()))
seq2assessment = getSeqLength(1)



keys    = list(seq2assessment.keys())

counts  = np.array([seq2assessment[key] for key in keys])

argsort = np.argsort(counts)



tupleCounts5 = {}

for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    for j in range(len(key)-1):

        tupleCounts5[(key[j],key[j+1])] = count

draw(tupleCounts5,'most common length-1 path')   



for i in range(5):

    key   = keys  [argsort[~i]]

    count = counts[argsort[~i]]

    print('{:30s} {:4d}/{:4d}'.format(key[-1],count,train[train.title==key[-1]].game_session.nunique()))
np.random.seed(84)

for title,title_train_labels in train_labels.groupby('title'):

    for iid in title_train_labels.installation_id.sample(5):

        draw_nodes()

        seq    = train[train.installation_id==iid].groupby('game_session',sort=False).title.first().values

        if len(seq)>1:

            tupleCounts = defaultdict(lambda:0)

            for i in range(len(seq)-1):

                tupleCounts[(seq[i],seq[i+1])] += 1

            draw_edges(tupleCounts,cmap=plt.cm.Wistia,alpha=0.7)

            plt.title(title + ' ' + iid)

            plt.tight_layout()

        else:

            plt.title(title + ' ' + iid + ', this kid directly took the assessment without doing anything else prior and after')

            plt.tight_layout()            
np.random.seed(85)

for title,title_train_labels in train_labels.groupby('title'):

    for ri,row in title_train_labels.sample(5).iterrows():

        

        installation_id = row.installation_id

        game_session    = row.game_session

        

        draw_nodes()

        seq    = train[train.installation_id==installation_id].groupby('game_session',sort=False).title.first()

        seq    = seq.values[:list(seq.index).index(game_session)+1]

        

        

        if len(seq)>1:

            tupleCounts = defaultdict(lambda:0)

            for i in range(len(seq)-1):

                tupleCounts[(seq[i],seq[i+1])] += 1

            draw_edges(tupleCounts,cmap=plt.cm.Wistia,alpha=0.7)

            plt.title(title + ' ' + iid)

            plt.tight_layout()

        else:

            plt.title(title + ' ' + iid + ', this kid directly took the assessment without doing anything else prior and after')

            plt.tight_layout()            

    