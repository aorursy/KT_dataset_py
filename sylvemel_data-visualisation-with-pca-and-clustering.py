import numpy as np

import pandas as pd

pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',1000)



import missingno as msno



import itertools

import warnings

warnings.filterwarnings("ignore")



import io



from plotly.offline import init_notebook_mode, plot,iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization
df=pd.read_csv("../input/data.csv")

df.head(10)
best_players_per_position=df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Position','Name','Overall']]

best_players_per_position
forwards=['RF', 'ST', 'LW', 'LF', 'RS', 'LS', 'RM', 'LM','RW']

midfielders=['RCM','LCM','LDM','CAM','CDM','LAM','RDM','CM','RAM','CF']

defenders=['RCB','CB','LCB','LB','RB','RWB','LWB']

goalkeepers=['GK']



def pos2(position):

    if position in forwards:

        return 'Forward'

    

    elif position in midfielders:

        return 'Midfielder'

    

    elif position in defenders:

        return 'Defender'

    

    elif position in goalkeepers:

        return 'GK'

    

    else:

        return 'nan'



df["Position2"]=df["Position"].apply(lambda x: pos2(x))



df["Position2"].value_counts()
n_sne=2000

df_sne=df.loc[:n_sne]

skills_ratings = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',

                  'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',

                  'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',

                  'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

                  'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 

                  'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',

                  'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']



X=df_sne[skills_ratings]
from sklearn import (manifold, decomposition)



import time



time_start = time.time()

tsne = manifold.TSNE(n_components=2, verbose=1,perplexity=30, n_iter=1000)

X_tsne = tsne.fit_transform(X)



print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
bool_striker= df_sne["Position2"] == 'Forward'

bool_midfielder= df_sne["Position2"] == 'Midfielder'

bool_defender= df_sne["Position2"] == 'Defender'

bool_gk= df_sne["Position2"] == 'GK'



bool_crack=df_sne["Overall"] > 85



palette=['navy','red','#A2D5F2','orange','green','pink']  

data=[]



acp_striker =go.Scatter(x=X_tsne[bool_striker,0], y=X_tsne[bool_striker,1],name='Striker',

                      text=df_sne.loc[bool_striker,'Name'],

                      opacity=0.9,marker=dict(color=palette[2],size=5),mode='markers')



acp_midfielder =go.Scatter(x=X_tsne[bool_midfielder,0], y=X_tsne[bool_midfielder,1],name='Midfielder',

                      text=df_sne.loc[bool_midfielder,'Name'],

                      opacity=0.6,marker=dict(color=palette[1],size=5),mode='markers')



acp_defender =go.Scatter(x=X_tsne[bool_defender,0], y=X_tsne[bool_defender,1],name='Defender',

                      text=df_sne.loc[bool_defender,'Name'],

                      opacity=0.7,marker=dict(color=palette[3],size=5),mode='markers')



acp_gk =go.Scatter(x=X_tsne[bool_gk,0], y=X_tsne[bool_gk,1],name='GK',

                      text=df_sne.loc[bool_gk,'Name'],

                      opacity=0.4,marker=dict(color=palette[4],size=5),mode='markers')



acp_crack =go.Scatter(x=X_tsne[bool_crack,0], y=X_tsne[bool_crack,1],name='Top player',

                      text=df_sne.loc[bool_crack,'Name'],textfont=dict(family='sans serif',color='black',size=16),

                      opacity=0.9,mode='text')



data=[acp_striker,acp_midfielder,acp_defender,acp_gk,acp_crack]



layout = go.Layout(title="t-SNE - Fifa Players",titlefont=dict(size=40),

                autosize=False, width=1100,height=1100)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
n_acp=18000



# We exclude the GK and the weak players



df_acp=df.loc[:n_acp]

df_acp=df_acp[(df_acp["Position"]!='GK')&(df['Overall']>70)]



skills_ratings = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 

                  'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 

                  'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 

                  'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 

                  'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

                  'Marking', 'StandingTackle', 'SlidingTackle']



# We create the perfect player to see where it is located in our visualisation



MachineGunDict={'Name':'MachineGun','Overall':99}

for skills in skills_ratings:

    MachineGunDict[skills]=99

    

df_acp=df_acp.append(MachineGunDict,ignore_index=True)



X=df_acp[skills_ratings]
from sklearn import preprocessing

std_scale=preprocessing.StandardScaler().fit(X)

X_scaled=std_scale.transform(X)



from sklearn import decomposition

pca=decomposition.PCA(n_components=2)

pca.fit(X_scaled)



print (pca.explained_variance_ratio_)

print (pca.explained_variance_ratio_.cumsum())



X_projected=pca.transform(X_scaled)

print (X_projected.shape)



pcs=pca.components_
#Graph 1

data=[]



for i, (x,y) in enumerate(zip(pcs[0,:],pcs[1,:])):

    graph=go.Scatter(x=[0,x],y=[0,y],text=X.columns[i],

                     mode='lines+markers+text',textposition='top left',textfont=dict(family='sans serif',size=15))

    data.append(graph)



layout = go.Layout(title="ACP - Fifa Skills",titlefont=dict(size=40),

            xaxis=dict(title='F1'),

            yaxis=dict(title='F2'),

            autosize=False, width=1000,height=1000,

            showlegend=False)



fig = go.Figure(data=data, layout=layout)



iplot(fig)



#Graph 2



#Choose your player

recherche_joueur=df_acp["Name"]=='M. Sakho'



bool_crack=df_acp["Overall"] > 85

bool_no_crack=df_acp["Overall"]<86

bool_machinegun=df_acp["Name"]=='MachineGun'



palette=['navy','red','#A2D5F2','orange','green','pink']  

data=[]



acp_crack =go.Scatter(x=X_projected[bool_crack,0], y=X_projected[bool_crack,1],name='Crack',

                      text=df_acp.loc[bool_crack,'Name'],

                      textfont=dict(family='sans serif',size=15,color='black'),

                      opacity=0.9,marker=dict(color=palette[2],size=7),mode='markers+text')



acp_no_crack =go.Scatter(x=X_projected[bool_no_crack,0], y=X_projected[bool_no_crack,1],name='Average player',

                         text=df_acp.loc[bool_no_crack,'Name'],

                         opacity=0.6,marker=dict(color=palette[1],size=3),mode='markers')



acp_machinegun =go.Scatter(x=X_projected[bool_machinegun,0], y=X_projected[bool_machinegun,1],name='Perfect player',

                           textfont=dict(family='sans serif',size=20,color='black'),

                           opacity=0.6,marker=dict(color=palette[3],size=30),mode='markers+text')





joueur_recherche =go.Scatter(x=X_projected[recherche_joueur,0], y=X_projected[recherche_joueur,1],name='Searched player',

                           text=df_acp.loc[recherche_joueur,'Name'],

                            textfont=dict(family='sans serif',size=20,color='black'),

                           opacity=1,marker=dict(color=palette[4],size=40),mode='markers+text')



data=[acp_no_crack,acp_crack,acp_machinegun,joueur_recherche]



layout = go.Layout(title="ACP - Fifa Players",titlefont=dict(size=40),

                xaxis=dict(title='F1'),

                yaxis=dict(title='F2'),

                autosize=False, width=1000,height=1000)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
club_avg_overall=df.groupby("Club")["Overall"].mean().reset_index().sort_values("Overall",ascending=False)

club_avg_overall.head()
teamA='Paris Saint-Germain'

teamB='Olympique de Marseille'



bool_teamA=df_acp["Club"]==teamA

bool_teamB=df_acp["Club"]==teamB



palette=['navy','red','#A2D5F2','orange','green','pink','grey']  

data=[]



acp_teamA =go.Scatter(x=X_projected[bool_teamA,0], y=X_projected[bool_teamA,1],name=teamA,

                      text=df_acp.loc[bool_teamA,'Name'],

                      textfont=dict(family='sans serif',size=20,color='black'),

                      opacity=0.7,marker=dict(color=palette[0],size=10),mode='markers+text')



acp_teamB =go.Scatter(x=X_projected[bool_teamB,0], y=X_projected[bool_teamB,1],name=teamB,

                      text=df_acp.loc[bool_teamB,'Name'],

                      textfont=dict(family='sans serif',size=20,color='black'),

                      opacity=0.7,marker=dict(color=palette[2],size=10),mode='markers+text')



acp_all =go.Scatter(x=X_projected[:,0], y=X_projected[:,1],name='All',

                         text=df_acp.loc[:,'Name'],

                         opacity=0.3,marker=dict(color=palette[6],size=3),mode='markers')





data=[acp_teamA,acp_teamB,acp_all]



layout = go.Layout(title="ACP - {} vs {}".format(teamA,teamB),titlefont=dict(size=40),

                xaxis=dict(title='F1'),

                yaxis=dict(title='F2'),

                autosize=False, width=1000,height=1000)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
n_kmean=18000



# On exclue les gardiens de but



df_kmean=df.loc[:n_kmean]

df_kmean=df_kmean[(df_kmean["Position"]!='GK')&(df_kmean['Overall']>69)]



skills_ratings = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']



# On crée le joueur parfait pour savoir où il se situe



MachineGunDict={'Name':'MachineGun','Overall':99}

for skills in skills_ratings:

    MachineGunDict[skills]=99

    

df_kmean=df_kmean.append(MachineGunDict,ignore_index=True)



df_skills=df_kmean[skills_ratings]

X=df_skills
from sklearn.cluster import KMeans

from sklearn import decomposition



# Nombre de clusters souhaités

n_clust = 5



km = KMeans(n_clusters=n_clust)

km.fit(X)



# Récupération des clusters attribués à chaque individu

clusters = km.labels_



# Affichage du clustering par projection des individus sur le premier plan factoriel

pca = decomposition.PCA(n_components=2).fit(X)

X_projected = pca.transform(X)
data=[]

bool_crack=df_kmean["Overall"] > 85

bool_no_crack=df_kmean["Overall"]<86



kmean_clusters = go.Scatter(x=X_projected[:,0], y=X_projected[:,1],

                           mode='markers',

                           marker=dict(

                                size=5,

                                color = clusters.astype(np.float), #set color equal to a variable

                                colorscale='Portland',

                                showscale=False)

                           )



acp_crack =go.Scatter(x=X_projected[bool_crack,0], y=X_projected[bool_crack,1],name='Top players',

                      text=df_kmean.loc[bool_crack,'Name'],

                      textfont=dict(family='sans serif',size=10,color='black'),

                      opacity=0.9,mode='text')



data=[kmean_clusters,acp_crack]



layout = go.Layout(title="ACP + Clustering ",titlefont=dict(size=40),

                xaxis=dict(title='F1'),

                yaxis=dict(title='F2'),

                autosize=False, width=1000,height=1000)



fig = go.Figure(data=data, layout=layout)



iplot(fig)