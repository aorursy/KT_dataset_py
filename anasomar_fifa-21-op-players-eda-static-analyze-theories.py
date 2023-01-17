#Importing the Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Turning off the warnings
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
#Imprting the Date
fifa = pd.read_csv('../input/fifa-21-complete-player-dataset/fifa21_male2.csv')
fifa.shape
#to display all the columns 
pd.set_option('display.max_columns',170)
fifa.head()
fifa.columns
fifa.drop(['ID','Position','Club Logo','Player Photo','Flag Photo','Team & Contract','Team & Contract'],axis=1,inplace=True)
fifa.drop(['Joined','Loan Date End','Value','Wage','Release Clause','Contract'],axis=1,inplace=True)
fifa.drop(['IR','LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LAM'],axis=1,inplace=True)
fifa.drop(['LM','LCM','CM','RCM','RM','LWB','LDM','CDM','RDM','RWB'],axis=1,inplace=True)
fifa.drop(['LB','LCB','CB','RCB','RB','GK','Gender','Hits'],axis=1,inplace=True)
fifa=fifa[(fifa['OVA']>75)].reset_index(drop=True)
fifa.shape
fifa.info()
fifa.head(2)
def cleaning(x):
    try:
        #Height
        x = x.replace("'",".")
        x = x.replace('"',"")
        #Weight
        x=x.replace("lbs",'')
        # W/F and SM
        x=x.replace("★",'')
        if ((".") in x):
            x = x.split(".")
            x = round((int(x[0])*12+int(x[1])) * 2.54, 1)
        return int(x)
    
    except:
        return int(x)
fifa['Height']= fifa['Height'].apply(cleaning)
#clean the weight and conver it to KG as integers
fifa['Weight'] = ((fifa['Weight'].apply(cleaning))*0.453592).astype(int)
fifa['W/F']= fifa['W/F'].apply(cleaning)
fifa['SM'] = fifa['SM'].apply(cleaning)
fifa.info()
fifa[fifa['BP']=='GK'].head()
def missing_value (name_of_the_column):
    
    fifa[name_of_the_column] = fifa.groupby(['OVA','BP'])[name_of_the_column].apply(lambda x: x.fillna(x.mean()))
    fifa[name_of_the_column] = fifa[name_of_the_column].fillna(int(fifa[name_of_the_column].mean()))

for i in ['Balance','Jumping','Volleys','Curve','Agility','Interceptions','Positioning','Vision','Composure','Sliding Tackle']:
    missing_value(i)


def missing_object (name_of_the_column):
    
    fifa[name_of_the_column] = fifa.groupby(['OVA','BP'])[name_of_the_column].apply(lambda x: x.fillna(x.value_counts().idxmax()))
    fifa[name_of_the_column] = fifa[name_of_the_column].fillna(fifa[name_of_the_column].value_counts().idxmax())
for j in ['A/W','D/W']:
    missing_object(j)

fifa.isnull().sum().sum()
positions=[]
for i in fifa.BP.unique():
    positions.append(i)
    vars()[i]=fifa[fifa['BP']==i].reset_index(drop=True)


print(positions)
LWB.head(2)
attacking=pd.concat([ST, CF], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)
defending=pd.concat([RWB,RB,CB,LB,LWB], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)
midfield=pd.concat([CM,CAM,CDM], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)

rightfield=pd.concat([RM,RW], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)
leftfield=pd.concat([LM,LW], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)

sides=pd.concat([RM,RW,LM,LW], axis = 0).sort_values(by=['OVA'], ascending=False).reset_index(drop=True)
sides['BP'].value_counts() 
sides.head(2)
fifa.select_dtypes([np.number]).hist(figsize=(24,24))
plt.show()
def sampler(population, n=30, k=500):
    sample_means = []
    for i in range(k):
        sample = np.random.choice(population, size=n, replace=True)
        sample_means.append(np.mean(sample))
    return sample_means


m,n=0 ,0
fig, ax = plt.subplots(4,2,figsize=(10,20))
for i in ['Height','Acceleration','Reactions','Jumping','Stamina','Aggression','Interceptions']:
    sns.distplot(sampler(fifa[i]), bins=100, kde=True,ax=ax[n][m]).set_title(i)
    m+=1
    if m==2:
        n+=1
        m=0

def OP(dataframe, column,top=0.05):
    P=stats.norm.ppf(1-top)
    z=stats.zscore(dataframe[column])
    a,b,c,d,e,f=[],[],[],[],[],[]
    for x,i in  enumerate(z):
#this functi
        if z[x]>P:
            a.append(dataframe['Name'][x])
            b.append(dataframe['BP'][x])
            c.append(z[x])
            d.append(dataframe['Club'][x])
            e.append(dataframe['OVA'][x])
            f.append(dataframe['Nationality'][x])
            
    outstanding=pd.DataFrame()
    outstanding['name'] =a
    outstanding['club']=d
    outstanding['nationality']=f
    outstanding['position']=b
    outstanding['rating']=e
    outstanding['z value']=c
    outstanding=outstanding.sort_values(by=['z value'], ascending=False).reset_index(drop=True)
    
    del outstanding['z value']
    return outstanding
OP(midfield,'Total Stats')
# the .corr() will return a corrletion matrix between all fields. I am intrested in rating effect so I will loc its row

corr = midfield.corr().loc[:,['OVA']]

fig, ax = plt.subplots(figsize=(5,15))

ax = sns.heatmap(corr.sort_values(by=['OVA'],ascending=False), ax=ax,annot=True,cbar=True,cmap="Greens")
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
attacking.head(1)
A_stat=['Finishing','Heading Accuracy','Short Passing','Dribbling','Acceleration','Sprint Speed','Agility','Reactions','Balance',
      'Shot Power','Stamina','Strength','Long Shots','Positioning','Vision','Composure','W/F','SM']
A_panelty=[0.9,0.3,0.5,0.7,1.8,4,2,0.9,0.8,
         1.8,0.75,0.6,0.5,3,0.7,0.75,20*2,20*3]

S_stat=A_stat
S_stat.append('Crossing')
S_stat.append('Curve')
S_panelty=[0.6,0.2,3,3,2,2,2,2,2,
          1,3,2,1,2,3,2,20*2,20*3,2,1.5]

attacking['Important STAT']=0

for m in range(len(A_panelty)):
    attacking['Important STAT']+=A_panelty[m]*attacking[A_stat[m]]

sides['Important STAT']=0
for n in range(len(S_panelty)):
    sides['Important STAT']+=S_panelty[n]*sides[S_stat[n]]
OP(attacking,'Important STAT',0.1)
OP(sides,'Important STAT')
defending.head(1)
D_stat=['Height','Acceleration','Sprint Speed','Reactions','Jumping','Stamina','Strength','Aggression','Interceptions',
       'Composure','Marking','Standing Tackle','Sliding Tackle']
D_panelty=[1.5,4,4,1,1,2,1,1.5,2,
          1,3,4,1]

defending['Important STAT']=0

for j in range(len(D_panelty)):
    defending['Important STAT']+=D_panelty[j]*defending[D_stat[j]]
   
OP(defending,'Important STAT')
defender=defending.copy()
del defender['Important STAT']

corr = defender.corr().loc[:,['Height','Weight']]

fig, ax = plt.subplots(figsize=(5,15))

ax = sns.heatmap(corr.sort_values(by=['Height'],ascending=False), ax=ax,annot=True,cbar=True,cmap="YlGnBu")
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
Bundesliga=["1. FC Köln", "1. FSV Mainz 05", "DSC Arminia Bielefeld", "	Borussia Dortmund", "FC Augsburg", "FC Bayern München", "FC Schalke 04",
    "Eintracht Frankfurt", "Hertha BSC", "Bayer 04 Leverkusen", "Borussia Mönchengladbach", "RB Leipzig","SC Freiburg",
    "TSG 1899 Hoffenheim","1. FC Union Berlin", "VfB Stuttgart", "VfL Wolfsburg", "SV Werder Bremen"]


OP(defending[defending['Club'].isin(Bundesliga)].reset_index(drop=True),'Important STAT',0.1)
OP(fifa,'Total Stats',0.1)
GK.describe()
X=GK.copy()

del X["Name"]
del X['Club']
del X['Nationality']
del X['D/W']
del X['A/W']
del X['foot']
del X['BP']

gk=X.copy()

X = pd.get_dummies(X, drop_first=True)
from sklearn.preprocessing import StandardScaler

SS = StandardScaler()
X = pd.DataFrame(SS.fit_transform(X),columns=X.columns)
# now the similarity of the data will be investigated by applying supervised clustring first and then unsupervised
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=4).fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
GK["Kmeans label"]=labels

gk["Kmeans label"]=labels

GK[GK['Club']=="FC Bayern München"]
KM=GK[['Name','OVA','Nationality','Club']][GK['Kmeans label']==1].reset_index(drop=True)
KM.head(15)
from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=5, min_samples=2).fit(X)
clustering.labels_
GK['DSBCAN label']=clustering.labels_
gk['DSBCAN label']=clustering.labels_
GK['DSBCAN label'].value_counts()
GK[['Name','OVA','Nationality','Club','DSBCAN label']][GK['DSBCAN label']>=0].reset_index(drop=True)
kmeans=KMeans(n_clusters=20).fit(X)
labels = kmeans.predict(X)
centroids = kmeans.cluster_centers_
GK["Kmeans label"]=labels

gk["Kmeans label"]=labels

GK[['Name',"Kmeans label"]][GK['Club']=="FC Bayern München"]
GK[['Name','OVA','Nationality','Club']][GK["Kmeans label"]==8].reset_index(drop=True)
del GK['DSBCAN label']
GK["Kmeans label"]=GK["Kmeans label"].astype(str)
GK_Kmeans=pd.concat([GK,pd.get_dummies(GK["Kmeans label"])], axis = 1)
GK_Kmeans.head(2)
corr = GK_Kmeans.corr().loc[['8'],:].T

fig, ax = plt.subplots(figsize=(7,17))

ax = sns.heatmap(corr.sort_values(by=['8'],ascending=False), ax=ax,annot=True,cbar=True,cmap="seismic")
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
goalkeeper = GK[['GK Diving','GK Handling','GK Kicking','GK Positioning','GK Reflexes','Height']]
corr=goalkeeper.corr().loc[:,['Height']].T
del corr['Height']
fig, ax = plt.subplots(figsize=(10,2))

ax = sns.heatmap(corr.sort_values(by=['Height'],axis=1,ascending=False), ax=ax,annot=True,cbar=True,cmap="rocket_r")
ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
Benford=pd.DataFrame()
cols=['fifa','midfield','sides','attacking','defending']
for column in cols:
    Benford[column]=vars()[column].sum(axis =1, skipna = True).astype(int).astype(str).str[0]


Benford.head()
m,n=0 ,0
fig, ax = plt.subplots(2,3,figsize=(10,10))

for column in cols:
    Benford[column].value_counts().sort_index().plot(kind='bar',figsize=(12,6),title=column,ax=ax[n][m])
    m+=1
    if m==3:
        n+=1
        m=0