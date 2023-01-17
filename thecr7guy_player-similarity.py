# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial import distance

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")
df1=df.iloc[:,0:20]
df2=df.iloc[:,20:40]
df3=df.iloc[:,40:60]
df4=df.iloc[:,60:80]
df5=df.iloc[:,80:]
df1.info()
df1.describe().transpose()
df1.columns
df1=df1.drop(['sofifa_id', 'player_url', 'long_name', 'age', 'dob',
       'height_cm', 'weight_kg', 'nationality', 'club', 'overall', 'potential',
       'value_eur', 'wage_eur','international_reputation','player_positions','preferred_foot'],axis=1)
df1.head()
df2.head()
df2.columns
df2=df2.drop(['body_type', 'real_face', 'release_clause_eur', 'team_jersey_number', 'loaned_from', 'joined',
       'contract_valid_until', 'nation_jersey_number',"nation_position",'player_tags','team_position','gk_diving', 'gk_handling', 'gk_kicking'],axis=1)
df2.head()
df2.info()
df2=df2.fillna(0)
df2.info()
df3.info()
df3.columns
df3=df3.drop(["player_traits",'gk_reflexes', 'gk_speed', 'gk_positioning'],axis=1)
df3.head()
df3=df3.fillna(0)
df4.info()
df4.head()
df4=df4.drop(["ls","st"],axis=1)
df1.columns
df2.columns
df3.columns
df4.columns
df=pd.concat([df1,df2,df3,df4],axis=1)
df.head()
df["weak_foot"].unique()
df["skill_moves"].unique()
def wr(x):
    x=x.split("/")
    return x[0]
df["offensive_work_rate"]=df["work_rate"].apply(lambda x: wr(x))
def wr(x):
    x=x.split("/")
    return x[1]
df["defensive_work_rate"]=df["work_rate"].apply(lambda x: wr(x))
df=df.drop(["work_rate"],axis=1)
df.head()
df["offensive_work_rate"].unique()
def wr(x):
    if x=="High":
        return 0
    elif x=="Medium":
        return 1
    else:
        return 2
df["offensive_work_rate"]=df["offensive_work_rate"].apply(lambda x: wr(x))
def wr(x):
    if x=="High":
        return 0
    elif x=="Medium":
        return 1
    else:
        return 2
df["defensive_work_rate"]=df["defensive_work_rate"].apply(lambda x: wr(x))
df.head()
df.info()
X=df.drop(["short_name"],axis=1)
X
player_name="L. Messi"
p_ind=df[df["short_name"]==player_name].index[0]
dist=[]
for i in range (0,len(X.index)):
    dist.append(distance.euclidean(X.iloc[p_ind].values,X.iloc[i].values))
len(dist)
pd.Series(dist)
sn=df["short_name"].to_list()
sim={"name":sn,"distance":dist}
sim=pd.DataFrame(sim)
sim.iloc[sim["distance"].sort_values().index].head(10)
cossim=[]
for i in range (0,len(X.index)):
    cossim.append(1 - spatial.distance.cosine(X.iloc[p_ind].values,X.iloc[i].values))
pd.Series(cossim)
sim2={"name":sn,"cossim":cossim}
sim2=pd.DataFrame(sim2)
sim2.iloc[sim2["cossim"].sort_values(ascending=False).index].head(10)
sim.iloc[sim["distance"].sort_values().index].head(10)
X
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
X=std.fit_transform(X)
cossim=[]
for i in range (0,len(X)):
    cossim.append(1 - spatial.distance.cosine(X[p_ind],X[i]))
pd.Series(cossim)
sim2={"name":sn,"cossim":cossim}
sim2=pd.DataFrame(sim2)
sim2.iloc[sim2["cossim"].sort_values(ascending=False).index].head(10)
dist=[]
for i in range (0,len(X)):
    dist.append(distance.euclidean(X[p_ind],X[i]))
pd.Series(dist)
sim={"name":sn,"distance":dist}
sim=pd.DataFrame(sim)
sim.iloc[sim["distance"].sort_values().index].head(10)

