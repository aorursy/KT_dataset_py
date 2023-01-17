# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")
print(data.columns.values)
print(data["shot_made_flag"].isnull().sum())
data.info()
print(max(data["loc_x"]))
print(min(data["loc_x"]))
print(max(data["loc_y"]))
print(min(data["loc_y"]))
plt.figure(figsize=(2*7,7*84/50))

ax1=plt.subplot(121)
hit=data.loc[data.shot_made_flag==1]
plt.scatter(hit.loc_x,hit.loc_y,color='g',alpha=0.05)
plt.title("Shot in")
ax1.set_ylim([-100,800])

ax2=plt.subplot(122)
hit=data.loc[data.shot_made_flag==0]
plt.scatter(hit.loc_x,hit.loc_y,color='r',alpha=0.05)
plt.title("Shot out")
ax2.set_ylim([-100,800])
print(data.combined_shot_type.unique())
from matplotlib.pyplot import cm

fig,ax=plt.subplots(figsize=(2*7,7*84/50))
color=iter(cm.rainbow(np.linspace(0,1,6)))

for shot in data.combined_shot_type.unique():
    c=next(color)
    shot_type=data.loc[data.combined_shot_type==shot]
    ax.scatter(shot_type.loc_x,shot_type.loc_y,color=c,alpha=1,label=shot)
    ax.set_ylim([-100,600])
    ax.legend()
def graphe_acc(df,factor):
    ct=pd.crosstab(df.shot_made_flag,df[factor],normalize=true)
    x,y=ct.columns,ct.values[1,:] # prend la première valeur
    plt.plot(x,y)
    plt.xlabel(factor)
    plt.ylabel("%made")
    
graphe_acc(data,"shot_distance")