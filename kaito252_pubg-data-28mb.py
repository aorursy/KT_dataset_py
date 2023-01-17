# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')
print(data)
solo_WinRatio = data['solo_WinRatio']
solo_KillDeathRatio = data['solo_KillDeathRatio']
duo_KillDeathRatio = data['duo_KillDeathRatio']
duo_WinRatio = data['duo_WinRatio']
squad_KillDeathRatio = data['squad_KillDeathRatio']
squad_WinRatio = data['squad_WinRatio']
plt.scatter(solo_WinRatio,solo_KillDeathRatio,label='solo')
plt.scatter(duo_WinRatio,duo_KillDeathRatio,label='duo')
plt.scatter(squad_WinRatio,squad_KillDeathRatio,label='squad')
plt.xlabel('WinRatio')
plt.ylabel('KillDeathRatio')
plt.legend()
solo_TimeSurvived = data['solo_TimeSurvived']
plt.scatter(solo_WinRatio,squad_KillDeathRatio)
plt.xlabel('solo_WinRatio')
plt.ylabel('solo_TimeSurvived')

solo_HeadshotKillsPg = data['solo_HeadshotKillsPg']
solo_KillsPg = data['solo_KillsPg']
plt.xlabel('solo_HeadshotKillsPg')
plt.ylabel('solo_KillsPg')
plt.scatter(solo_HeadshotKillsPg,solo_KillsPg)
solo_HeadshotKillsPg = data['solo_HeadshotKillsPg']
solo_KillsPg = data['solo_KillsPg']
duo_HeadshotKillsPg = data['duo_HeadshotKillsPg']
duo_KillsPg = data['duo_KillsPg']
squad_HeadshotKillsPg = data['squad_HeadshotKillsPg']
squad_KillsPg = data['squad_KillsPg']
plt.xlabel('HeadshotKillsPg')
plt.ylabel('KillsPg')
plt.scatter(solo_HeadshotKillsPg,solo_KillsPg,label='solo')
plt.scatter(duo_HeadshotKillsPg,duo_KillsPg,label='duo')
plt.scatter(squad_HeadshotKillsPg,squad_KillsPg,label='squad')
plt.legend()
from mpl_toolkits.mplot3d import Axes3D
solo_Rating = data['solo_Rating']
X = solo_KillDeathRatio
Y = solo_Rating
Z = solo_WinRatio
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("solo_Rating")
ax.set_ylabel("solo_WinRatio")
ax.set_zlabel("solo_KillDeathRatio")
ax.scatter3D(Y, Z, X)
plt.show()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D

data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

solo_WinRatio = data['solo_WinRatio']
duo_WinRatio = data['duo_WinRatio']
squad_WinRatio = data['squad_WinRatio']

solo_KillDeathRatio = data['solo_KillDeathRatio']
duo_KillDeathRatio = data['duo_KillDeathRatio']
squad_KillDeathRatio = data['squad_KillDeathRatio']

solo_Rating = data['solo_Rating']
duo_Rating = data['duo_Rating']
squad_Rating = data['squad_Rating']

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Rating")
ax.set_ylabel("WinRatio")
ax.set_zlabel("KillDeathRatio")

solo = [solo_KillDeathRatio,solo_Rating,solo_WinRatio,'solo']
duo = [duo_KillDeathRatio,duo_Rating,duo_WinRatio,'duo']
squad = [squad_KillDeathRatio,squad_Rating,squad_WinRatio,'squad']
dt = [solo,duo,squad]
for i in dt:
    ax.scatter3D(i[1],i[2],i[0],label=i[3])
plt.legend()
plt.show()