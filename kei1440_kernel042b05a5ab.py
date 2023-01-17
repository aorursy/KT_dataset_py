# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

kd=data['solo_KillDeathRatio']
top10=data['solo_WinTop10Ratio']
kd1=data['duo_KillDeathRatio']
top101=data['duo_WinTop10Ratio']
kd2=data['squad_KillDeathRatio']
top102=data['squad_WinTop10Ratio']
plt.scatter(top10,kd/10,color='blue',alpha=0.4,label='solo')
plt.scatter(top101,kd1/10,color='orange',alpha=0.3,label='duo')
plt.scatter(top102,kd2/10,color='green',alpha=0.2,label='squad')
plt.ylabel('kill/death')
plt.title('influence of kd to top10 ratio')
plt.axis(xmin=0,ymin=0)
plt.legend()
plt.show()

import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

kd=data['solo_KillDeathRatio']
win=data['solo_WinRatio']
kd1=data['duo_KillDeathRatio']
win1=data['duo_WinRatio']
kd2=data['squad_KillDeathRatio']
win2=data['squad_WinRatio']
plt.scatter(win/10,kd/10,color='blue',alpha=0.4,label='solo')
plt.scatter(win1/10,kd1/10,color='orange',alpha=0.3,label='duo')
plt.scatter(win2/10,kd2/10,color='green',alpha=0.2,label='squad')
plt.xlabel('win ratio')
plt.ylabel('kill/death ratio')
plt.title('influence of kd to win ratio')
plt.axis(xmin=0,ymin=0)
plt.legend()
plt.show()
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

kd=data['solo_DamagePg']
win=data['solo_WinRatio']
kd1=data['duo_DamagePg']
win1=data['duo_WinRatio']
kd2=data['squad_DamagePg']
win2=data['squad_WinRatio']
plt.scatter(win/10,kd,color='blue',alpha=0.4,label='solo')
plt.scatter(win1/10,kd1,color='orange',alpha=0.3,label='duo')
plt.scatter(win2/10,kd2,color='green',alpha=0.2,label='squad')
plt.xlabel('win ratio')
plt.ylabel('damage pergame')
plt.title('influence of damagepg to win ratio')
plt.axis(xmin=0,ymin=0)
plt.legend()
plt.show()
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

kd=data['solo_MoveDistance']
win=data['solo_WinRatio']
kd1=data['duo_MoveDistance']
win1=data['duo_WinRatio']
kd2=data['squad_MoveDistance']
win2=data['squad_WinRatio']
plt.scatter(win/10,kd,color='blue',alpha=0.4,label='solo')
plt.scatter(win1/10,kd1,color='orange',alpha=0.3,label='duo')
plt.scatter(win2/10,kd2,color='green',alpha=0.2,label='squad')
plt.xlabel('win ratio')
plt.ylabel('movedistance')
plt.title('influence of movedistance to win ratio')
plt.axis(xmin=0,ymin=0)
plt.legend()
plt.show()
