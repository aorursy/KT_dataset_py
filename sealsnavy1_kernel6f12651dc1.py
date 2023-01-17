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
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')
import pandas as pd 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/PUBG_Player_Statistics.csv')

kd=data['solo_HeadshotKills']
win=data['solo_WinRatio']
kd1=data['duo_HeadshotKills']
win1=data['duo_WinRatio']
kd2=data['squad_HeadshotKills']
win2=data['squad_WinRatio']


plt.scatter(win/10,kd/10,color='blue',label='solo')
plt.scatter(win1/10,kd1/10,color='orange',label='duo')
plt.scatter(win2/10,kd2/10,color='green',label='squad')

plt.xlabel('win ratio')
plt.ylabel('Headsshotkills')

plt.title('influence of headshotkills to win ratio')
plt.axis(xmin=0,ymin=0)
plt.legend()
plt.show()