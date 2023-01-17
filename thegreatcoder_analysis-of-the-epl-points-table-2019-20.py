# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/english-premier-league-points-table-20192020/Premier League Points Table 2019-20.csv")
df.head()
%matplotlib inline

plt.rcParams['figure.figsize']=30,15
df.sample()
df['Position Change']=df['Position']-df['Previous position']
df.head()
type(df['GF'])
sdf = df[['Club','Played','Won','Drawn','Lost']]
import matplotlib.pyplot as plt

import seaborn as sns
sdf.plot('Club')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Matches Stats",fontsize = 35)

plt.title("Matches Stats per Team",fontsize = 50)

plt.show()
sdf[:10].plot('Club')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Matches Stats",fontsize = 35)

plt.title("Matches Stats per Team",fontsize = 50)

plt.show()
sdf[:10].plot('Club',kind = 'bar')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Matches Stats",fontsize = 35)

plt.title("Matches Stats per Team",fontsize = 50)

plt.show()
df.plot(x = 'Club' , y = 'GF' , kind = 'bar')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Goals Conceded",fontsize = 35)

plt.title("Goals Conceded per Team",fontsize = 50)

plt.show()
df.plot(x = 'Club' , y = 'GA' , kind = 'bar')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Goals Scored",fontsize = 35)

plt.title("Goals Scored per Team",fontsize = 50)

plt.show()
df.plot(x = 'Club' , y = 'GD' , kind = 'bar')

plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)

plt.xticks(fontsize = 30,rotation = 65)

plt.yticks(fontsize = 30)

plt.xlabel("Club Name",fontsize = 35)

plt.ylabel("Goal Difference",fontsize = 35)

plt.title("Goals Difference per Team",fontsize = 50)

plt.show()