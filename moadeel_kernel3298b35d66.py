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
df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
df.head()
df.shape
df.isnull().sum()
df = df.dropna()
df.shape
import seaborn as sns

import matplotlib.pyplot as plt
df.columns
sns.catplot(x = 'Genre',y = 'Global_Sales', data = df, kind = 'bar',height=8,aspect=2)
Sports = 0

Platform = 0

Racing = 0

Role_Playing = 0

Puzzle = 0

Misc=0

Shooter = 0

Simulation = 0

Action=0

Fighting=0

Adventure=0

Strategy = 0

for row in df.Genre:

    

    if(row == 'Sports'):

        Sports += 1

    elif(row == 'Platform'):

        Platform += 1

    elif(row == 'Racing'):

        Racing += 1

    elif(row == 'Role-Playing'):

        Role_Playing += 1

    elif(row == 'Puzzle'):

        Puzzle += 1

    elif(row == 'Misc'):

        Misc += 1

    elif(row == 'Shooter'):

        Shooter += 1

    elif(row == 'Simulation'):

        Simulation += 1

    elif(row == 'Action'):

        Action += 1

    elif(row == 'Fighting'):

        Fighting += 1

    elif(row == 'Adventure'):

        Adventure += 1

    elif(row == 'Strategy'):

        Strategy += 1



        

explodeTuple = (0.1, 0.25, 0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)



values = [Sports,Platform,Racing,Role_Playing,Puzzle,Misc,Shooter,Simulation,Action,Fighting,Adventure,Strategy]

percent=[]

for i in values:

    a = i/sum(values)

    b = a * 100

    percent.append(b)

categories = ['Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle', 'Misc',

       'Shooter', 'Simulation', 'Action', 'Fighting', 'Adventure',

       'Strategy']

        

plt.pie(percent, explode=explodeTuple,labels=categories,autopct='%1.2f',wedgeprops   = { 'linewidth' : 1},

        startangle=0)

plt.axis('equal')

plt.tight_layout()

plt.show()
df.head()
sns.catplot(x = 'Genre',y = 'NA_Sales', data = df, kind = 'bar',height=8,aspect=2)
sns.catplot(x = 'Genre',y = 'EU_Sales', data = df, kind = 'bar',height=8,aspect=2)
sns.catplot(x = 'Genre',y = 'JP_Sales', data = df, kind = 'bar',height=8,aspect=2)
sns.catplot(x = 'Genre',y = 'Other_Sales', data = df, kind = 'bar',height=8,aspect=2)
df.Platform.unique()
sns.catplot(x="Platform", y="NA_Sales",

             kind='bar',data=df,height=8,aspect=2)
sns.catplot(x="Platform", y="EU_Sales",

             kind='bar',data=df,height=8,aspect=2)
sns.catplot(x="Platform", y="JP_Sales",

             kind='bar',data=df,height=8,aspect=2)
sns.catplot(x="Platform", y="Other_Sales",

             kind='bar',data=df,height=8,aspect=2)
sns.catplot(x="Platform", y="Global_Sales",

             kind='bar',data=df,height=8,aspect=2)