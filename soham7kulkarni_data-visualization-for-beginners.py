# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/fifa19/data.csv')
data
data.columns
new_data = data[['Name','Age','Nationality','Overall','Potential','Club','Value','Preferred Foot','Weak Foot','Skill Moves','Position','BallControl']]
df = new_data.set_index('Club')

df
df1 = df.loc[['FC Barcelona','Juventus','Paris Saint-Germain',

              'Manchester United','Manchester City','Chelsea',

              'Atlético Madrid','FC Bayern München','Real Madrid',

             'Lazio','Inter','Liverpool','Tottenham Hotspur','Milan']]

df1
df2 = df1.groupby(by = 'Club').mean()

df2
df2['Overall'] = df2['Overall'].astype(int)

df2['Potential'] = df2['Potential'].astype(int)

df2['Age'] = df2['Age'].astype(int)

df2['Skill Moves'] = df2['Skill Moves'].astype(int)

df2['BallControl'] = df2['BallControl'].astype(int)

df2
fig,ax = plt.subplots(figsize = (10,7))

chart = sns.barplot(x = df2.index , y = df2['Age'])

chart.set_xticklabels(chart.get_xticklabels() , rotation = 45)

plt.title('Average age of squad' , pad = 20)

plt.xlabel('Club' , labelpad = 20)

plt.ylabel('Age' , labelpad = 20)
fig,ax = plt.subplots(figsize = (10,7))

chart = sns.barplot(x = df2.index , y = df2['Age'])

chart.set_xticklabels(chart.get_xticklabels() , rotation = 45)

plt.title('Average age of squad' , pad = 20)

plt.xlabel('Club' , labelpad = 20)

plt.ylabel('Age' , labelpad = 20)

for p in chart.patches:

        chart.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')

plt.show()
fig,ax = plt.subplots()

fig1 = ax.bar(df2.index , df2['Potential'])

fig2 = ax.bar(df2.index , df2['Overall'])

plt.gcf()

plt.xticks(rotation = 90)
x = np.arange(len(df2.index))

fig,ax = plt.subplots(figsize = (16,8))

bar_width = 0.3



b1 = ax.bar(x , df2['Overall'] , width = bar_width)

b2 = ax.bar(x + bar_width , df2['Potential'] , width = bar_width)

ax.set_xticks(x + bar_width/2)

ax.set_xticklabels(df2.index)

plt.xticks(rotation = 45)

ax.set_title('Ratings comparison' , pad = 20)

plt.legend(['Overall' , 'Potential'])

plt.xlabel('Club')

plt.ylabel('Ratings' , labelpad = 20)

plt.ylim(0,130)

def autolabel(figure , xpos = 'center'):

    """

    Attach a text label above each bar in *fig*, displaying its height.



    *xpos* indicates which side to place the text w.r.t. the center of

    the bar. It can be one of the following {'center', 'right', 'left'}.

    """

    xpos = xpos.lower()

    ha = {'center' : 'center' , 'left' : 'left' , 'right' : 'right'}

    offset = {'center' : 0.5 , 'left' : 0.43 , 'right' : 0.57}

    

    for fig in figure :

        height = fig.get_height()

        ax.text(fig.get_x() + fig.get_width()*offset[xpos] , 1.01*height,

               '{}'.format(height) , ha = ha[xpos] , va = 'bottom')



autolabel(b1)

autolabel(b2)
x = np.arange(len(df2.index))

fig,ax = plt.subplots(figsize = (16,8))

width = 0.4



b1 = ax.bar(x , df2['Skill Moves'] , width = width)

b2 = ax.bar(x + width, df2['Weak Foot'] , width = width)

ax.set_xticks(x+width/2)

ax.set_xticklabels(df2.index)

plt.xticks(rotation = 45)

plt.legend(['Skill Moves' , 'Weak Foot'])

plt.title('Skills and Weak Foot' , pad = 10)

plt.xlabel('Club' , labelpad = 20)

plt.ylabel('Ratings' , labelpad = 20)