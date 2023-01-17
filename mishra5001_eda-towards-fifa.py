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
df = pd.read_csv("../input/FIFA19 - Ultimate Team players.csv")
df.describe()
df.isnull().sum()
#checking Null values
df.describe()
df.columns
df.shape
df.corr()
import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# Let Us Explore Nationality at First Instance
import matplotlib.pyplot as plt
p = sns.countplot(x="nationality" , data=df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 60)
#Looking at Pace
p = sns.countplot(x="pace" , data=df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 60)
p = sns.countplot(x="drib_ball_control" , data=df , palette = "muted")
_ = plt.setp(p.get_xticklabels(),rotation = 60)
p = sns.countplot(x="dribbling" , data=df , palette = "bright")
_ = plt.setp(p.get_xticklabels(),rotation = 60)
#Seeing the variation of Overall Rating with age 
p = sns.countplot(x='overall' , data = df , hue = 'age' , palette = 'bright')
_ = plt.setp(p.get_xticklabels(), rotation = 90)
df['cam'].value_counts().plot(kind = 'bar')
df = df.sample(frac = 0.1 , random_state=1)
print(df.shape)
p = sns.countplot(x="dribbling" , data=df , palette = "bright")
_ = plt.setp(p.get_xticklabels(),rotation = 60)
df['cam'].value_counts().plot(kind = 'bar')
#Defining a list of positions present in our data
position = ['cam', 'cb', 'cdm', 'cf', 'cm',
       'lb', 'lf', 'lm', 'lw', 'lwb', 'rb', 'rf',
       'rm', 'rw', 'rwb']
#Searching for Top 10 CAM players by position
inputs_good = 0
while inputs_good==0:
    user_input = 'cam' 
    input_list = user_input.split(',')

    search = []
    for i in input_list:
        search.append(i.strip().lower())
    inputs_good = all(elem in position for elem in search)
    if inputs_good:
        print('User wants to search for Top 10: ', ", ".join(search))
    else:
        print('Invalid position. Please re-enter the position (e.g. RAM, CF, CDM)')
#Searching for players by Positions.
for i in search:
    print('\n\n','Top 10', i, 'in FIFA 19', '\n')
    print(df.sort_values(i, ascending=False).head(10)[['player_id', 'nationality', 'club', 'overall']])
#printing all the top 10 players by position for reference
for i in position:
    print('\n\n','Top 10', i, 'in FIFA 19', '\n')
    print(df.sort_values(i, ascending=False).head(10).reset_index()[['player_id', 'nationality', 'club', 'overall']])
pd.crosstab(df.position,df.age).plot(kind='bar');
df.pivot_table(index='age' , columns = 'position',values='overall',aggfunc='mean')
df.groupby(['overall','position']).age.mean()