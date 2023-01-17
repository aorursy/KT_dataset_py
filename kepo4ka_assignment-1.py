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
data_fl = pd.read_csv('../input/athlete_events.csv')
data_fl.head()

data_fl[(data_fl['Sex']=='M') & (data_fl['Year']==1996)]['Age'].min()

data_fl[(data_fl['Sex']=='F') & (data_fl['Year']==1996)]['Age'].min()
data_fl[(data_fl['Sex']=='M') & (data_fl['Year']==1996)].groupby(['Age']).min()
data_fl[(data_fl['Sex']=='F') & (data_fl['Year']==1996)].groupby(['Age']).min()
# //////////////////////////
# Question 2
gCount = data_fl[(data_fl['Sport']=='Gymnastics') & (data_fl['Year']==2000) & (data_fl['Sex']=='M')].drop_duplicates(['ID']).count()
AllCount = data_fl[(data_fl['Year']==2000) & (data_fl['Sex']=='M')].drop_duplicates('ID').count()['ID']
print(gCount/AllCount * 100)
# //////////////////////////
# Question 3
Height = data_fl[(data_fl['Sex']=='F') & (data_fl['Year']==2000) & (data_fl['Sport']=='Basketball')].drop_duplicates(['ID'])['Height']
Height.fillna(Height.mean()).std()
# //////////////////////////
# Question 4
data_fl[(data_fl['Year']==2002)].sort_values('Weight', ascending=False)[['Weight', 'Sport']].head(1)['Weight']

data_fl[(data_fl['Year']==2002)].groupby(['Weight']).max().tail(1)['Sport']
# //////////////////////////
# Question 5
data_fl[(data_fl['Name']=='Pawe Abratkiewicz')].drop_duplicates(['Year'])['ID'].count()
# //////////////////////////
# Question 6
data_fl[(data_fl['Year']==2000) & (data_fl['Team']=='Australia') & (data_fl['Sport']=='Tennis') & (data_fl['Medal'].notnull())].drop_duplicates(['ID'])['ID'].count()
# //////////////////////////
# Question 7
Swet = data_fl[(data_fl['Year']==2016) & (data_fl['Team']=='Switzerland') & (data_fl['Medal'].notnull())].drop_duplicates(['ID'])['ID'].count()
Swet
Ser = data_fl[(data_fl['Year']==2016) & (data_fl['Team']=='Serbia') & (data_fl['Medal'].notnull())].drop_duplicates(['ID'])['ID'].count()
Ser
if Swet<Ser:
 print('Yes')
else:
    print('No')
 
# //////////////////////////
# Question 9
if data_fl[(data_fl['Season']=='Summer') & (data_fl['City']=='Lake Placid')]['ID'].count() ==0:
 print('No')
else:
    print('Yes')
if data_fl[(data_fl['Season']=='Winter') & (data_fl['City']=='Sankt Moritz')]['ID'].count() ==0:
 print('No')
else:
    print('Yes')
    

# //////////////////////////
# Question 10
y1996 = data_fl[(data_fl['Year']==1996)].drop_duplicates(['Sport'])['Sport'].count()
y1996
y2016 = data_fl[(data_fl['Year']==2016)].drop_duplicates(['Sport'])['Sport'].count()
y2016

y2016 - y1996






















































































































































































