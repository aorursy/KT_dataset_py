# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df=pd.read_csv('../input/NBA_player_of_the_week.csv')
df.head(2)
# Any results you write to the current directory are saved as output.
df.info()
df['Height'].unique()
df['Date']=pd.to_datetime(df['Date'])
def height_categ(x):
    """
    """
    a=x.split('-')
    if len(a) > 1:
        a=float(a[0])+float(a[1])/12
        if (a<6.0):
            return 'below 6'
        elif (a>=6.0) & (a<=6.5):
            return '6 to 6.5 feet'
        elif (a>6.5) & (a<7.0):
            return '6.5 to 7 feet'
        elif (a>=7.0):
            return 'over 7'
    else:
        a=x
        a=a.replace('cm',' ')
        a=float(a)*0.0328084
        if (a<6.0):
            return 'below 6'
        elif (a>=6.0) & (a<=6.5):
            return '6 to 6.5 feet'
        elif (a>6.5) & (a<7.0):
            return '6.5 to 7 feet'
        elif (a>=7.0):
            return 'over 7'

df['Height_category']=df['Height'].apply(height_categ)
#df['Height_category'].unique()
#df['Position'].unique()
temp=df.groupby('Height_category')['Height_category'].count()
plt.bar(temp.index,temp)
df['Weight'].unique()

def conv_weight(x):
    if 'kg' in x:
        a=x
        a=x.replace('kg','')
        a=float(a)*2.20462
        a=int(a)
        return a 
    else:
        return x
df['Weight']=df['Weight'].apply(conv_weight)
def weight_categ(x):
    if x>100 and x<151:
        return '100 to 150 pounds'
    elif x>150 and x<201:
        return '150 to 200 pounds'
    elif x>200 and x<251:
        return '200 to 250 pounds'
    elif x>250 and x<301:
        return '250 to 300 pounds'
    elif x>300:
        return 'over 300 pounds!!!'
    else:
        return 'None'
df['Weight']=pd.to_numeric(df['Weight']);
df['weight_categ']=df['Weight'].apply(weight_categ);
df['weight_categ'].unique()
df['Position'].unique()
#https://www.reddit.com/r/nba/comments/g6i6k/can_someone_explain_the_various_positions_in/
positions={
'PG' : 'point guard' ,
'SG' : 'shooting guard'  ,
'F' : 'forward' ,
'C' : 'center' ,
'SF' : 'small forward' ,
'PF' : 'power forward' ,
'G' :  'guard' ,
'FC' : 'forward center'  ,
'GF' : 'guard forward' ,
'F-C': 'forward center'  ,
'G-F': 'guard forward' 
}
df['positions_descr']=df['Position'].map(positions)
df[['Position','positions_descr']].head(2)
plt.scatter(df['Height_category'],df['positions_descr'])

#plt.subplot(4,1,1)
temp=df[df['Height_category']=='below 6']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.bar(temp.index,temp)
plt.xticks(rotation=60)
plt.title('below 6 feet player positions' )
plt.show()

#plt.subplot(4,1,2)
temp=df[df['Height_category']=='6 to 6.5 feet']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=60)
plt.bar(temp.index,temp)
plt.title('6 to 6.5 feet player positions')
plt.show()

#plt.subplot(4,1,3)
temp=df[df['Height_category']=='6.5 to 7 feet']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=45)
plt.bar(temp.index,temp)
plt.title('6.5 to 7 feet player positions')
plt.show()

#plt.subplot(4,1,3)
temp=df[df['Height_category']=='over 7']
temp=temp.groupby('positions_descr')['positions_descr'].count()
plt.xticks(rotation=45)
plt.bar(temp.index,temp)
plt.title('over 7 feet player positions')
plt.show()

temp=df
temp['count']=0
temp=temp.groupby(['Height_category','weight_categ'])['count'].count()
temp=temp.reset_index()
temp.sort_values(ascending=False,by='count')
temp['ht_wt_category']=temp['Height_category']+ ' and ' +temp['weight_categ']
temp=temp.sort_values(ascending=False,by='count')
plt.barh(temp['ht_wt_category'].head(5),temp['count'].head(5))
plt.xticks(rotation=45)
temp=df.groupby('positions_descr')['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index,temp)
plt.xticks(rotation=60)
temp=df.groupby(['Age'])['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index,temp)
temp=df.groupby(['Team'])['count'].count()
temp=temp.sort_values(ascending=False)
plt.bar(temp.index[:5],temp[:5])
plt.xticks(rotation=60)