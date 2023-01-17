# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/AppleStore.csv')
# Any results you write to the current directory are saved as output.

Maxrating = df.groupby(["prime_genre"],as_index=False)['user_rating'].max()
df1 = pd.DataFrame({'labels' : Maxrating.prime_genre, 'values': Maxrating.user_rating})
df1 = df1.set_index('labels')
fig, ax = plt.subplots(figsize=(10, 6))
a = ax.barh(df1.index, df1['values'], 0.8, color = ['#8B7355','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B']) # plot a vals
plt.title('Max rating')            
plt.show()

Minrating = df[df['user_rating'] >0].groupby(["prime_genre"],as_index=False)['user_rating'].min()
df1 = pd.DataFrame({'labels' : Minrating.prime_genre, 'values': Minrating.user_rating})
df1 = df1.set_index('labels')
fig, ax = plt.subplots(figsize=(10, 6))
a = ax.barh(df1.index, df1['values'], 0.8, color = ['#8B7355','#FF6103','#8EE5EE','#458B00','#FFF8DC','#68228B']) # plot a vals
plt.title('Minimum rating')            
plt.show()

plt.subplots(figsize=(10,6))
st = df[((df['user_rating'] >0) & (df['prime_genre'] != 'Games'))].groupby(['prime_genre','user_rating'],as_index=False).size().reset_index(name ='Count').sort_values(by ='Count', ascending = False)
sns.barplot(x='prime_genre',y='Count',data=st,hue='user_rating')
plt.ylabel("Number of user ratings",fontsize=15,color='navy')    
plt.title("Number of user rating per prime genre",fontsize=30,color='navy')
plt.xlabel("prime genre--->",fontsize=15,color='navy')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.legend(loc = 'upper right')
plt.show()



xyz=df[((df['price'] >0) & (df['prime_genre'] != 'Games'))].groupby(['prime_genre','user_rating'])['price'].sum().reset_index()        
#xyz.drop('match_id',axis=1,inplace=True)
xyz=xyz.sort_values(by=['prime_genre','price'],ascending=True)
sns.boxplot(x = 'prime_genre',y = 'price',data= xyz,width =0.5).set_title('Price Range')
plt.xticks(rotation=45,ha='right')
plt.show()


df['AppSizeInKB'] =df['size_bytes']/1024
xyz=df[((df['price'] >0) & (df['prime_genre'] != 'Games'))].groupby(['prime_genre','user_rating'])['AppSizeInKB'].sum().reset_index()        
#xyz.drop('match_id',axis=1,inplace=True)
xyz=xyz.sort_values(by=['prime_genre','AppSizeInKB'],ascending=True)
sns.boxplot(x = 'prime_genre',y = 'AppSizeInKB',data= xyz,width =0.5).set_title('App Size')
plt.xticks(rotation=45,ha='right')
plt.show()