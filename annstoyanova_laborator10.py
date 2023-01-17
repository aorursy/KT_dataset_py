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
import matplotlib as mpl

import matplotlib.pyplot as plt
df = pd.read_csv('../input/drinks.csv', sep=',')

df
print(df.shape)

print(df.shape[0],df.shape[1])
a=df[df['country']=='Ukraine'].iloc[0,0:2]

a

#suma=df['beer_servings'].sum()

#print(suma)

v=df[['country','wine_servings']]

v

sortwine=v.sort_values(by='wine_servings',ascending=False)

sortwine

print(sortwine.head(5))
sred1=df['beer_servings'].mean()

sred1
sred2=df['spirit_servings'].mean()

sred2
sred3=df['wine_servings'].mean()

sred3
sred4=df['total_litres_of_pure_alcohol'].mean()

sred4
svod=df.pivot_table(values=['wine_servings','spirit_servings','beer_servings','total_litres_of_pure_alcohol'], index=['continent'], aggfunc='mean')

svod
svod1=svod.sort_values(by='beer_servings',ascending=False)

svod1
df1=df.sort_values(by='total_litres_of_pure_alcohol',ascending=False)

df1


info=svod1.iloc[:,0:2].plot.bar(figsize=(20,7))

info;
corr=df['wine_servings'].corr(df['beer_servings'])

corr#коэффициент корреляции
#def delete_n(s):

    #return s[:-1] if s.endswith('\n') else s

def chomp(s):

    if s.endwith('\n'):

        return float(s[1:])

    else:

        return s

    
#soot=(df['wine_servings']/df['beer_servings'])*100

#soot

df['wine_servings']=df['wine_servings'].map(chomp)

df['beer_servings']=df['beer_servings'].map(chomp)



#fig, ax = plt.subplots(figsize=(20, 7))

#plt.boxplot(df['wine_servings'], df['beer_servings'], vert=False, labels=['wine', 'beer']);



#fig, ax = plt.subplots(figsize=(10, 20))

#plt.violinplot([wine, beer]);
