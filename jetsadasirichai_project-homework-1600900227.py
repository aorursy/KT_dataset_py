# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr
%matplotlib inline
b=pd.read_csv('../input/top-tracks-of-2017/featuresdf.csv')
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import pearsonr
%matplotlib inline
a=pd.read_csv('../input/top-spotify-tracks-of-2018/top2018.csv')
filename='/kaggle/input/top-spotify-tracks-of-2018/top2018.csv'
a=pd.read_csv(filename,encoding='ISO-8859-1')
a.head()
filename='/kaggle/input/top-spotify-tracks-of-2018/top2018.csv'
spoti=pd.read_csv(filename,encoding='ISO-8859-1')
spoti.head(50)


filename='/kaggle/input/top-tracks-of-2017/featuresdf.csv'
b=pd.read_csv(filename,encoding='ISO-8859-1')
b.head()
filename='/kaggle//input/top-tracks-of-2017/featuresdf.csv'
spoti=pd.read_csv(filename,encoding='ISO-8859-1')
spoti.head(50)
sns.set_style(style='darkgrid')
sns.distplot(a['danceability'],hist=True,kde=True)


sns.set_style(style='darkgrid')
sns.distplot(b['danceability'],hist=True,kde=True)

Vd=a['danceability']>=0.75
Rd=(a['danceability']>=0.5) & (a['danceability']<0.75)
Ld=a['danceability']<0.5
data=[Vd.sum(),Rd.sum(),Ld.sum()]
Dance=pd.DataFrame(data,columns=['percent'],
                   index=['Very','Regular','Low'])
Dance
Vd=b['danceability']>=0.75
Rd=(b['danceability']>=0.5) & (b['danceability']<0.75)
Ld=b['danceability']<0.5
data=[Vd.sum(),Rd.sum(),Ld.sum()]
Dance=pd.DataFrame(data,columns=['percent'],
                   index=['Very','Regular','Low'])
Dance
a['Rhythm']=a['tempo']
a.loc[a['tempo']>168,'Rhythm']='Presto'
a.loc[(a['tempo']>=110) & (a['tempo']<=168),'Rhythm']='Allegro'
a.loc[(a['tempo']>=76) & (a['tempo']<=108),'Rhythm']='Andante'
a.loc[(a['tempo']>=66) & (a['tempo']<=76),'Rhythm']='Adagio'
a.loc[a['tempo']<65,'Rhythm']='Length'
a['Rhythm'].value_counts()


b['Rhythm']=b['tempo']
b.loc[b['tempo']>168,'Rhythm']='Presto'
b.loc[(b['tempo']>=110) & (b['tempo']<=168),'Rhythm']='Allegro'
b.loc[(b['tempo']>=76) & (b['tempo']<=108),'Rhythm']='Andante'
b.loc[(b['tempo']>=66) & (b['tempo']<=76),'Rhythm']='Adagio'
b.loc[b['tempo']<65,'Rhythm']='Length'
b['Rhythm'].value_counts()
sns.set_style(style='darkgrid')
Rhy=a['Rhythm'].value_counts()
Rhy_A=pd.DataFrame(Rhy)
sns.barplot(x=Rhy_A.Rhythm, y=Rhy_A.index, palette="viridis")
plt.title('Popular keys')
sns.set_style(style='darkgrid')
Rhy=b['Rhythm'].value_counts()
Rhy_B=pd.DataFrame(Rhy)
sns.barplot(x=Rhy_B.Rhythm, y=Rhy_B.index, palette="viridis")
plt.title('Popular keys')
a[['name','artists','danceability','valence','tempo','Rhythm']].sort_values(by='danceability',ascending=False).head(10)
b[['name','artists','danceability','valence','tempo','Rhythm']].sort_values(by='danceability',ascending=False).head(10)
a[['name','artists','energy','valence','tempo','Rhythm']].sort_values(by='valence',ascending=False).head(10)

b[['name','artists','energy','valence','tempo','Rhythm']].sort_values(by='valence',ascending=False).head(10)