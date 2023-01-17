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
end20=pd.read_csv("/kaggle/input/2020-democratic-primary-endorsements/endorsements-2020.csv")
end20.head()
end20['date']=pd.to_datetime(end20['date'])
end20=end20.iloc[1:]
start=min(end20.date)

end=max(end20.date)
idx=pd.date_range(start,end,freq='D')

L=len(idx)

names=list(end20.endorsee.unique())
#drop nan

names=names[:-1]
def blank():

    outdf=pd.DataFrame(index=idx,data={names[i]:L*[0] for i in range(len(names))})

    return outdf

def endts(df):

    outdf=blank()

    #remove NaT

    df=df.dropna(subset=['date'])

    for i in range(len(df)):

        date=df.iloc[i].date

        ende=df.iloc[i].endorsee

        pnts=df.iloc[i].points

        outdf.loc[date,ende]+=pnts

    return outdf
allts=endts(end20)
allts.head()
def rollavgs(df,ende,pds):

    return sum([df[ende].rolling(window=i).mean() for i in pds])/len(pds)
import matplotlib.pyplot as plt

plt.close()

for name in names:

    rollavgs(allts,name,[7]).plot()

plt.show()
nh=end20.loc[end20['state']=='NH',:]
nhts=endts(nh)
plt.close()

for name in names:

    rollavgs(nhts,name,[7]).plot()

plt.legend(loc='upper left')

plt.show()
biden=end20.loc[end20['endorsee']=='Joe Biden',:]
biden.groupby('endorser party').count()
def byparty(cand):

    df=end20.loc[end20['endorsee']==cand,:]

    return df.groupby('endorser party').count()
ia=end20.loc[end20['state']=='IA',:]

iats=endts(ia)
plt.close()

for name in ['Joe Biden','Bernie Sanders','Amy Klobuchar','Elizabeth Warren','Pete Buttigieg','Michael Bloomberg']:

    rollavgs(iats,name,[7]).plot()

plt.legend(loc='upper left')

plt.show()
plt.close()

for name in ['Joe Biden','Bernie Sanders','Amy Klobuchar','Elizabeth Warren','Pete Buttigieg','Michael Bloomberg']:

    rollavgs(nhts,name,[7]).plot()

plt.legend(loc='upper left')

plt.show()
ia.groupby('endorsee').sum()['points']
ia.groupby('endorsee').count()['endorser']
ia.groupby('endorsee').mean()['points']
nh.groupby('endorsee').mean()['points']
sc=end20.loc[end20['state']=='SC',:]
sc.groupby('endorsee').mean()['points']
end20.dropna(subset=['date']).groupby('state').count()['points'].sort_values(ascending=False)
tx=end20.loc[end20['state']=='TX',:]

tx.groupby('endorsee').mean()['points']
texavgend=tx.groupby('endorsee').mean()['points']
texavgend.index