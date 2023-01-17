# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

from sklearn import cluster





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv('../input/Mass Shootings Dataset.csv',encoding = "ISO-8859-1",parse_dates=['Date'])

print(df.shape) 



# Any results you write to the current directory are saved as output.
print(df.duplicated(subset='Date').sum())

print(len(df))

df = df.drop_duplicates(subset='Date')

print(len(df))
df = df.sort_values('Date', ascending=True)

df.plot(x='Date',y='Fatalities',style='o',alpha=0.4,legend=False)

plt.xticks(rotation='horizontal')

plt.xlabel('Date', fontsize=12)

plt.ylabel('Fatalities', fontsize=12)

plt.show()
df['Year'] = df['Date'].dt.year

counts = df['Year'].value_counts(sort=False)

counts.plot(kind='bar')

plt.show()
df['Ratio'] = df['Fatalities']/(df['Fatalities'] + df['Injured'])

df.plot(x='Year',y='Ratio',style='o',alpha=0.4,legend=False)

plt.show()
print(df.isnull().sum())
print(len(df))
df = df.dropna()

print(len(df))
def group_years(year):

    if year in [1966,1967,1968,1969,1970,1971,1972,1973,1974,1975,1976,1977,1978]:

        group = 'A'

    elif year in [1979,1980,1981,1982,1983,1984]:

        group = 'B'

    elif year in [1985,1986,1987,1988,1989]:

        group = 'C'

    elif year in [1990,1991,1992,1993,1994]:

        group = 'D'

    elif year in [1995,1996,1997,1998,1999]:

        group = 'E'

    elif year in [2000,2001,2002,2003,2004]:

        group = 'F'

    elif year in [2005,2006,2007,2008,2009,2010]:

        group = 'G'

    elif year in [2011,2012,2013,2014,2015,2016,2017]:

        group = 'H'

    else:

        group = ''

    return group



df = df[df['S#'].isin([315,291,292])==False] #drop the two Hawaii and one Alaska rows, it keeps the plot more compact



df['Group'] = df['Year'].apply(lambda x: group_years(x))



colors = {'A':'#080707','B':'#282626','C':'#3d3939','D':'#686666','E':'#797777','F':'#a9a9a3','G':'#bebfc1','H':'#d2d2d2'}

fig, ax = plt.subplots()

ax.scatter(df['Longitude'],df['Latitude'], c=df['Group'].apply(lambda x: colors[x]),alpha=0.6)

plt.show()
k=15

f1 = df['Longitude'].values

f2 = df['Latitude'].values



X=np.matrix(list(zip(f1,f2)))

kmeans = cluster.KMeans(n_clusters=k).fit(X)



labels = kmeans.labels_

centroids = kmeans.cluster_centers_



for i in range(k):

    # select only data observations with cluster label == i

    ds = X[np.where(labels==i)]

    # plot the data observations

    plt.plot(ds[:,0],ds[:,1],'o')

    # plot the centroids

    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')

    # make the centroid x's bigger

    plt.setp(lines,ms=15.0)

    plt.setp(lines,mew=2.0)

plt.show()