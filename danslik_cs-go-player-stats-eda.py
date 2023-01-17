import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df = pd.read_csv('../input/playerStats.csv')
#Get unique value for each match Id from the MatchID series, this has no relevance i just wanted

#to be able to do it

df['ID'] = [i[:7] for i in df['MatchID']]

#By default the head method returns the top five entries

df.head()
#The tail method returns the last rows in the DataFrame, by default it returns the last 5 rows

df.tail()
num_matches = len(df['MatchID'].unique())

print('Number of matches in data set:',  num_matches)
df.info()
#Subsets the dataframe to remove null examples

df = df[df.ADR.notnull()]
df.info()
del df['MatchID']

df['KDR'] = df.Kills/df.Deaths
#create a dict comprehension where values are percentages rounded to two decimal pts *100

map_counts = {i: round(float(len(df[df.Map ==i].ID.unique()))/num_matches, 2)*100 for i in df.Map.unique() }

plt.bar(left = range(0, len(map_counts.keys())), height =  map_counts.values())

plt.xticks(range(0, len(map_counts.keys())), map_counts.keys(), rotation=90)

plt.ylabel('Percentage of games played [%]')

plt.title('Percentage of games played on a given map')

plt.show()
#Percentage of games played on a given map

map_counts
'''Two subplots one underneath the other, nrows specifies the number of rows (no. of figures

in a given column) and nrows is the number of columns''' 

fig, axes = plt.subplots(nrows=2, ncols=1)

plt.suptitle('PMF and CMF of Kills in dataset')

# Plot the PMF

df.Kills.plot(ax=axes[0], kind='hist', bins =len(df.Kills.unique()),range=(0, 40), normed = True, width=0.5)



# Plot the CDF

df.Kills.plot(ax=axes[1], kind='hist', bins=len(df.Kills.unique()), normed = True, range=(0, 40), cumulative=True)



plt.show()
df.Kills.mean()
df.Kills.median()
df.Kills.plot(kind='box')

plt.show()
fig, axs = plt.subplots(nrows=2, ncols=1)

plt.suptitle('PMF & CMF of the KDR')

df.KDR.plot(ax = axs[0], kind = 'hist', normed =True, bins = 60, range = (0, 4), figsize = (8, 4))



df.KDR.plot(ax = axs[1], kind ='hist', bins = 60, normed =True, range=(0, 4), cumulative =True)

plt.show()
df.KDR.median()
df.KDR.mean()
df.KDR.max()
#Importing infinity from the math library

from math import inf

df_notinf = df[df['KDR'] != inf]
round(df_notinf.KDR.mean(), 2)
df_notinf.KDR.median()
df['KAST%'].describe()
df['KAST%'].plot(kind='box')

plt.ylabel('KAST%')

plt.show()
#We only want to look at columns which do not contain object type datatypes (strings), we are

#also uninterested in the player feature

cols = [i for i in list(df) if df[i].dtype !='O' and i !='Player']

'''Other approach to this is by looping:

for i in list(df):

    if df[i].dtype == 'O' or i=='Player':

        continue

    else:

        cols.append(i)'''

df[cols].plot(kind='box', figsize = (12, 6))

plt.yticks(range(0, int(max(df.ADR)), 10))

plt.title('Boxplots of numerical data in DataFrame')

plt.show()
cols = ['Rating', 'KDR']

df[cols].plot(kind='box', figsize= (10, 7))

plt.yticks(range(0, 27))

plt.title('Boxplots of Rating and KDR features')

plt.show()
#Dataset can be sliced by either loc, passing the names of indices  (: ==all) and the list of

#Columns of interest

df.loc[:,['Kills', 'Deaths', 'ADR', 'KAST%', 'Rating']].describe()
#Or via iloc whereby the columns are replaced by their location in the header (e.g the first

#column has location 0, second 1, and so forth)

corr = df.iloc[:,2:9].corr()

corr
plt.scatter( x= df_notinf['Kills'], y= df_notinf.Rating, marker ='o', c='red', s=1)

plt.xlabel('No. of kills')

plt.ylabel('Rating')

plt.show()



# I decided to calculate the log (base 10) of the following quantities for better scaling

plt.scatter( x=df_notinf['KAST%'], y= np.log(df_notinf.Rating), marker='o',s=1, c='blue')

plt.ylabel('Rating')

plt.xlabel('KAST%')

plt.show()



plt.scatter( x=np.log(df_notinf['KDR']/10), y= df_notinf.Rating, marker='o', s =1, c='green')

plt.ylabel('Rating')

plt.xlabel('KDR')

plt.show()
from matplotlib import cm as CM

gridsize =100

plt.hexbin(df_notinf.KDR/10, df_notinf['KAST%'], C=df_notinf.Rating, gridsize=gridsize, cmap=CM.jet, bins=None)



plt.ylim(0, df_notinf['KAST%'].max()+2)

plt.xlim(0, (df_notinf.KDR/10).max()+0.25)

plt.xlabel('KDR')

plt.ylabel('KAST%')



cb = plt.colorbar()

cb.set_label('Rating')



plt.show()