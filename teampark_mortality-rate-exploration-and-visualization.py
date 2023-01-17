import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline



sns.set(style = 'darkgrid')
m = pd.read_csv('../input/mort.csv')

list(m.columns)
tokeep = ['Location',

 'FIPS',

 'Category',

 'Mortality Rate, 1980*',

 'Mortality Rate, 1985*',

 'Mortality Rate, 1990*',

 'Mortality Rate, 1995*',

 'Mortality Rate, 2000*',

 'Mortality Rate, 2005*',

 'Mortality Rate, 2010*',

 'Mortality Rate, 2014*',]



m = m[(m.Location.str.contains(',') == True)]

m.reset_index(drop=True, inplace=True)
x = 0

while x < len(m.columns):

    if m.columns[x] not in tokeep:

        m = m.drop(m.columns[x],1)

    x=x+1

m.info()
m['County'] = ''

m['State'] = ''



county = []

state = []



x=0

while x<len(m):

    p = m['Location'][x].index(',')

    county.append(m['Location'][x][:p])

    state.append(m['Location'][x][p+2:])

    x=x+1

    

m['County'] = county

m['State'] = state

m.info()
n = pd.melt(m, id_vars = ['Location','FIPS','Category','County','State'], value_vars = ['Mortality Rate, 1980*',

                                                                                    'Mortality Rate, 1985*',

                                                                                   'Mortality Rate, 1990*',

                                                                                   'Mortality Rate, 1995*',

                                                                                   'Mortality Rate, 2000*',

                                                                                   'Mortality Rate, 2005*',

                                                                                   'Mortality Rate, 2010*',

                                                                                   'Mortality Rate, 2014*',])

n=n.rename(columns = {'value':'Mortality Rate'})



x=0

year = []

while x<len(n):

    year.append(n['variable'][x][16:20])

    x=x+1

    

n['Year'] = year

n.Year = n.Year.astype(float)

n=n.drop('variable',1)
y = sorted(list(n['State'].unique()))



plt.figure(figsize = (15,3))

sns.violinplot(y = 'Mortality Rate', x = 'State', data = n[(n['Year'] == 2014)], inner = 'quartile', 

               palette = 'Set2', order = y)
n['Sk Mortality Rate']= np.log1p(n['Mortality Rate'])



y = sorted(list(n['State'].unique()))



plt.figure(figsize = (3,15))

sns.violinplot(x = 'Sk Mortality Rate', y = 'State', data = n[(n['Year'] == 2014)], inner = 'quartile', 

               palette = 'Set2', order = y)
o = pd.pivot_table(n[(n['Year'] == 2014)], values = 'Mortality Rate', index = ['County','Year','State'], 

                   columns = 'Category', aggfunc = np.mean)

    



ocols = ['Cardiovascular',

 'Chronic resp',

 'Chronic liver',

 'Diabetes',

 'Diarrhea',

 'Digestive diseases',

 'Non Natural',

 'HIV/AIDS and TB',

 'Maternal disorders',

 'Mental disorders',

 'Musculoskeletal disorders',

 'Tropical diseases',

 'Neonatal disorders',

 'Neoplasms',

 'Neurological disorders',

 'Nutritional deficiencies',

 'Other communicable',

 'Other non-communicable',

 'Violence',

 'Transport injuries',

 'Unintentional injuries']





o.columns = ocols

o.head()
corr = o.corr()

fg, ax = plt.subplots(figsize = (11,9))



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask = mask, linewidths = .5, square = True)
sns.set(style = 'ticks')

p = (sns.jointplot(o['Neoplasms'], o['Cardiovascular'],

                  stat_func=None,edgecolor="w",xlim = (0,550), ylim = (0,550))

     .plot_joint(sns.kdeplot, zorder = 1, n_levels = 10))
p=o.ix[:,:5]

sns.set(style = 'darkgrid')

g = sns.PairGrid(p)

g.map_upper(plt.scatter, s = 5)

g.map_diag(plt.hist, lw = 0, edgecolor = 'w')

g.map_lower(sns.kdeplot, cmap = 'Blues_d')



g.set(ylim = (0,None))

g.set(xlim = (0,None))
cardio = n[(n['Category'] == 'Cardiovascular diseases')]

cardio = cardio[['Year','State','Mortality Rate']]

cardio = cardio.groupby(['Year','State']).mean()

cardio.reset_index(level=0, inplace=True)

cardio.reset_index(level=0, inplace=True)

cardio.head()
grid = sns.FacetGrid(cardio, col = 'State', hue = 'State', col_wrap = 5, size = 2)



grid.map(plt.plot, 'Year', 'Mortality Rate',ms = 4, marker ='o')



grid.set(xlim=(1980, 2014), ylim = (0,None))



grid.set_xticklabels(rotation = 45)



grid.fig.tight_layout(w_pad = 1)
cardio = pd.pivot_table(cardio, values = 'Mortality Rate', index = ['State'], 

                   columns = 'Year', aggfunc = np.mean)



cardio.reset_index(level=0, inplace=True)
cardio.columns = ['State','One','Two','Three','Four','Five','Sx','Seven','Eight']

#changing columns because I could not call when the column name was a year number
cardio['Delta'] = ''

delta = []

x=0

while x < len(cardio):

    y = cardio['One'][x] 

    z = cardio['Eight'][x]

    delta.append((z-y)/y)

    

    x=x+1



cardio['Delta'] = delta

    

cardio.head()
sns.set(style="darkgrid")



#fg, ax = plt.subplots(figsize = (11,9))



g = sns.PairGrid(cardio.sort_values(by = 'Delta', ascending = True), x_vars = 'Delta', y_vars = 'State', size = 10, aspect = .4)



g.map(sns.stripplot, size = 10, orient = 'h', palette = 'coolwarm', edgecolor = 'w')



g.set(xlabel='', ylabel='')



titles = ['1980 - 2014 % Change']



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    ax.xaxis.grid(False)

    ax.yaxis.grid(True)