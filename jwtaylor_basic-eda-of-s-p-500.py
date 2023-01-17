# import numpy, matplotlib as np, plt and set %matplotlib inline amongst other imports...

%pylab --no-import-all inline



# table and graphical visualisation

import pandas  as pd

import seaborn as sns



pd.options.display.float_format = '{:.2f}'.format
data = pd.read_csv('../input/all_stocks_1yr.csv', parse_dates = ['Date'])
# Explore nulls...

data.isnull().sum()
# Explore where the nulls are...

data.loc[data.isnull().sum(axis = 1).astype(bool)].nunique()
# Get dates...

dates = data.loc[data.isnull().sum(axis = 1).astype(bool)]['Date'].unique()



# Drop...

data.drop(np.arange(len(data))[data['Date'].isin(dates)], inplace = True)



# Re-check for nulls...

data.isnull().sum()
# Preview Data...

data.head()
# Verify count of each company using np.unique...

# Using a nested np.unique like this returns the unique count of days in

# the first array and the count of those unique days in the second array

np.unique(np.unique(data['Name'], return_counts = True)[1], return_counts = True)
# Verify number of companies...

data['Name'].nunique()
loc = data['Name'].isin(np.unique(data['Name'])[np.unique(data['Name'], return_counts = True)[1] < 250])

data.drop(loc.index[loc], inplace = True)
# Re-verify count of each company...

np.unique(np.unique(data['Name'], return_counts = True)[1], return_counts = True)
# Obtain all names to corresponding close...

name, close = data[data['Date'] == data['Date'].min()][['Name', 'Close']].T.values



# Map to dictionary...

base = {n : c for n, c in zip(name, close)}



# Use base to add Growth to data...

Base           = np.array(list(map(lambda x : base[x], data['Name'].values)))

data['Growth'] = data['Close'] / Base - 1
# Summary statistics...

data['Growth'].describe()
# change to an integer of your choice

# large n will cause the later visualisation to slow down

n = 8



# last day of the year for all companies only showing Date, Name and Growth sorted by Growth...

obj = data.loc[251::250][['Date', 'Name', 'Growth']].sort_values('Growth')



print('Largest fall in share price:')

display(obj.head(n))

print('Largest gain in share price:')

display(obj.tail(n))
lo_names = obj.head(n)['Name']

hi_names = obj.tail(n)['Name']

names    = [lo_names, hi_names]

pre      = ['Worst', 'Best']

title    = ' {} Performing Companies in the S&P 500'.format(n)



for i in range(2):

    # subset of data...

    sub = data[data['Name'].isin(names[i])].groupby(['Name', 'Date'])['Growth'].sum().reset_index()

    

    # create figure...

    plt.figure(figsize(20, 10))

    

    sns.pointplot(x = 'Date', y = 'Growth', hue = 'Name', data = sub, scale = 0.3)

    

    # remove the 250 x ticks...

    plt.xticks([])

    

    # add title and x / y labels 

    plt.title(pre[i] + title  , size = 25)

    plt.xlabel('Time (1 Year)', size = 20)

    plt.ylabel('Growth'       , size = 20)

    

    # always add a legend and make sure it is readable

    plt.legend(markerscale = 2, prop = {'size' : 15})

    

    plt.show()
cor = data.pivot('Date', 'Name', 'Growth').corr()

sns.heatmap(cor)

plt.show()
def get_correlation(cor, upper = 0.99, lower = -0.99, blank = False):

    corr = {}

    for i in cor:

        obj = cor[i][cor[i].apply(abs).argsort()[::-1]]

        obj = obj[(obj > upper) | (obj < lower)]

        obj = obj[obj.index != i]

        if len(obj) == 0:

            if blank: corr[i] = None

        else:

            corr[i] = {}

            for j in obj.index:

                corr[i][j] = cor[i].loc[j]

    return corr
corr = get_correlation(cor, lower = -1)

corr
sns.heatmap(cor.loc[list(corr), list((corr))], annot = True, fmt = '.2f')

plt.xlabel('')

plt.ylabel('')

plt.show()
# And try for negative correlation...

corr = get_correlation(cor, upper = 1, lower = -0.95)

corr
sns.heatmap(cor.loc[list(corr), list((corr))], annot = True, fmt = '.2f')

plt.xlabel('')

plt.ylabel('')

plt.show()
obj['Growth'].sum(), obj['Growth'].mean()