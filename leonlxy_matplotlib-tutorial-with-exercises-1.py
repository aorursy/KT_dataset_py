# Before we start, let's import the libraries that will be used in later of this tutorials.

import matplotlib.pyplot as plt



import numpy as np

import pandas as pd

import seaborn as sns

# Don't worry if you don't know how to use numpy, pandas and seaborn, I will explain them in details in other notebooks.
# Before we start anything, let's create some sample data for our plotting.
# set random seed so that you could have the exactly same results as mine.

np.random.seed(0)



df = pd.DataFrame(data={'a':np.random.randint(0, 100, 30),

                        'b':np.random.randint(0, 100, 30),

                        'c':np.random.randint(0, 100, 30)})

df.head()
# Let's create a figure and call it fig.

fig = plt.figure()

# This will return an empty figure.
# Let's create a figure with figsize (15, 8) and also call it fig (thus overwriting the reference to the previous fig).

# The 15x8 figsize is arbitrary, but I use it as a standard size to work with for visibility.

fig = plt.figure(figsize=(15,8))
fig = plt.figure(figsize=(15,8))

ax = plt.subplot(1,1,1) # (rows, columns, and location)

                        # this would create a 1x1 grid of subplots

                        # and choose axes #1
fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(2,1,1) # this would create a 2x1 grid of subplots

                         # and choose axes #1

ax2 = plt.subplot(2,1,2) # this would create a 2x1 grid of subplots

                         # and choose axes #2
fig, ax = plt.subplots(2, 1, figsize=(15,8)) # This creates a figure of size 15x8 with

                                             # a 2x1 grid of subplots.
fig, ax = plt.subplots(2, 1, figsize=(15,8)) # This creates a figure of size 15x8 with

                                             # a 2x1 grid of subplots.

ax[0] # The top axes

ax[1] # The bottom axes
fig, ax = plt.subplots(2, 2, figsize=(15,8)) # This creates a figure of size 15x8 with

                                             # a 2x1 grid of subplots.



ax[0][0].plot(df.index.values, df['a']) # The top-left axes

ax[0][1].plot(df.index.values, df['b']) # The top-right axes

ax[1][0].plot(df.index.values, df['c']) # The bottom-left axes

ax[1][1].plot(df.index.values, range(len(df))) # The bottom-right axes
fig, ax = plt.subplots(1,1, figsize=(15,8))



x = df.index.values # The index the dataframe we created up above. Equivalent to [0, 1, ..., 28, 29]

y = df['a'] # Column 'a' from df.



ax.plot(x, y)
# The above plot can be generated without creating the variables

# x and y by passing the values directly to the function.



fig, ax = plt.subplots(2,1, figsize=(15,8))



ax[0].plot(df.index.values, df['a'])

ax[1].plot(df.index.values, df['b'])
fig, ax = plt.subplots(1,1, figsize=(15,8))



x = df.index.values # The index the dataframe we created up above. Equivalent to [0, 1, ..., 28, 29]

y1 = df['a'] # Column 'a' from df.

y2 = df['b'] # Column 'a' from df.



ax.plot(x, y1)

ax.plot(x, y2)
sns.set_style('darkgrid') # setting the plotting style

                          # we only need to call this once,

                          # usually before we start plotting.



fig, ax = plt.subplots(1,1, figsize=(15,8))



ax.plot(df.index.values, df['a'])

ax.plot(df.index.values, df['b'])
sns.set_style('darkgrid') # setting the plotting style



fig, ax = plt.subplots(1,1, figsize=(15,8))



ax.plot(df.index.values, df['a'], color='red', ls='-.')

ax.plot(df.index.values, df['b'], color='orange', lw=10)

ax.plot(df.index.values, df['c'], color='yellow', lw=1, marker='o')
fig, ax = plt.subplots(1,1, figsize=(15,8))



ax.plot(df.index.values, df['a'], label='Line A') # add the label

ax.plot(df.index.values, df['b'], label='Line B') # kwarg to each

ax.plot(df.index.values, df['c'], label='Line C') # function



ax.legend(loc='best') # and now call the ax.legend() function

            # it will read all of the labels from graphical

            # objects under ax
fig, ax = plt.subplots(3,1, figsize=(15,8))



ax[0].plot(df.index.values, df['a'], label='Line A') # Top

ax[1].plot(df.index.values, df['b'], label='Line B') # Middle

ax[2].plot(df.index.values, df['c'], label='Line C') # Bottom



ax[0].legend(loc=4) # This will create a legend for ax[0] in the bottom-right.

ax[1].legend(loc=6) # This will create a legend for ax[1] centre-left.



# Also note that all lines will default to the first color in the default color cycle--blue.
# Start your answer from here
# My Sample Answer

fig, ax = plt.subplots(2, 1, figsize=(15,8))



ax[0].plot(df.index.values, df['a'], c='green')

ax[0].plot(df.index.values, df['b'], c='orange')

ax[0].legend(loc=9) # "9": upper center



ax[1].plot(df.index.values, df['c'], marker='o', lw=0) # set line width = 0, means no visuable line
fig, ax = plt.subplots(1, 1, figsize=(15,8))



bar_kwargs = {'color':'tomato', 'alpha':0.5}



ax.bar(df.index.values, df['a'], label='a', **bar_kwargs)

ax.legend()
# Start your answer from here
# My Sample Answer

fig, ax = plt.subplots(3, 1, figsize=(15,18))



ax[0].bar(df.index.values, df['a'])

for i in range(df.shape[0]):

    ax[0].text(i, df['a'][i]+1, df['a'][i], horizontalalignment='center')

ax[0].legend('a')

    

ax[1].bar(df.index.values, df['b'])

for i in range(df.shape[0]):

    ax[1].text(i, df['b'][i]+1, df['b'][i], horizontalalignment='center')

ax[1].legend('b')



ax[2].bar(df.index.values, df['a'])

ax[2].bar(df.index.values, df['b'], bottom=df['a'])

for i in range(df.shape[0]):

    ax[2].text(i, df['a'][i]+df['b'][i]+1, df['a'][i]+df['b'][i], horizontalalignment='center')

ax[2].legend(['a','b'])
np.random.seed(0)



fig, ax = plt.subplots(figsize=(15,7))



ax.plot(df.index.values, df['a'], marker='^')

ax.set_title('This is the Title')

ax.set_ylabel('This is the Y Axis')

ax.set_xlabel('This is the X Axis', fontsize=20)



ax.set_xticks(df.index.values)

ax.set_xticklabels(np.random.randint(1,30,30), fontsize=15, color='red')



ax.legend()

fig.tight_layout()
# mock data

np.random.seed(0)



turnover_data = pd.DataFrame({'boardid' : ['DAY', 'DAY_X', 'DAY_U', 'TSE'], 

                              'turnover' : np.random.randint(1e6, 1e9, 4)})



# sort by turnover value

turnover_data = turnover_data.sort_values(by='turnover').reset_index().drop('index', axis=1)



# convert value to Million unit for easy-reading

turnover_data['turnover_simplified'] = turnover_data['turnover'] // 1000000



# market share

turnover_data['market_share'] = round((turnover_data['turnover'] / sum(turnover_data['turnover'])*100), 1)



turnover_data
# Start your answer from here
# My Sample Answer

fig, ax1 = plt.subplots(figsize=(15,10))



ax1.bar(turnover_data['boardid'], turnover_data['turnover'], width=0.5, color='lightgreen')

for i in range(turnover_data.shape[0]):

    ax1.text(turnover_data['boardid'][i], turnover_data['turnover'][i]+1e7, str(turnover_data['turnover_simplified'][i])+'M', fontsize=15, horizontalalignment='center')



ax1.set_title('Turnover Data', fontsize=20)

ax1.set_xticklabels(turnover_data['boardid'], fontsize=15);



ax1.tick_params(labelsize=15, axis='y')



ax1.set_ylabel('Turnover Values(Million)', fontsize=20)

ax1.set_ylim(0, max(turnover_data['turnover']+1e8))



# Format ax1 y axis - method 1

vals = ax1.get_yticks()

ax1.set_yticklabels([str(x/1000000)+'M' for x in vals])





ax2 = ax1.twinx()

ax2.plot(turnover_data['boardid'], turnover_data['market_share'], c='red', lw=4, marker='o')

for i in range(turnover_data.shape[0]):

    ax2.text(turnover_data['boardid'][i], turnover_data['market_share'][i]+1, str(turnover_data['market_share'][i])+'%', fontsize=15, horizontalalignment='center', color='black')



ax2.set_ylabel('Market Shares(%)', fontsize=20)

ax2.tick_params(labelsize=15, axis='y', rotation=30)



# Format ax2 y asix - method 1

# ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y/100)))



# Format ax2 y asix - method 2

vals = ax2.get_yticks()

ax2.set_yticklabels([str(x)+'%' for x in vals]);