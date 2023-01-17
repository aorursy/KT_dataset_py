import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization



sns.set(font_scale=2.5)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import json

# Read in json file as a string

with open('/kaggle/input/currentData.json') as file:

    data = file.read()



# Parse JSON into a python object

decoder = json.JSONDecoder()

d = decoder.decode(data)['counts']



# The data does not have location counts for some early days. 

# In this kernel I am only using the location data, so I am going to ignore days without location data.

for i, e in enumerate(d):

    if 'location' in e:

        first = i

        break



# Read this python object and parse it into a DataFrame

df = pd.DataFrame()

for day in d[first:]:

    df = pd.concat([df, pd.DataFrame(day['location']).assign(date=day['label'])])



# Reset index in the DataFrame

df = df.reset_index(drop=True)



# Fix cases table

# Some cases are missing so I will drop them

df = df[~(df['cases'] == '***')]

# Convert cases to an integer Series

df['cases'] = df['cases'].map(lambda x: int(x.replace(",", "")))





# Fix population

# Replace unknown populations with 0 (I will ignore these in later analyses)

df['population'] = df['population'].replace('Not Available', 0).replace('', 0)

# Convert population to an integer Series

df['population'] = df['population'].map(lambda x: int(str(x).replace(',','')))



# Convert date to a DateTime Series

df['date'] = pd.to_datetime(df['date'], format = '%b %d').map(lambda x: x.replace(year=2020))





print(df['date'].max())

# Data is now ready to use in DataFrame df.

df.head()



# plotPerColumnDistribution(df[~(df['city'] == 'All of Orange County') & (df['date'] == df['date'].max())], 10, 10)

# plotCorrelationMatrix(df[~(df['city'] == 'All of Orange County') & (df['date'] == df['date'].max())], 10)

# plotScatterMatrix(df[~(df['city'] == 'All of Orange County') & (df['date'] == df['date'].max())], 10, 20)
# Make some initial charts

plt.figure(figsize=(10,10))



# Pick only real cities, sort by which city has the maximum cases

all_oc = df[df['city'] == 'All of Orange County']



# Let's graph the three worst cities so far

delta = all_oc.shape[0]

x = range(delta)

y = all_oc['cases']



[a, b] = np.polyfit(x, y, 1)



ax = sns.scatterplot(x, y, s = 100)

ax.plot(x, np.poly1d([a, b])(x), linewidth=5)



ax.set_title('COVID-19 growth in OC')

ax.set_ylabel('Cases')

ax.set_xlabel('Days since March 27')



print(f'Orange County has {a:.1f} new cases every day')
# Make some initial charts

plt.figure(figsize=(10,10))



# Pick only real cities, sort by which city has the maximum cases

d2 = df[~(df['city'].isin(['All of Orange County', 'Unknown**', 'Other*']))].groupby('city')['cases'].max().sort_values(ascending=False).index



# Let's graph the three worst cities so far

for city in d2[:3]:

    delta = sum(df['city'] == city)

    x = range(delta)

    y = df[df['city'] == city]['cases']



    [a, b] = np.polyfit(x, y, 1)



    ax = sns.scatterplot(x, y, label=city, s = 70)

    ax.plot(x, np.poly1d([a, b])(x), linewidth=2)

    

    print(f'{city} has {a:.1f} new cases every day')



ax.set_title('COVID-19 growth in OC: Top 3 cities'.format(city))

ax.set_ylabel('Cases')

ax.set_xlabel('Days since March 27')



ax.text(12.5,25, 'fit using')

ax.text(12.5,10, 'linear regression')



ax.legend(markerscale=1.5)



None
l1 = all_oc['cases'].reset_index(drop=True)

first_derivative = [0] + [l1[i + 1] - l1[i] for i in range(l1.size - 1)]





plt.figure(figsize=(10,10))

ax = sns.lineplot(x = range(len(first_derivative)), y = first_derivative)

ax.set_title('Cases per day - Orange County')

ax.set_ylabel('New cases per day')

ax.set_xlabel('Days since March 27')





plt.figure(figsize=(10,10))



ax = sns.lineplot(x=range(len(first_derivative)), y=pd.Series(first_derivative).rolling(5).mean())

ax.set_title('5-day moving average of new cases')

ax.set_ylabel('Mean new cases over last 5 days')

ax.set_xlabel('Days since March 27')
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    # filename = df.dataframeName

    filename = 'covid'

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
