# Load standard libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Ploting Libraries

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

py.init_notebook_mode(connected=True)



# Render matplotlib plots inline

%matplotlib inline







# Read data into pandas dataframe

# df = pd.read_csv('../input/database.csv')

df = pd.read_csv('../input/database.csv', low_memory=False)
# Dimension of data

df.shape
# Available columns

df.columns
# Data Types

df.dtypes
# Count of different types in data set

df.dtypes.value_counts()
# Detailed infor on variables

df.info()

# It seems we have null in the data
# Quick look at the dataset

#df.head()

df.sample(10)
# Look at the statistical details of numeric variables

df.describe()
# Look at the statistical summary of categorical variables

dfcat = df.select_dtypes(include=["object"])



dfcat.describe()
# Seaborn Style

# Plot Height

# Check Height distribution

df['Height'].value_counts()
# Check nulls in Height data

df['Height'].isnull().sum()
# impute null with 0

df['Height'].fillna(0, inplace = True)
# Check null

df['Height'].isnull().sum()
# plot Height using seaborn distplot

plt.figure(figsize = (10,5))

sns.distplot(df[df['Height'] != 0 ]['Height'])

plt.title('Height Distribution', fontsize = 15)

plt.xlabel('Height', fontsize = 15)

plt.ylabel('Samples', fontsize = 15)
plt.figure(figsize = (10,5))

sns.distplot(np.log(df[df['Height'] != 0]['Height']+1))

plt.title('Logrithmic Height Distribution', fontsize = 15)

plt.xlabel('LogHeight', fontsize = 15)

plt.ylabel('Samples', fontsize = 15)
# Plot using matplotlib

plt.figure(figsize = (10,5))

plt.hist(df[df['Height'] != 0]['Height'])

plt.title('Matplot Stype Height Distribution', fontsize = 15)

plt.xlabel('Height', fontsize = 15)

plt.ylabel('Samples', fontsize = 15)

plt.show()
plt.subplot(1,2,1)

df[df['Height'] != 0]['Height'].plot.hist(bins = 50, figsize = (10,5), edgecolor = 'white')

plt.subplot(1,2,2)

np.log(df[df['Height'] != 0]['Height']+1).plot.hist(bins = 50, figsize = (19,5), edgecolor = 'white')



plt.show()
# Plotly Style ploting



layout = go.Layout(

    title = 'Height Distribution',

    xaxis = dict(

        title = 'Height'

    ),

    yaxis = dict(

        title = 'Frequency'

    )

)



trace1 = go.Histogram(

    x = np.log(df[df['Height'] != 0]['Height']+1)

)



data = [trace1]



fig = go.Figure(data = data, layout = layout)

py.iplot(fig)