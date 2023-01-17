import pandas as pd

data = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = 'latin-1')
data.head(5)
data.columns
# renaming the columns, removing the unwanted characters 

renamed_data = data.rename(columns = {'Track.Name' : 'track', 'Artist.Name' : 'artist', 'Beats.Per.Minute' : 'beatspermin', 'Loudness..dB..' : 'loudness', 'Valence.' : 'valence', 'Length.' : 'length', 'Acousticness..' : 'Acoustic', 'Speechiness.' : 'Speechiness'})
renamed_data.head()
# DataFrame of tracks having popularity rate above 90

renamed_data.loc[renamed_data.Popularity >= 90] 
# checking for any null value

renamed_data.isnull().any()
# Description of Genre

renamed_data.Genre.describe()
# Description of artist

renamed_data.artist.describe()
# total Individual artist  

renamed_data.artist.unique()
# total number of songs by each artist in 2019

renamed_data.artist.value_counts()
# displaying total number of songs by each artist using pie chart

import matplotlib.pyplot as plt

labels = renamed_data.artist.value_counts().index

sizes = renamed_data.artist.value_counts().values



colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','cyan', 'green', 'black','yellow']



plt.figure(figsize = (12,12))

plt.pie(sizes, labels=labels, colors=colors)

autopct=('%1.1f%%')

plt.axis('equal')

plt.show()
# Grouping 'Genre' and 'Artist' by popularity rates

renamed_data.groupby(['Genre', 'artist']).apply(lambda p : p.loc[p.Popularity.idxmax()])
# scatter matrix

from pandas.plotting import scatter_matrix

scatter_matrix(renamed_data)

plt.gcf().set_size_inches(30, 30)

plt.show()
# checking for skewed data

skew = renamed_data.skew()

print(skew)
# distribution of 'Liveness' column

import seaborn as sns

sns.distplot(a = renamed_data['Liveness'], kde = False)
# Applying boxcox transform to 'liveness' column and foring it to normal distribution

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

transform = np.asarray(renamed_data['Liveness'].values)

data_transformed = stats.boxcox(transform)[0]

plt.hist(data_transformed, bins = 10)
# Applying boxcox transform on Popularity column and applying KDE 

pop_transformed = np.asarray(renamed_data['Popularity'].values)

data_transformed = stats.boxcox(pop_transformed)[0]



sns.distplot(renamed_data['Popularity'], bins = 10, kde = True)

#sns.distplot(data_transformed, bins = 10)
sns.distplot(data_transformed, bins = 10, kde = True)
transfor = np.asarray(renamed_data['Acoustic'].values)

transformed = stats.boxcox(transfor)[0]

sns.distplot(transformed, bins = 10, kde = True)
# Applying pearson correlation co-efficient on the data and displaying it into a heatmap

correlation = renamed_data.corr(method = 'pearson')



plt.figure(figsize = (12, 6))

plt.title('Correlation heatmap')

sns.heatmap(correlation, annot = True)
# Analysing the relationship between energy and loudness using regression line

fig=plt.subplots(figsize=(7,7))

sns.regplot(x = 'Energy',y = 'loudness', data = renamed_data)
# Analysing the relationship between 'Speechiness' and 'beatsperminute' using regression line

#https://seaborn.pydata.org/generated/seaborn.regplot.html

fig=plt.subplots(figsize=(7,7))

sns.regplot(x = 'Speechiness',y = 'beatspermin', data = renamed_data)
# Draw a plot of two variables with bivariate and univariate graphs

# https://seaborn.pydata.org/generated/seaborn.jointplot.html

sns.jointplot(x = renamed_data['loudness'], y = renamed_data['Energy'], kind = 'kde')
# Draw a plot of two variables with bivariate and univariate graphs

# https://seaborn.pydata.org/generated/seaborn.jointplot.html

sns.jointplot(x = renamed_data['Speechiness'], y = renamed_data['beatspermin'], kind = 'kde')