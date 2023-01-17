import pandas as pd

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

from scipy import stats as sts

import datetime as dt

import os



print(os.listdir("../input/"))
data = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")
data.head()
data.dtypes
data.isnull().any()
data['Fatalities'].fillna(0, inplace = True)

data['Aboard'].fillna(0, inplace = True)

data['Ground'].fillna(0, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].dt.strftime("%m/%d/%Y")

data['Date'].head(5)
data['Year'] = pd.DatetimeIndex(data['Date']).year

data['Year'].head(5)
data['Survived'] = data['Aboard'] - data['Fatalities']

data['Survived'].fillna(0, inplace = True)
data.head(5)
data.describe()
matplotlib.rcParams['figure.figsize'] = (20, 10)

sns.set_context('talk')

sns.set_style('whitegrid')

sns.set_palette('tab20')
total_crashes_year = data[['Year', 'Date']].groupby('Year').count()

total_crashes_year = total_crashes_year.reset_index()

total_crashes_year.columns = ['Year', 'Crashes']
sns.lineplot(x = 'Year', y = 'Crashes', data = total_crashes_year)

plt.title('Total Airplane Crashes per Year')

plt.xlabel('')
total_crashes_year[total_crashes_year['Crashes'] > 80]
#summarise

pcdeaths_year = data[['Year', 'Fatalities']].groupby('Year').sum()

pcdeaths_year.reset_index(inplace = True)
# Plot

sns.lineplot(x = 'Year', y = 'Fatalities', data = pcdeaths_year)

plt.title('Total Number of Fatalities by Air Plane Crashes per Year')

plt.xlabel('')
# summarise

abrd_per_year = data[['Year', 'Aboard']].groupby('Year').sum()

abrd_per_year = abrd_per_year.reset_index()
# plot

sns.lineplot(x = 'Year', y = 'Aboard', data = abrd_per_year)

plt.title('Total of People Aboard Airplanes per Year')

plt.xlabel('')

plt.ylabel('Count')
sts.pearsonr(data.Fatalities, data.Aboard)
sts.spearmanr(data.Fatalities, data.Aboard)
sns.regplot(x = 'Aboard', y = 'Fatalities', data = data, scatter_kws=dict(alpha = 0.3), line_kws=dict(color = 'red', alpha = 0.5))

plt.title('Fatalities x People Aboard')
#summarise

FSG_per_year = data[['Year', 'Fatalities', 'Survived', 'Ground']].groupby('Year').sum()

FSG_per_year = FSG_per_year.reset_index()
#plot

sns.lineplot(x = 'Year', y = 'Fatalities', data = FSG_per_year)

sns.lineplot(x = 'Year', y = 'Survived', data = FSG_per_year)

sns.lineplot(x = 'Year', y = 'Ground', data = FSG_per_year)

plt.legend(['Fatalities', 'Survival', 'Ground'])

plt.xlabel('')

plt.ylabel('Count')

plt.title('Fatalities vs Survived vs Killed on Ground per Year')
data['Ground'].max()
data[data['Ground'] == 2750]
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score
text_data = data['Summary'].dropna()

text_data = pd.DataFrame(text_data)
documents = list(text_data['Summary'])

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(documents)
true_k = 7

model = KMeans(n_clusters=true_k, max_iter=100, n_init=1)

model.fit(X)
print ('Most Common Terms per Cluster:')

order_centroids = model.cluster_centers_.argsort()[:,::-1]

terms = vectorizer.get_feature_names()



for i in range(true_k):

    print('Cluster %d:' % i)

    for ind in order_centroids[i, :10]:

        print ('%s' % terms[ind]),

    print