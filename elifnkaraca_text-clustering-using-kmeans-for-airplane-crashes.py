# This Python 3 environment comes with many helpful analytics libraries installed



import pandas as pd #linear algebra

import seaborn as sns #visualization tool

import matplotlib

from matplotlib import pyplot as plt

from scipy import stats as sts  #data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

from collections import Counter

import os



print(os.listdir("../input/"))
data = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")
data.head()
data.info()
data.isnull().any()
data['Fatalities'].fillna(0, inplace = True)

data['Aboard'].fillna(0, inplace = True)

data['Ground'].fillna(0, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])

data['Date'] = data['Date'].dt.strftime("%d/%m/%Y")

data['Date'].head()
data['Year'] = pd.DatetimeIndex(data['Date']).year

data['Year'].head()
data['Survived'] = data['Aboard'] - data['Fatalities']

data['Survived'].fillna(0, inplace = True)
data.head()
matplotlib.rcParams['figure.figsize'] = (20, 10)

sns.set_context('talk')

sns.set_style('whitegrid')

sns.set_palette('tab20')
total_crashes_year = data[['Year', 'Date']].groupby('Year').count()

total_crashes_year = total_crashes_year.reset_index()

total_crashes_year.columns = ['Year', 'Crashes']
sns.lineplot(x = 'Year', y = 'Crashes', data = total_crashes_year)

plt.title('Total Airplane Crashes per Year')

plt.xlabel('years')

plt.ylabel('number of crashes')
pcdeaths_year = data[['Year', 'Fatalities']].groupby('Year').sum()

pcdeaths_year.reset_index(inplace = True)
# Plot

sns.lineplot(x = 'Year', y = 'Fatalities', data = pcdeaths_year)

plt.title('Total Number of Fatalities by Air Plane Crashes per Year')

plt.xlabel('Fatalities')

plt.xlabel('Years')
# summarise

abrd_per_year = data[['Year', 'Aboard']].groupby('Year').sum()

abrd_per_year = abrd_per_year.reset_index()
# plot

sns.lineplot(x = 'Year', y = 'Aboard', data = abrd_per_year)

plt.title('Total of People Aboard Airplanes per Year')

plt.xlabel('Years')

plt.ylabel('Count')
#summarise

FSG_per_year = data[['Year', 'Fatalities', 'Survived', 'Ground']].groupby('Year').sum()

FSG_per_year = FSG_per_year.reset_index()
#plot

sns.lineplot(x = 'Year', y = 'Fatalities', data = FSG_per_year, color = 'green')

sns.lineplot(x = 'Year', y = 'Survived', data = FSG_per_year, color = 'blue')

sns.lineplot(x = 'Year', y = 'Ground', data = FSG_per_year, color = 'red')

plt.legend(['Fatalities', 'Survival', 'Ground'])

plt.xlabel('Years')

plt.ylabel('Count')

plt.title('Fatalities vs Survived vs Killed on Ground per Year')
oper_list = Counter(data['Operator']).most_common(10)

operators = []

crashes = []

for tpl in oper_list:

    if 'Military' not in tpl[0]:

        operators.append(tpl[0])

        crashes.append(tpl[1])

print('Top 10 the worst operators')

pd.DataFrame({'Count of crashes' : crashes}, index=operators)
loc_list = Counter(data['Location'].dropna()).most_common(10)

locs = []

crashes = []

for loc in loc_list:

    locs.append(loc[0])

    crashes.append(loc[1])

print('Top 10 the most dangerous locations')

pd.DataFrame({'Crashes in this location' : crashes}, index=locs)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.metrics import adjusted_rand_score

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
text_data = data['Summary'].dropna()

text_data = pd.DataFrame(text_data)

# for reproducibility

random_state = 0 
documents = list(text_data['Summary'])

vectorizer = TfidfVectorizer(stop_words='english') # Stop words are like "a", "the", or "in" which don't have significant meaning

X = vectorizer.fit_transform(documents)
model = MiniBatchKMeans(n_clusters=5, random_state=random_state)

model.fit(X)
model.cluster_centers_
# predict cluster labels for new dataset

model.predict(X)



# to get cluster labels for the dataset used while

# training the model (used for models that does not

# support prediction on new dataset).

model.labels_
print ('Most Common Terms per Cluster:')



order_centroids = model.cluster_centers_.argsort()[:,::-1] #sort cluster centers by proximity to centroid

terms = vectorizer.get_feature_names()



for i in range(5):

    print("\n")

    print('Cluster %d:' % i)

    for j in order_centroids[i, :10]: #replace 10 with n words per cluster

        print ('%s' % terms[j]),

    print
# reduce the features to 2D

pca = PCA(n_components=2, random_state=random_state)

reduced_features = pca.fit_transform(X.toarray())



# reduce the cluster centers to 2D

reduced_cluster_centers = pca.transform(model.cluster_centers_)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=model.predict(X))

plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
print("\n")

print("Prediction")



Y = vectorizer.transform(["engine failure"])

prediction = model.predict(Y)

print(prediction)



Y = vectorizer.transform(["terrorism"])

prediction = model.predict(Y)

print(prediction)

 