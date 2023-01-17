import pandas as pd

from pylab import rcParams



user_review = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

google_apps = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
google_apps.head()
google_apps.shape
total = google_apps.isnull().sum().sort_values(ascending=False)

percent = (google_apps.isnull().sum()/google_apps.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(6)
google_apps.dropna(how ='any', inplace = True)
total = google_apps.isnull().sum().sort_values(ascending=False)

percent = (google_apps.isnull().sum()/google_apps.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(6)
print(google_apps.shape)
google_apps.info()
google_apps.isnull().sum()
import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import seaborn as sns

sns.set()
google_apps.Price.unique()
google_apps.Price = google_apps.Price.apply(lambda x: str(x).replace("$",""))

google_apps.Price.unique()
google_apps.Size = google_apps.Size.apply(lambda x: str(x).replace("M",""))

google_apps.Size.unique()
print(google_apps.shape)

google_apps = google_apps.drop_duplicates(subset=['App'], keep = 'first')

print(google_apps.shape)
google_apps.Reviews = pd.to_numeric(google_apps.Reviews, errors='coerce')

google_apps.Price = pd.to_numeric(google_apps.Price, errors='coerce')

google_apps.Rating = pd.to_numeric(google_apps.Rating, errors='coerce')

google_apps.Size = pd.to_numeric(google_apps.Size, errors='coerce')

google_apps.dtypes
rcParams['figure.figsize'] = 11.7,8.27

g = sns.kdeplot(google_apps.Rating, color="Red", shade = True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

plt.title('Distribution of Rating',size = 20)
rcParams['figure.figsize'] = 11.7,8.27

g = sns.kdeplot(google_apps.Reviews, color="Blue", shade = True)

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Distribution of Reviews',size = 20)
plt.figure(figsize=(30,5))

fig = sns.countplot(data=google_apps, x="Installs", palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('Count Installation By User',size = 30)

plt.show(fig)
plt.figure(figsize=(30,5))

fig = sns.countplot(data=google_apps, x="Type", palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('Free Apps vs. Paid Apps',size = 30)

plt.show(fig)
labels =google_apps['Type'].value_counts(sort = True).index

sizes = google_apps['Type'].value_counts(sort = True)





colors = ["palegreen","orangered"]

explode = (0.1,0)

 

rcParams['figure.figsize'] = 8,8



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Free Apps vs. Paid Apps (By Persentation)',size = 20)

plt.show()
google_apps['Type'].unique()
plt.figure(figsize=(30,5))

fig = sns.countplot(data=google_apps, x="Android Ver", palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('Kind of Android Version',size = 30)

plt.show(fig)
plt.figure(figsize=(30,5))

fig = sns.countplot(data=google_apps, x="Content Rating", palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('Kind of Content Rating',size = 30)

plt.show(fig)
plt.figure(figsize=(30,15))

fig = sns.countplot(data=google_apps, x="Genres", palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('Kind of Apps Genres',size = 30)

plt.show(fig)
sorted_by_rating = google_apps.sort_values(by=['Rating'], ascending=False)

sorted_by_rating.head()
plt.figure(figsize=(30,10))

fig = sns.barplot(x=sorted_by_rating['App'][:20], y=sorted_by_rating['Rating'][:20], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('App vs. Rating',size = 30)

plt.tight_layout()

plt.show(fig)
sorted_by_reviews = google_apps.sort_values(by=['Reviews'], ascending=False)

sorted_by_reviews.head()
plt.figure(figsize=(30,10))

fig = sns.barplot(x=sorted_by_reviews['App'][:20], y=sorted_by_reviews['Reviews'][:20], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.title('App vs. Reviews',size = 30)

plt.tight_layout()

plt.show(fig)
google_apps['Installs'].unique()
google_apps.Installs = google_apps.Installs.apply(lambda x: str(x).replace("+",""))

google_apps.Installs = google_apps.Installs.apply(lambda x: x.replace(',',''))

google_apps.Installs = google_apps.Installs.apply(lambda x: int(x))

google_apps.Installs.unique()
sorted_by_install = google_apps.sort_values(by=['Installs'], ascending=False)

sorted_by_install.head()
Sorted_value = sorted(list(google_apps['Installs'].unique()))
google_apps['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
google_apps['Installs'].head()
plt.figure(figsize=(30,10))

fig = sns.barplot(x=sorted_by_install['Category'], y=sorted_by_install['Installs'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.tight_layout()

plt.title('Category VS Installs',size = 40)

plt.show(fig)
sorted_by_type = google_apps.sort_values(by=['Type'], ascending=False)

sorted_by_type.head()
plt.figure(figsize=(20,5))

fig = sns.barplot(y=sorted_by_type['Installs'], x=sorted_by_type['Type'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.tight_layout()

plt.title("Type vs. Installs", size=20)

plt.show(fig)
google_apps['Installs'].head()
plt.figure(figsize = (10,5))

sns.regplot(x="Installs", y="Rating", color = 'teal',data=sorted_by_install);

plt.title('Rating VS Installs',size = 20)
plt.figure(figsize=(40,10))

fig = sns.barplot(x=sorted_by_install['Genres'], y=sorted_by_install['Installs'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.tight_layout()

plt.title('Genres VS Installs',size = 40)

plt.show(fig)
fig = sns.catplot(x="Genres",y="Installs",data=google_apps, kind="boxen", height = 10 ,palette = "Paired")

fig.despine(left=True)

fig.set_xticklabels(rotation=90)

fig = fig.set_ylabels("Installs")

plt.tight_layout()

plt.title('Genres Vs. Installs (Boxenplate Version)',size = 20)
google_apps['new'] = pd.to_datetime(google_apps['Last Updated'])

google_apps['new'].describe()
google_apps['new'].max() 
google_apps['new'][0] -  google_apps['new'].max()
google_apps['lastupdate'] = (google_apps['new'] -  google_apps['new'].max()).dt.days

google_apps['lastupdate'].head()
plt.figure(figsize = (10,10))

sns.regplot(x="lastupdate", y="Installs", color = 'purple',data=google_apps );

plt.title('Installs  VS Last Update ( days ago )',size = 20)
plt.figure(figsize=(30,10))

fig = sns.barplot(x=sorted_by_install['Content Rating'], y=sorted_by_install['Installs'], palette="hls")

fig.set_xticklabels(fig.get_xticklabels(),rotation=90)

plt.tight_layout()

plt.title('Content Rating VS Installs',size = 40)

plt.show(fig)