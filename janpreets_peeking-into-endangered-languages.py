# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import re

from wordcloud import WordCloud

from nltk.corpus import stopwords

from mpl_toolkits.basemap import Basemap

import plotly

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()





%matplotlib inline

data = pd.read_csv("../input/data.csv") #loading data
data.shape
data.columns
data.head()
features = ["ID","Name in English","Countries","Degree of endangerment","Alternate names","Name in the language","Number of speakers","Sources","Latitude","Longitude","Description of the location"]
data = data[features] #pandas command for selecting features
data.isnull().sum()
data = data.drop(["Name in the language"],axis = 1)

data = data.drop(["Alternate names"],axis = 1)
data[data.Countries.isnull() == True] #cheking for that one language which has missing country
data.Countries = data.Countries.fillna("nocountry") #assigning the missing country as "nocountry"
text = ''

np.array(data.Countries.values)[1].split(",")[0]

for i in np.arange(len(np.array(data.Countries.values))):

    val = np.array(data.Countries.values)[i].split(",")

    for j in np.arange(len(val)):

        text = " ".join([text,val[j]])

        

text = text.strip()
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=312)

plt.imshow(wordcloud)

plt.title("Wordcloud for countries ")

plt.axis("off")

plt.show()
plt.xticks(rotation = -45)

sns.countplot(x="Degree of endangerment", data=data, palette="Greens_d")
language_spread = pd.DataFrame(data["Name in English"].values, columns=["Name in English"])

language_spread["num_countries"] = 0

language_spread["ID"] = data.ID

language_spread["Countries"] = data.Countries
language_spread.shape
language_spread.num_countries = data.Countries.apply(lambda x: len(re.findall(r",",x)))

language_spread.num_countries = language_spread.num_countries + 1

language_spread = language_spread.sort_values(by = "num_countries",ascending=False)
language_spread["Name in English"].head()
sns.distplot(language_spread.num_countries)
text = ''



for i in np.arange(len(np.array(data.Countries.values))):

    val = np.array(language_spread.Countries.values)[i].split(",")

    for j in np.arange(len(val)):

        text = " ".join([text,val[j]])

text = text.strip()
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=312)

plt.imshow(wordcloud)

plt.title("Wordcloud for top 40 languages ")

plt.axis("off")

plt.show()
data.Latitude = data.Latitude.fillna(0)

data.Longitude = data.Longitude.fillna(0)
plt.figure(figsize=(12,6))



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

en_lon = data[data["Degree of endangerment"] == "Vulnerable"]["Longitude"].astype(float)

en_lat = data[data["Degree of endangerment"] == "Vulnerable"]["Latitude"].astype(float)

x, y = m(list(en_lon), list(en_lat))

m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")



plt.title('Vulnerable languages')

plt.show()



plt.figure(figsize=(12,6))



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

en_lon = data[data["Degree of endangerment"] == "Definitely endangered"]["Longitude"].astype(float)

en_lat = data[data["Degree of endangerment"] == "Definitely endangered"]["Latitude"].astype(float)

x, y = m(list(en_lon), list(en_lat))

m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")



plt.title('Vulnerable languages')

plt.show()



plt.figure(figsize=(12,6))



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

en_lon = data[data["Degree of endangerment"] == "Severely endangered"]["Longitude"].astype(float)

en_lat = data[data["Degree of endangerment"] == "Severely endangered"]["Latitude"].astype(float)

x, y = m(list(en_lon), list(en_lat))

m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")



plt.title('Vulnerable languages')

plt.show()



plt.figure(figsize=(12,6))



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

en_lon = data[data["Degree of endangerment"] == "Critically endangered"]["Longitude"].astype(float)

en_lat = data[data["Degree of endangerment"] == "Critically endangered"]["Latitude"].astype(float)

x, y = m(list(en_lon), list(en_lat))

m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")



plt.title('Vulnerable languages')

plt.show()



plt.figure(figsize=(12,6))



m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')

m.drawcoastlines()

m.drawcountries()

en_lon = data[data["Degree of endangerment"] == "Extinct"]["Longitude"].astype(float)

en_lat = data[data["Degree of endangerment"] == "Extinct"]["Latitude"].astype(float)

x, y = m(list(en_lon), list(en_lat))

m.plot(x, y, 'go', markersize = 6, alpha = 0.7, color = "blue")



plt.title('Vulnerable languages')

plt.show()


