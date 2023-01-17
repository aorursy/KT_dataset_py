# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
from collections import Counter
from collections import OrderedDict
import numpy as np 


resto = pd.read_csv('../input/restaurant_data_kaggles.csv', error_bad_lines=False)

#print(resto.head())
#print(resto.tail())

#print(resto.keys())
#renomme les attributs sans espaces et avec des noms plus pertinants
resto = resto.rename(columns={'restaurant_name ' : 'nom', 'review_number ':'review_number', ' food_type ':'type', ' ranking ':'rank',
       ' overallRating  ':'rating', ' wifi ':'wifi', ' livraison ':'livraison', ' average_price ':'prix', ' lat ':'lat', ' lng ':'long', ' district ':'arr'})
#print(resto.keys())

#remplace donnée categorielle manuscrites en chiffre pour facilité l'analyse
# yes = 1, no =0
resto.loc[resto.livraison == 'no','livraison'] = 0
resto.loc[resto.livraison == 'yes','livraison'] = 1

resto.loc[resto.wifi == 'no','wifi'] = 0
resto.loc[resto.wifi == 'yes','wifi'] = 1

resto.loc[resto.lat == 'no information','lat'] = 0
resto.loc[resto.long == 'no information','long'] = 0

resto.head(5)

import folium


a1 = resto.lat.astype(float)
b1 = resto.long.astype(float)
c1 = resto.type
a=list(a1)
b=list(b1)
c=list(c1)

#print(resto.nom)
#print(resto.type)

m=folium.Map(location=[45.764043,4.835659],zoom_start = 15)


i=0
while i<250:
    folium.Marker([a[0+i],b[0+i]], c[0+i]).add_to(m)
    i = i+1


display(m)
k=0
arr=[]
while k<len(resto['arr']):
    arr+=Counter(resto['arr'][k].split()).most_common()
    k+=1
subclass_arr = [j for j in arr if j[0] != 'Lyon' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Villeurbanne' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Saint-Priest' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'en' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Vaulx' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Velin' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Bron' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Chaponnay' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Decines-Charpieu' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Charbonni\xc3\xa8res-les-Bains' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Limonest' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Chassieu' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Ecully' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Francheville' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'LYON' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'lyon' ]
subclass_arr = [j for j in subclass_arr if j[0] != 'Charbonnières-les-Bains  ' ]

val_arr=  OrderedDict(sorted(Counter(subclass_arr).most_common(15), key=lambda t: t[1],reverse=True)).values()
keys_arr=  OrderedDict(sorted(Counter(subclass_arr).most_common(15), key=lambda t: t[1],reverse=True)).keys()
keys_list = [x[0] for x in keys_arr]



plt.figure(figsize=(15,10))
sns.barplot(y=list(keys_list), x= list(val_arr), orient='h')
plt.title('Top 15 district to eat in Lyon', fontsize=16)
plt.xlabel('Number Of restaurant', fontsize=16)
plt.ylabel('district', fontsize=16)
plt.show()
plt.close
#remplace le character "|" par un vide pour pouvoir analyser mot par mot le jeu de données
data = resto['type'].str.replace("|","")
i=0
#fonction split permet de recuperer pour chaque ligne les mot separé (ex: (un chat) deviens (un:1),(chat:1))
#on stock ces mot dans un tableau que l'on nomme u 
u=[]
while i<len(data):
    u+=Counter(data[i].split()).most_common()
    i+=1
    
#supprime les mot inutiles ou repetitif de la list u pour pouvoir realiser de meilleures analyses 
subclass = [j for j in u if j[0] != 'no' ]

subclass = [j for j in subclass if j[0] != 'de' ]

subclass = [j for j in subclass if j[0] != 'à' ]

subclass = [j for j in subclass if j[0] != 'bienvenus' ]

subclass = [j for j in subclass if j[0] != 'restauration' ]
subclass = [j for j in subclass if j[0] != 'Restauration' ]

subclass = [j for j in subclass if j[0] != '&' ]

subclass = [j for j in subclass if j[0] != 'food' ]

subclass = [j for j in subclass if j[0] != 'cuisine' ]

subclass = [j for j in subclass if j[0] != 'rue' ]

subclass = [j for j in subclass if j[0] != 'choix' ]

subclass = [j for j in subclass if j[0] != 'plats' ]

subclass = [j for j in subclass if j[0] != 'gluten' ]

subclass = [j for j in subclass if j[0] != 'sans' ]

subclass = [j for j in subclass if j[0] != 'Saine' ]

subclass = [j for j in subclass if j[0] != 'information' ]
#Fonction Counter permet de compter le nombre d'un fois qu'un element apparais dans notre tableau 

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
values=list(Counter(subclass).values())
labels=list(Counter(subclass).keys())
label_list = [x[0] for x in labels]
# figure
fig = {
  "data": [
    {
      "values": values,
      "labels": label_list,
      "domain": {"x": [0, .5]},
      "hoverinfo":"label",
      "hole": .2,
      "type": "pie"
    },],
  "layout": {        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Food Categories",
                "x": 0.15,
                "y": 1.15
            },
        ]
    }
}
iplot(fig)



from collections import Counter
from collections import OrderedDict
import numpy as np 

# Make fake dataset
height = OrderedDict(sorted(Counter(subclass).most_common(15), key=lambda t: t[1],reverse=True)).values()
bars = OrderedDict(sorted(Counter(subclass).most_common(15), key=lambda t: t[1],reverse=True)).keys()
y_pos = np.arange(len(bars))


# Create horizontal bars
my_colors = ['red','orange','#ffe119','#ff6666','#42d4f4','blue','#f032e6','#e6beff','#911eb4','#3cb44b', '#ffcc99', '#99ff99', '#66b3ff']
plt.barh(y_pos, list(height), color=my_colors)

# Create names on the y-axis
plt.yticks(y_pos, bars)

# Show graphic
plt.title('Top 15 popular food categories in Lyon')
plt.show()
plt.close
resto.groupby('prix').size()

#analysis group by good rating
datrating = resto.loc[resto['rating'] != 'no information']
datrating = datrating.loc[datrating['rating'] > str(4.5)]

#analysis group by low rating
datrating_low = resto.loc[resto['rating'] != 'no information']
datrating_low = datrating_low.loc[datrating_low['rating'] < str(2.0)]



sns.catplot(y="rating", hue="arr", kind="count",
            palette="pastel", edgecolor=".6",
            data=datrating);
plt.title('Good Rating count per district')
plt.show()
plt.close

sns.catplot(y="rating", hue="arr", kind="count",
            palette="pastel", edgecolor=".6",
            data=datrating_low);
plt.title('Bad Rating count per district')
plt.show()
plt.close




k1=0
arr1=[]
while k1<len(resto['arr']):
    arr1+=Counter(resto['arr'][k1].split()).most_common()
    k1+=1
subclass_arr1 = [j for j in arr if j[0] != 'Lyon' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Villeurbanne' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Saint-Priest' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'en' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Vaulx' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Velin' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Bron' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Chaponnay' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Decines-Charpieu' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Charbonni\xc3\xa8res-les-Bains' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Limonest' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Chassieu' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Ecully' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Francheville' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'LYON' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'lyon' ]
subclass_arr1 = [j for j in subclass_arr if j[0] != 'Charbonnières-les-Bains  ' ]

val_arr1=  OrderedDict(sorted(Counter(subclass_arr1).most_common(10), key=lambda t: t[1],reverse=True)).values()
keys_arr1=  OrderedDict(sorted(Counter(subclass_arr1).most_common(10), key=lambda t: t[1],reverse=True)).keys()
keys_list1 = [x[0] for x in keys_arr1]


plt.figure(figsize=(20,20))
nobu =datrating.loc[datrating["rating"] != '4.5']

# best restaurant & top area

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,3))

# What are the best restaurant in Lyon?

sns.countplot(y='rating',hue='nom', data = nobu.sort_values(by='rating',ascending=False)[:10], palette="Set3", ax=axis1)

# What are the top area to eat in Lyon?
sns.barplot(y=list(keys_list), x= list(val_arr), orient='h', palette="Set3", ax=axis2)

plt.title('Most popular area in Lyon', fontsize=16)
plt.xlabel('count', fontsize=16)
plt.ylabel('district', fontsize=16 )
plt.show()
plt.close



# bad restaurant & bad area
low_rating=datrating_low.loc[datrating_low["rating"] != '1.5']

val_arr_bad=  OrderedDict(sorted(Counter(subclass_arr1).most_common(10), key=lambda t: t[1],reverse=False)).values()
keys_arr_bad=  OrderedDict(sorted(Counter(subclass_arr1).most_common(10), key=lambda t: t[1],reverse=False)).keys()
keys_list_bad = [x[0] for x in keys_arr_bad]


# What are the best restaurant in Lyon?

sns.countplot(y='rating',hue='nom', data = low_rating.sort_values(by='rating',ascending=False)[:10], palette="Set3")

plt.title('Less popular restaurant in Lyon', fontsize=16)
plt.xlabel('count', fontsize=16)
plt.show()
plt.close


