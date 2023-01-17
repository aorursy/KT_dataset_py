import numpy as np

import pandas as pd

import seaborn as sns

import sklearn.cluster as clust

import matplotlib.pyplot as plt

import matplotlib

import folium as fl
df = pd.read_csv('/kaggle/input/bournemouth-venues/bournemouth_venues.csv')
df['Venue Category'].value_counts()
new_cat = df['Venue Category'].copy()

new_cat = new_cat.str.replace(r".*(Shop|Store).*","Shop",case=False)

new_cat = new_cat.str.replace(r".*Restaurant.*","Rest",case=False)

new_cat = new_cat.str.replace(r".*(Brewery|Bar|Café|Pub).*","Drink",case=False)

new_cat = new_cat.str.replace(r".*(Sandwich|Diner|Burger|Pizza|Noodle).*","Eat",case=False)

new_cat = new_cat.str.replace(r".*(Art|Theater|Aquarium|Nightclub|Gym).*","Art",case=False)

new_cat = new_cat.str.replace(r".*(Bus|Platform|Train).*","Tran",case=False)

new_cat = new_cat.str.replace(r".*(Other|Lookout|Garden|Plaza|Multiplex|Park|Beach).*","Place",case=False)

df['Venue Category'] = new_cat

plt.figure(figsize=(10,6))

sns.countplot(new_cat).set_title("Counting categories")
model_clast = clust.KMeans(n_clusters=6,random_state=15).fit(df[['Venue Latitude','Venue Longitude']])

df['Venue LatLong Claster'] = model_clast.predict(df[['Venue Latitude','Venue Longitude']])

df
df_pt = df.groupby(by=['Venue Category','Venue LatLong Claster']).count().reset_index().pivot('Venue LatLong Claster','Venue Category','Venue Name').fillna(0)

plt.figure(figsize=(10,6))

sns.heatmap(df_pt  ,annot=True,cmap='RdPu')
map_center = df[['Venue Latitude','Venue Longitude']].mean().as_matrix()

map = fl.Map(map_center,zoom_start=15)



for n in range(df_pt.shape[0]):

    [lt,ln] = model_clast.cluster_centers_[n]

    tit = ''

    for i in range(df_pt.shape[1]):

        v = df_pt.iloc[n,i]

        tit += '<br>' + df_pt.columns[i] + ": " + str(int(v))

        if v==0:

            tit += ' (too little!!!)'

    fl.Marker(location=[lt,ln],icon=fl.Icon(icon='info-sign'),tooltip='Claster #' + str(n)+'<br>----' + tit).add_to(map)



cat_list = list(df['Venue Category'].unique())

colormap=matplotlib.cm.rainbow(np.linspace(0, 1, len(cat_list)))



for n in range(df.shape[0]):

    cat = df['Venue Category'][n]

    t = df['Venue Name'][n] + " (" + cat + ")"

    c = matplotlib.colors.to_hex(colormap[cat_list.index(cat)])

    fl.CircleMarker(location=df[['Venue Latitude','Venue Longitude']].as_matrix()[n,:],radius = 5,tooltip=t,color=c,fill=True).add_to(map)

    

map