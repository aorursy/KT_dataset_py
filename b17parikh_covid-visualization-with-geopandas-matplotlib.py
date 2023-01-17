import geopandas as gpd

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
map_data = gpd.read_file('/kaggle/input/india-states/Igismap/Indian_States.shp')
map_data['st_nm'] = map_data['st_nm'].str.replace('&', 'and')

map_data['st_nm'].replace('Arunanchal Pradesh', 'Arunachal Pradesh', inplace = True)

map_data['st_nm'].replace('Telangana', 'Telengana', inplace = True)

map_data['st_nm'].replace('NCT of Delhi', 'Delhi', inplace = True)

map_data['st_nm'].replace("Andaman and Nicobar Island",'Andaman and Nicobar Islands',inplace=True)
df=pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/state_level_latest.csv")

df.rename(columns={'State':"st_nm"},inplace=True)

df.fillna(0,inplace=True)



df
map_data.plot()
merged_data=pd.merge(map_data,df,how="left",on='st_nm')

merged_data


merged_data['coords'] = merged_data['geometry'].apply(lambda x: x.representative_point().coords[:])

merged_data['coords'] = [coords[0] for coords in merged_data['coords']]


merged_data.fillna(0, inplace = True)

merged_data['Confirmed']=merged_data['Confirmed'].astype(int)

bbox_props = dict(boxstyle="round", fc="w", ec="gray", alpha=0.8,lw=1)

print(bbox_props)

# create figure and axes for Matplotlib and set the title



fig, ax = plt.subplots(1, figsize=(20, 12))

for idx, row in merged_data.iterrows():

    ax.annotate(s=row['st_nm']+'\n'+'cases'+" "+str(int(row['Confirmed'])), xy=row['coords'], color='black',

                 horizontalalignment='center', bbox=bbox_props, fontsize=9)

ax.axis('off')

ax.set_title('Covid-19 Statewise Data - Confirmed Cases', fontdict = {'fontsize': '25', 'fontweight' : '3'})

# plot the figure

merged_data.plot(column = 'Confirmed', cmap='Reds', linewidth=0.8, ax=ax, edgecolor='0.8', legend = True)

plt.show()

plt.savefig("Indiancases.png")