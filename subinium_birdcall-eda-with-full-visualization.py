import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import folium

from folium import plugins

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



import missingno as msno



import os
# version check 



print(f'numpy : {np.__version__}')

print(f'pandas : {pd.__version__}')

print(f'matplotlib : {mpl.__version__}')

print(f'folium : {folium.__version__}')
# configuration for notebook



pd.options.display.max_columns = 40  
PATH = '/kaggle/input/birdsong-recognition/'

train = pd.read_csv(f'{PATH}/train.csv')

test = pd.read_csv(f'{PATH}/test.csv')
print(train.shape)

train.head(3)
msno.matrix(train)
fig, ax = plt.subplots(1, 1, figsize=(20, 5), dpi=200)

ebird_code = train['ebird_code'].value_counts()

ax.bar(ebird_code.index, ebird_code, color='#6EB5FF')



ax.text(134, 105, 

        '100 samples for 134 birsds',)





ax.text(235, 120,

        f"mean : {ebird_code.mean():.2f} std: {ebird_code.std():.2f}",

        color="black", fontsize=11, fontweight='bold',

         bbox=dict(boxstyle='round', pad=0.3, color='lightgray')

)



ax.set_ylim(0, 130)

ax.set_xticks([])

ax.margins(0.01, 0.01)



ax.set_title('Distribution : bird type', 

             fontsize=15, fontweight='bold', fontfamily='serif',

             x=0.075, y=1.04,)







plt.tight_layout()

plt.show()
train['latitude'] = train['latitude'].apply(lambda x : float(x) if '.' in x else None)

train['longitude'] = train['longitude'].apply(lambda x : float(x) if '.' in x else None)



try : 

    train.drop(['license', 'file_type'], inplace=True)

except :

    pass
m = folium.Map()



train_for_map = train[['latitude', 'longitude', 'species']].dropna()



# Marker Cluster

plugins.MarkerCluster(train_for_map[['latitude', 'longitude']].values,

                      list(train_for_map['species'].apply(str).values)

).add_to(m)



# Mouse Check

formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"

plugins.MousePosition(

    position='topright',

    separator=' | ',

    empty_string='NaN',

    lng_first=True,

    num_digits=20,

    prefix='Coordinates:',

    lat_formatter=formatter,

    lng_formatter=formatter,

).add_to(m)



# minimap

minimap = plugins.MiniMap()

m.add_child(minimap)





m


