# set up env
import pandas as pd
import zipfile
from sklearn.neighbors import NearestNeighbors
# This method can be used to access zip file without download locally. 
#But it doesn't work on kernel. Therefore below I added a a mthod to access uploaded here file.
#httml = 'http://adn.biol.umontreal.ca/~numericalecology/labo/Scotch/ScotchData.zip'

#import pandas as pd
#import requests, zipfile, io

#req = requests.get(httml)
#archive = zipfile.ZipFile(io.BytesIO(req.content))

#columns=['full_name', 'short_name', 'color_wyne', 'color_yellow', 'color_v.pale',
#       'color_pale', 'color_p.gold', 'color_gold', 'color_o.gold',
#       'color_f.gold', 'color_bronze', 'color_p.amber', 'color_amber',
#       'color_f.amber', 'color_red', 'color_sherry', 'NOSE_AROMA', 'NOSE_PEAT',
#       'NOSE_SWEET', 'NOSE_LIGHT', 'NOSE_FRESH', 'NOSE_DRY', 'NOSE_FRUIT',
#       'NOSE_GRASS', 'NOSE_SEA', 'NOSE_SHERRY', 'NOSE_SPICY', 'NOSE_RICH',
#       'BODY_soft', 'BODY_med', 'BODY_full', 'BODY_round', 'BODY_smooth',
#       'BODY_light', 'BODY_firm', 'BODY_oily', 'PAL_full', 'PAL_dry',
#       'PAL_sherry', 'PAL_big', 'PAL_light', 'PAL_smooth', 'PAL_clean',
#       'PAL_fruit', 'PAL_grass', 'PAL_smoke', 'PAL_sweet', 'PAL_spice',
#       'PAL_oil', 'PAL_salt', 'PAL_arome', 'FIN_full', 'FIN_dry', 'FIN_warm',
#       'FIN_big', 'FIN_light', 'FIN_smooth', 'FIN_clean', 'FIN_fruit',
#       'FIN_grass', 'FIN_smoke', 'FIN_sweet', 'FIN_spice', 'FIN_oil',
#       'FIN_salt', 'FIN_arome', 'FIN_ling', 'FIN_long', 'FIN_very',
#       'FIN_quick', '_AGE', '_DIST', '_SCORE', '_%', '_REGION', '_DISTRICT',
#       '_islay', '_midland', '_spey', '_east', '_west', '_north ', '_lowland',
#       '_campbell', '_islands']

#data = pd.read_excel(archive.open('ScotchData/Scotch data (Windows)/scotch.xlsx'), 
#                     skiprows=[0, 1],  header=None, names=columns)
# import data:
dir = '../input/'
file = 'scotch.xlsx'
data = pd.read_excel(dir + file, sheet_name='scotch.xls', header=0)
data.dropna(axis=0, inplace=True)
# List of the possible unique features.
dict = {'Feature': 'Atributes'}
for type in ['color', 'NOSE', 'BODY', 'PAL', 'FIN']:
    col_names = [col for col in data.columns if col.startswith(type)]
    dict[type] = [col.replace(type + '_', '').lower() for col in col_names]
for key, value in dict.items():
    print(key, ':\t', value)
# whisky db
print('_'*90, '\n\nAVAILABLE WHISKY IN DATA SET\n', '_'*90, '\n')
for letter in data.full_name.str[0].unique():
        print(letter, ':\t', [element for element in list(data.full_name) if element.startswith(letter)])
print('_'*90)
# Find common whisky
choice = 'Lagavulin'

data['id'] = data.index + 1
knn_columns = ['full_name', 'short_name', '_AGE', '_DIST', '_SCORE', '_%', '_REGION', '_DISTRICT',
       '_islay', '_midland', '_spey', '_east', '_west', '_north ', '_lowland',
       '_campbell', '_islands']
X_train = data.drop(knn_columns, axis=1)
y_train = data['id']
X_test = data.query('full_name == @choice').drop(knn_columns, axis=1)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train)
distance, indeces = nn.kneighbors(X_test, n_neighbors=6)

print('_'*70, '\nTaking into cosideration 5 distinct Whisky characteristics,',
              'Most recommended scotch (based on {0}) are:'.format(choice), '_'*70, sep='\n')
pd.concat([data.iloc[indeces[0], [0, -16, -12]], 
           pd.Series(distance[0], index=data.iloc[indeces[0], 0].index, name='Closest')], axis=1)

