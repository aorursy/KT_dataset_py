# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
forest_data = pd.read_csv('/kaggle/input/learn-together/train.csv')
# Just making things easier here:

features = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

soil = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

wilderness_area = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4']
y = forest_data.Cover_Type

X1 = forest_data[features] # Since this set is so dominate in models I've seen so far, including my own, this is here for the sake of it.

X2 = forest_data[features + wilderness_area + soil]
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y, test_size=0.4, random_state=1)

train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y, test_size=0.4, random_state=1)
rfc_model1 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model1.fit(train_X1, train_y1)

rfc_val_predictions1 = rfc_model1.predict(val_X1)

rfc_val_mae1 = mean_absolute_error(rfc_val_predictions1, val_y1)



rfc_model2 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model2.fit(train_X2, train_y2)

rfc_val_predictions2 = rfc_model2.predict(val_X2)

rfc_val_mae2 = mean_absolute_error(rfc_val_predictions2, val_y2)



print("Validation MAE for features-only: {:,.2f}".format(rfc_val_mae1))

print("Validation MAE for all variables: {:,.2f}".format(rfc_val_mae2))
# Looking at feature importance:

import seaborn as sns

import matplotlib.pyplot as plt



def feature_importances(clf, X, y, figsize=(18, 6)):

    clf = clf.fit(X, y)

    

    importances = pd.DataFrame({'Features': X.columns, 

                                'Importances': clf.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=figsize)

    sns.barplot(x='Features', y='Importances', data=importances)

    plt.xticks(rotation='vertical')

    plt.show()

    return importances
importances = feature_importances(rfc_model1, X1, y)
importances = feature_importances(rfc_model2, X2, y)
forest_data['Soil_Type']=''

forest_data['Wilderness_Area'] = ''



for col_name in forest_data[soil].columns:

    forest_data.loc[forest_data[col_name]==1,'Soil_Type']= forest_data['Soil_Type'] + col_name



for col_name in forest_data[wilderness_area].columns:

    forest_data.loc[forest_data[col_name]==1,'Wilderness_Area']= forest_data['Wilderness_Area'] + col_name
tmp = pd.DataFrame(forest_data['Soil_Type'])

tmp['Taxonomy_Groups'] = forest_data['Soil_Type']

tmp['Order_Groups'] = forest_data['Soil_Type']

tmp['USFS_ELU_Codes'] = forest_data['Soil_Type']



tmp.head()
tmp['Taxonomy_Groups'] = tmp['Taxonomy_Groups'].replace({'Soil_Type1' : 'Mollisols_Ustolls_Haplustolls',

                                                        'Soil_Type2' : 'Alfisols_Mollisols',

                                                        'Soil_Type3' : 'Mollisols_Borolls_Haploborolls',

                                                        'Soil_Type4' : 'Mollisols_Ustolls_Haplustolls',

                                                        'Soil_Type5' : 'Alfisols_Ustalfs_Haplustalfs',

                                                        'Soil_Type6' : 'Alfisols_Ustalfs_Haplustalfs',

                                                        'Soil_Type7' : 'Mollisols_Cryolls_Agricryolls',

                                                        'Soil_Type8' : 'Alfisols_Mollisols',

                                                        'Soil_Type9' : 'Alfisols_Cryalfs_Haplocryalfs',

                                                        'Soil_Type10' : 'Inceptisols_Cryepts',

                                                        'Soil_Type11' : 'Inceptisols_Cryepts',

                                                        'Soil_Type12' : 'Entisols_Orthents_Cryorthents',

                                                        'Soil_Type13' : 'Inceptisols_Cryepts',

                                                        'Soil_Type14' : 'Mollisols',

                                                        'Soil_Type15' : 'Unspecified',

                                                        'Soil_Type16' : 'Mollisols',

                                                        'Soil_Type17' : 'Mollisols',

                                                        'Soil_Type18' : 'Mollisols',

                                                        'Soil_Type19' : 'Histosols_Mollisols',

                                                        'Soil_Type20' : 'Inceptisols_Mollisols',

                                                        'Soil_Type21' : 'Inceptisols_Mollisols',

                                                        'Soil_Type22' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type23' : 'Inceptisols_Mollisols',

                                                        'Soil_Type24' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type25' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type26' : 'Alfisols_Inceptisols',

                                                        'Soil_Type27' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type28' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type29' : 'Entisols_Inceptisols',

                                                        'Soil_Type30' : 'Entisols_Inceptisols',

                                                        'Soil_Type31' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type32' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type33' : 'Inceptisols_Cryepts_Dystrocryepts',

                                                        'Soil_Type34' : 'Entisols_Orthents_Cryorthents',

                                                        'Soil_Type35' : 'Inceptisols',

                                                        'Soil_Type36' : 'Inceptisols',

                                                        'Soil_Type37' : 'Entisols_Inceptisols',

                                                        'Soil_Type38' : 'Inceptisols_Mollisols',

                                                        'Soil_Type39' : 'Entisols_Inceptisols',

                                                        'Soil_Type40' : 'Entisols_Inceptisols'}

                                                       )



tmp['Taxonomy_Groups'].describe()
tmp['Order_Groups'] = tmp['Order_Groups'].replace({'Soil_Type1' : 'Mollisols',

                                                   'Soil_Type2' : 'Alfisols_Mollisols',

                                                   'Soil_Type3' : 'Mollisols',

                                                   'Soil_Type4' : 'Mollisols',

                                                   'Soil_Type5' : 'Alfisols',

                                                   'Soil_Type6' : 'Alfisols',

                                                   'Soil_Type7' : 'Mollisols',

                                                   'Soil_Type8' : 'Alfisols_Mollisols',

                                                   'Soil_Type9' : 'Alfisols',

                                                   'Soil_Type10' : 'Inceptisols',

                                                   'Soil_Type11' : 'Inceptisols',

                                                   'Soil_Type12' : 'Entisols',

                                                   'Soil_Type13' : 'Inceptisols',

                                                   'Soil_Type14' : 'Mollisols',

                                                   'Soil_Type15' : 'Unspecified',

                                                   'Soil_Type16' : 'Mollisols',

                                                   'Soil_Type17' : 'Mollisols',

                                                   'Soil_Type18' : 'Mollisols',

                                                   'Soil_Type19' : 'Histosols_Mollisols',

                                                   'Soil_Type20' : 'Inceptisols_Mollisols',

                                                   'Soil_Type21' : 'Inceptisols_Mollisols',

                                                   'Soil_Type22' : 'Inceptisols',

                                                   'Soil_Type23' : 'Inceptisols_Mollisols',

                                                   'Soil_Type24' : 'Inseptisols',

                                                   'Soil_Type25' : 'Inceptisols',

                                                   'Soil_Type26' : 'Alfisols_Inceptisols',

                                                   'Soil_Type27' : 'Inceptisols',

                                                   'Soil_Type28' : 'Inceptisols',

                                                   'Soil_Type29' : 'Entisols_Inceptisols',

                                                   'Soil_Type30' : 'Entisols_Inceptisols',

                                                   'Soil_Type31' : 'Inseptisols',

                                                   'Soil_Type32' : 'Inseptisols',

                                                   'Soil_Type33' : 'Inseptisols',

                                                   'Soil_Type34' : 'Entisols',

                                                   'Soil_Type35' : 'Inceptisols',

                                                   'Soil_Type36' : 'Inceptisols',

                                                   'Soil_Type37' : 'Entisols_Inceptisols',

                                                   'Soil_Type38' : 'Inceptisols_Mollisols',

                                                   'Soil_Type39' : 'Entisols_Inceptisols',

                                                   'Soil_Type40' : 'Entisols_Inceptisols'}

                                                 )

tmp['Order_Groups'].describe()
tmp['USFS_ELU_Codes'] = tmp['USFS_ELU_Codes'].replace({'Soil_Type1' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type2' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type3' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type4' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type5' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type6' : 'lower_montane_igneous_and_metamorphic',

                                                       'Soil_Type7' : 'montane_dry_mixed_sedimentary',

                                                       'Soil_Type8' : 'montane_dry_mixed_sedimentary',

                                                       'Soil_Type9' : 'montane_glacial',

                                                       'Soil_Type10' : 'montane_igneous_and_metamorphic',

                                                       'Soil_Type11' : 'montane_igneous_and_metamorphic',

                                                       'Soil_Type12' : 'montane_igneous_and_metamorphic',

                                                       'Soil_Type13' : 'montane_igneous_and_metamorphic',

                                                       'Soil_Type14' : 'montane_dry_and_montane_alluvium',

                                                       'Soil_Type15' : 'montane_dry_and_montane_alluvium',

                                                       'Soil_Type16' : 'montane_and_subalpine_alluvium',

                                                       'Soil_Type17' : 'montane_and_subalpine_alluvium',

                                                       'Soil_Type18' : 'montane_and_subalpine_igneous_and_metamorphic',

                                                       'Soil_Type19' : 'subalpine_alluvium',

                                                       'Soil_Type20' : 'subalpine_alluvium',

                                                       'Soil_Type21' : 'subalpine_alluvium',

                                                       'Soil_Type22' : 'subalpine_glacial',

                                                       'Soil_Type23' : 'subalpine_glacial',

                                                       'Soil_Type24' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type25' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type26' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type27' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type28' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type29' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type30' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type31' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type32' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type33' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type34' : 'subalpine_igneous_and_metamorphic',

                                                       'Soil_Type35' : 'alpine_igneous_and_metamorphic',

                                                       'Soil_Type36' : 'alpine_igneous_and_metamorphic',

                                                       'Soil_Type37' : 'alpine_igneous_and_metamorphic',

                                                       'Soil_Type38' : 'alpine_igneous_and_metamorphic',

                                                       'Soil_Type39' : 'alpine_igneous_and_metamorphic',

                                                       'Soil_Type40' : 'alpine_igneous_and_metamorphic'}

                                                 )

tmp['USFS_ELU_Codes'].describe()
# Just an interim step:

tmp = pd.get_dummies(tmp, columns=['Taxonomy_Groups', 'Order_Groups', 'USFS_ELU_Codes'])



taxonomy = ['Taxonomy_Groups_Alfisols_Cryalfs_Haplocryalfs',

       'Taxonomy_Groups_Alfisols_Inceptisols',

       'Taxonomy_Groups_Alfisols_Mollisols',

       'Taxonomy_Groups_Alfisols_Ustalfs_Haplustalfs',

       'Taxonomy_Groups_Entisols_Inceptisols',

       'Taxonomy_Groups_Entisols_Orthents_Cryorthents',

       'Taxonomy_Groups_Histosols_Mollisols', 'Taxonomy_Groups_Inceptisols',

       'Taxonomy_Groups_Inceptisols_Cryepts',

       'Taxonomy_Groups_Inceptisols_Cryepts_Dystrocryepts',

       'Taxonomy_Groups_Inceptisols_Mollisols', 'Taxonomy_Groups_Mollisols',

       'Taxonomy_Groups_Mollisols_Borolls_Haploborolls',

       'Taxonomy_Groups_Mollisols_Ustolls_Haplustolls']

order = ['Order_Groups_Alfisols', 'Order_Groups_Alfisols_Inceptisols',

       'Order_Groups_Alfisols_Mollisols', 'Order_Groups_Entisols',

       'Order_Groups_Entisols_Inceptisols', 'Order_Groups_Histosols_Mollisols',

       'Order_Groups_Inceptisols', 'Order_Groups_Inceptisols_Mollisols',

       'Order_Groups_Inseptisols', 'Order_Groups_Mollisols']

usfs_codes = ['USFS_ELU_Codes_alpine_igneous_and_metamorphic',

       'USFS_ELU_Codes_lower_montane_igneous_and_metamorphic',

       'USFS_ELU_Codes_montane_and_subalpine_alluvium',

       'USFS_ELU_Codes_montane_and_subalpine_igneous_and_metamorphic',

       'USFS_ELU_Codes_montane_dry_and_montane_alluvium',

       'USFS_ELU_Codes_montane_dry_mixed_sedimentary',

       'USFS_ELU_Codes_montane_glacial',

       'USFS_ELU_Codes_montane_igneous_and_metamorphic',

       'USFS_ELU_Codes_subalpine_alluvium', 'USFS_ELU_Codes_subalpine_glacial',

       'USFS_ELU_Codes_subalpine_igneous_and_metamorphic']
# Going to use my single wilderness column the rest of the time to stratify:

X6 = forest_data[features + soil] # my benchmark stratified

X9 = pd.concat([forest_data[features], tmp[taxonomy]], axis=1, sort=False) # features + taxonomy

X11 = pd.concat([forest_data[features], tmp[order]], axis=1, sort=False) # features + order

X12 = pd.concat([forest_data[features], tmp[usfs_codes]], axis=1, sort=False) # features + codes
train_X6, val_X6, train_y6, val_y6 = train_test_split(X6, y, test_size=0.4, stratify=forest_data['Wilderness_Area'], random_state=1)

train_X9, val_X9, train_y9, val_y9 = train_test_split(X9, y, test_size=0.4, stratify=forest_data['Wilderness_Area'], random_state=1)

train_X11, val_X11, train_y11, val_y11 = train_test_split(X11, y, test_size=0.4, stratify=forest_data['Wilderness_Area'], random_state=1)

train_X12, val_X12, train_y12, val_y12 = train_test_split(X12, y, test_size=0.4, stratify=forest_data['Wilderness_Area'], random_state=1)
rfc_model6 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model6.fit(train_X6, train_y6)

rfc_val_predictions6 = rfc_model6.predict(val_X6)

rfc_val_mae6 = mean_absolute_error(rfc_val_predictions6, val_y6)



rfc_model9 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model9.fit(train_X9, train_y9)

rfc_val_predictions9 = rfc_model9.predict(val_X9)

rfc_val_mae9 = mean_absolute_error(rfc_val_predictions9, val_y9)



rfc_model11 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model11.fit(train_X11, train_y11)

rfc_val_predictions11 = rfc_model11.predict(val_X11)

rfc_val_mae11 = mean_absolute_error(rfc_val_predictions11, val_y11)



rfc_model12 = RandomForestClassifier(n_estimators=100, random_state=1)

rfc_model12.fit(train_X12, train_y12)

rfc_val_predictions12 = rfc_model12.predict(val_X12)

rfc_val_mae12 = mean_absolute_error(rfc_val_predictions12, val_y12)



print("Validation MAE for features-only       : {:,.2f}".format(rfc_val_mae1))

print("Validation MAE for all variables       : {:,.2f}".format(rfc_val_mae2))

print("Validation MAE for stratified benchmark: {:,.2f}".format(rfc_val_mae6))

print("Validation MAE for features + taxonomy : {:,.2f}".format(rfc_val_mae9))

print("Validation MAE for features + order    : {:,.2f}".format(rfc_val_mae11))

print("Validation MAE for features + codes    : {:,.2f}".format(rfc_val_mae12))
importances = feature_importances(rfc_model1, X1, y)
importances = feature_importances(rfc_model2, X2, y)
importances = feature_importances(rfc_model6, X6, y)   
importances = feature_importances(rfc_model9, X9, y)   
importances = feature_importances(rfc_model11, X11, y) 
importances = feature_importances(rfc_model12, X12, y) 