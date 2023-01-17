# importing modules and data 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 70)

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import KFold

import warnings

warnings.simplefilter(action='ignore')



train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')



cover_type = {1:'Spruce/Fir', 2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

train['Cover_type_description'] = train['Cover_Type'].map(cover_type)



# I put together train and test to work simultaneously on both of them



combined_data = [train, test]



def distance(a,b):

    return np.sqrt(np.power(a,2)+np.power(b,2))



extremely_stony = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1]

stony = [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

rubbly = [0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

other = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]



for data in combined_data:



    data['mean_Hillshade'] = (data['Hillshade_9am']+ data['Hillshade_Noon']+data['Hillshade_3pm'])/3

    data['Distance_to_hidrology'] = distance(data['Horizontal_Distance_To_Hydrology'],data['Vertical_Distance_To_Hydrology'])

    data['Distance_hydrology_roads'] =distance(data['Vertical_Distance_To_Hydrology'], data['Horizontal_Distance_To_Roadways'])

    data['Distance_hydrology_fire'] = distance(data['Vertical_Distance_To_Hydrology'], data['Horizontal_Distance_To_Fire_Points'])

    

    # there are 40 differents type of soil in Roosvelt National Forest but there are some common features between them

    # we can try to group soils according to some common features like stony 

    

    data['extremely_stony_level'] = data[[col for col in data.columns if col.startswith("Soil")]]@extremely_stony

    data['stony'] = data[[col for col in data.columns if col.startswith("Soil")]]@stony

    data['rubbly'] = data[[col for col in data.columns if col.startswith("Soil")]]@rubbly

    data['other'] = data[[col for col in data.columns if col.startswith("Soil")]]@other

   

    data['Hillshade_noon_3pm'] = data['Hillshade_Noon']- data['Hillshade_3pm']

    data['Hillshade_3pm_9am'] = data['Hillshade_3pm']- data['Hillshade_9am']

    data['Hillshade_9am_noon'] = data['Hillshade_9am']- data['Hillshade_Noon']

    

    data['Up_the_water'] = data['Vertical_Distance_To_Hydrology'] > 0

    data['Horizontal_plus_vertical_distance_to_hydrology'] = data['Horizontal_Distance_To_Hydrology'] + data['Vertical_Distance_To_Hydrology']

    data['Total_horizontal_distance'] = data['Horizontal_Distance_To_Hydrology']+ data['Horizontal_Distance_To_Roadways']+ data['Horizontal_Distance_To_Fire_Points']

    data['Elevation_of_hydrology'] = data['Elevation']+ data['Vertical_Distance_To_Hydrology']

    data['Elevation_of_hydrology2'] = data['Elevation']- data['Vertical_Distance_To_Hydrology']

    data['Distance_to_firepoints plus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']+ data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads plus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Hydrology']

    data['Distance_to_firepoints minus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']- data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads minus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Hydrology']

    data['Elevation_plus_slope'] = data['Elevation']+ data['Slope']

    data['Elevation_plus_aspect'] =data['Elevation']+ data['Aspect']

    data['Elevation_Aspect_Slope'] = data['Elevation']+ data['Aspect']+ data['Slope']

    data['Slope_plus_aspect'] = data['Slope']+ data['Aspect']

    data['Hillshade9_plus_hillshadenoon_plus_hillshade3'] = data['Hillshade_9am'] + data['Hillshade_Noon']+ data['Hillshade_3pm']

    data['Aspen'] = data['Soil_Type11']+data['Soil_Type13']+data['Soil_Type18']+data['Soil_Type19']+data['Soil_Type26']+data['Soil_Type30']

    data['Cottonwood/Willow'] = data['Soil_Type1']+data['Soil_Type3']+data['Soil_Type14']+data['Soil_Type17']

    data['Douglas-fir'] = data['Soil_Type5']+data['Soil_Type10']+data['Soil_Type16']

    data['Krummholz'] = data['Soil_Type35']+data['Soil_Type36']+data['Soil_Type37']+data['Soil_Type38']+data['Soil_Type39']+data['Soil_Type40']

    data['Lodgepole Pine'] = data['Soil_Type8']+data['Soil_Type9']+data['Soil_Type12']+data['Soil_Type20']+data['Soil_Type25']+data['Soil_Type28']+data['Soil_Type29']+data['Soil_Type32']+data['Soil_Type33']+data['Soil_Type34']

    data['Ponderosa Pine'] = data['Soil_Type2']+data['Soil_Type4']+data['Soil_Type6']

    data['Spruce/Fir'] = data['Soil_Type21']+data['Soil_Type22']+data['Soil_Type23']+data['Soil_Type24']+data['Soil_Type27']+data['Soil_Type31']

    data['Spruce/Fir2'] = data['Soil_Type19']+data['Soil_Type21']+data['Soil_Type22']+data['Soil_Type23']+data['Soil_Type24']+data['Soil_Type27']+data['Soil_Type31']+data['Soil_Type35']+data['Soil_Type38']+data['Soil_Type39']+data['Soil_Type40']

    data['Lodgepole Pine2'] = data['Soil_Type2']+data['Soil_Type3']+data['Soil_Type4']+data['Soil_Type6']+data['Soil_Type8']+data['Soil_Type9']+data['Soil_Type10']+data['Soil_Type11']+data['Soil_Type12']+data['Soil_Type13']+data['Soil_Type14']+data['Soil_Type16']+data['Soil_Type17']+data['Soil_Type18']+data['Soil_Type20']+data['Soil_Type25']+data['Soil_Type26']+data['Soil_Type28']+data['Soil_Type29']+data['Soil_Type30']+data['Soil_Type32']+data['Soil_Type33']+data['Soil_Type34']+data['Soil_Type36']

    data['Aspen_elev'] = data['Aspen'] * data['Elevation']

    data['Cottonwood/Willow_elev'] = data['Cottonwood/Willow'] * data['Elevation']

    data['Douglas-fir_elev'] = data['Douglas-fir'] * data['Elevation']

    data['Krummholz_elev'] = data['Krummholz'] * data['Elevation']

    data['Lodgepole Pine_elev'] = data['Lodgepole Pine'] * data['Elevation']

    data['Ponderosa Pine_elev'] =data['Ponderosa Pine'] * data['Elevation']

    data['Spruce/Fir_elev'] = data['Spruce/Fir'] * data['Elevation']

    data['Aspen_elev2'] = data['Aspen'] * data['Elevation_of_hydrology2']

    data['Cottonwood/Willow_elev2'] = data['Cottonwood/Willow'] * data['Elevation_of_hydrology2']

    data['Douglas-fir_elev2'] = data['Douglas-fir'] * data['Elevation_of_hydrology2']

    data['Krummholz_elev2'] = data['Krummholz'] * data['Elevation_of_hydrology2']

    data['Lodgepole Pine_elev2'] = data['Lodgepole Pine'] * data['Elevation_of_hydrology2']

    data['Ponderosa Pine_elev2'] =data['Ponderosa Pine'] * data['Elevation_of_hydrology2']

    data['Spruce/Fir_elev2'] = data['Spruce/Fir'] * data['Elevation_of_hydrology2']





soil_columns = [col for col in train.columns if col.startswith('Soil')]



# we can drop soil columns because we have group each soil according to their stoyness



for data in combined_data:

    data.drop(columns = soil_columns, inplace=True)

    

# for categorical value (up the water) we can encode them into dummy values  

columns_to_encode = ['Up_the_water']

        

train = pd.get_dummies(train,columns = columns_to_encode)

test = pd.get_dummies(test,columns = columns_to_encode)



# once we have encoded the columns we can drop them from both training and testing dataset



for data in combined_data:

    data.drop(columns= columns_to_encode, inplace=True)



target = 'Cover_Type'

features = [ col for col in train.columns if col not in ['Id','Cover_Type','Cover_type_description']]





X = train[features]

y = train[target]

    




X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)



#Random Forest





random_forest = RandomForestClassifier()

params_random_forest = {'n_estimators':[150,200],'criterion' : ['entropy','gini'], 'max_depth': [30,35,40]}

grid_search_random_forest = GridSearchCV(random_forest, param_grid= params_random_forest, cv=5, n_jobs=-1)

grid_search_random_forest.fit(X_train,y_train)



forest = RandomForestClassifier(n_estimators=grid_search_random_forest.best_params_['n_estimators'],criterion=grid_search_random_forest.best_params_['criterion'],max_depth=grid_search_random_forest.best_params_['max_depth'] )

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)



print("Accuracy random forest :", accuracy_score(y_test, y_pred))
gradient_boosting = GradientBoostingClassifier()

params_gradient_boosting = {'n_estimators':[150,200], 'max_depth': [30,35,40]}

grid_search_gradient_boosting = GridSearchCV(gradient_boosting, param_grid= params_gradient_boosting, cv=5, n_jobs=-1)

grid_search_gradient_boosting.fit(X_train,y_train)



boosting = GradientBoostingClassifier(n_estimators=grid_search_random_forest.best_params_['n_estimators'],max_depth=grid_search_random_forest.best_params_['max_depth'] )

boosting.fit(X_train, y_train)



y_pred = boosting.predict(X_test)



print("Accuracy gradient_classifier :", accuracy_score(y_test, y_pred))
extra_tree = ExtraTreesClassifier()

params_extra_tree = {'n_estimators':[150,200], 'max_depth': [30,35,40]}

grid_search_extra_tree = GridSearchCV(extra_tree, param_grid=params_extra_tree , cv=5, n_jobs=-1)

grid_search_extra_tree.fit(X_train,y_train)



extra_tree = ExtraTreesClassifier(n_estimators=grid_search_extra_tree.best_params_['n_estimators'],criterion='entropy',max_depth=grid_search_extra_tree.best_params_['max_depth'] ,  bootstrap=True)

extra_tree.fit(X_train,y_train)



y_pred = extra_tree.predict(X_test)



print("Accuracy extra_tree classifier :", accuracy_score(y_test, y_pred))

#Kfold cross validation for stacking clssifiers

kf = KFold(n_splits=10, shuffle = True)



classifiers = [forest,boosting,extra_tree]



stacking_clf = StackingCVClassifier(classifiers=classifiers,use_probas=True,meta_classifier=forest,cv=kf, use_features_in_secondary=True, random_state=1)



stacking_clf.fit(X_train, y_train)



y_pred = extra_tree.predict(X_test)



print("Accuracy stacking classifier :", accuracy_score(y_test, y_pred))



cover_type = ['Spruce/Fir', 'Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']

cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index = cover_type, columns= cover_type)



print(cm)
#Submitting

X_sub = test[features]

y_pred_sub = stacking_clf.predict(X_sub)



    

sub = pd.DataFrame({'Id': test.Id, 'Cover_Type': y_pred_sub})

sub.to_csv('submission.csv', index = False)