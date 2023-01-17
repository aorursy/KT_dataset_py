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
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 70)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.simplefilter(action='ignore')
train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')
cover_type = {1:'Spruce/Fir', 2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

train['Cover_type_description'] = train['Cover_Type'].map(cover_type)



combined_data = [train, test]



def distance(a,b):

    return np.sqrt(np.power(a,2)+np.power(b,2))



extremely_stony = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1]

stony = [0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

rubbly = [0,0,1,1,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

other = [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]

    



for data in combined_data:

    

    # grouping of Aspect feature

    data['Azimut'] = 0

    data['Azimut'].loc[(data['Aspect'] == 0)] = 'north'

    data['Azimut'].loc[(data['Aspect']>0 ) & (data['Aspect']< 90)] = 'north_east'

    data['Azimut'].loc[data['Aspect']==90] = 'east'

    data['Azimut'].loc[(data['Aspect'] >90) & (data['Aspect'] <180)] = 'south_east'

    data['Azimut'].loc[data['Aspect']==180] = 'south'

    data['Azimut'].loc[(data['Aspect']>180) & (data['Aspect']<270)] = 'south_west'

    data['Azimut'].loc[data['Aspect']== 270] = 'west'

    data['Azimut'].loc[(data['Aspect']> 270) & (data['Aspect']< 360)] = 'noth_west'

    data['Azimut'].loc[data['Aspect']== 360] = 'north'

    

    #grouping of Elevation feature

    

    data['Elevation_bins'] = 0

    data['Elevation_bins'].loc[data['Elevation']<= 2000] = 'less than 2000'

    data['Elevation_bins'].loc[(data['Elevation'] > 2000) & (data['Elevation']< 2500)] = 'between 2000 and 2500'

    data['Elevation_bins'].loc[(data['Elevation'] > 2500) & (data['Elevation'] <= 3000)] = 'between 2000 and 3000'

    data['Elevation_bins'].loc[(data['Elevation'] > 3000) & (data['Elevation'] <= 3500)] = 'between 3000 and 3500'

    data['Elevation_bins'].loc[data['Elevation'] > 3500] = 'greater than 3500'

    

    # grouping for slope

    data['Slope_category'] = 0

    data['Slope_category'].loc[(data['Slope'] <= 10)] = 'slope less than 10'

    data['Slope_category'].loc[(data['Slope'] > 10) & (data['Slope'] <= 20)] = 'slope between 10 and 20'

    data['Slope_category'].loc[(data['Slope'] > 20) & (data['Slope'] <= 30)] = 'slope between 20 and 30'

    data['Slope_category'].loc[(data['Slope'] > 30)] = 'slope greater than 30'

    

    data['mean_Hillshade'] = (data['Hillshade_9am']+ data['Hillshade_Noon']+data['Hillshade_3pm'])/3

    

    data['Distance_to_hidrology'] = distance(data['Horizontal_Distance_To_Hydrology'],data['Vertical_Distance_To_Hydrology'])

    



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

    data['Total_horizontal_distance'] = data['Horizontal_Distance_To_Hydrology']+ data['Horizontal_Distance_To_Roadways']+ data['Horizontal_Distance_To_Fire_Points']

    data['Elevation_of_hydrology'] = data['Elevation']+ data['Vertical_Distance_To_Hydrology']

    data['Distance_to_firepoints plus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']+ data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads plus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] + data['Horizontal_Distance_To_Hydrology']

    data['Distance_to_firepoints minus Distance_to_roads'] = data['Horizontal_Distance_To_Fire_Points']- data['Horizontal_Distance_To_Roadways']

    data['Distance_to_roads minus distance_to_hydrology'] = data['Horizontal_Distance_To_Roadways'] - data['Horizontal_Distance_To_Hydrology']





soil_columns = [col for col in train.columns if col.startswith('Soil')]





# we can drop soil columns because we have group each soil according to their stoyness



for data in combined_data:

    data.drop(columns = soil_columns, inplace=True)

    
# for categorical values we can encode them into dummy values 

    

columns_to_encode = ['Azimut','Elevation_bins','Slope_category','Up_the_water']

    

train = pd.get_dummies(train,columns = columns_to_encode)

test = pd.get_dummies(test,columns = columns_to_encode)



# once we have encoded the columns we can drop them from both training and testing dataset



for data in combined_data:

    data.drop(columns= columns_to_encode, inplace=True)



# after encoding all variables are numerical but, as we can see, thier scale is very different; there are variable encoded whose value is 0 or 1 and there are some

# others variables like Elevation who are much bigger

# we need to scale variable in order to make them confrontable



columns_to_scale = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'mean_Hillshade', 'Distance_to_hidrology','Hillshade_noon_3pm', 'Hillshade_3pm_9am',

       'Hillshade_9am_noon', 'Total_horizontal_distance',

       'Elevation_of_hydrology',

       'Distance_to_firepoints plus Distance_to_roads',

       'Distance_to_roads plus distance_to_hydrology',

       'Distance_to_firepoints minus Distance_to_roads',

       'Distance_to_roads minus distance_to_hydrology']

    



s_scaler = preprocessing.StandardScaler()

train_columns_to_scale = train[columns_to_scale]

test_columns_to_scale = test[columns_to_scale]



train_scaled = s_scaler.fit_transform(train_columns_to_scale)

test_scaled = s_scaler.fit_transform(test_columns_to_scale)



train_scaled_df = pd.DataFrame(data = train_scaled, columns = columns_to_scale)

test_scaled_df = pd.DataFrame(data = test_scaled, columns = columns_to_scale)



# dropping columns scaled from training and testing dataset

train.drop(columns = columns_to_scale, inplace = True)

test.drop(columns = columns_to_scale, inplace = True)



#now we can concatenate scaled columns to both training and testing dataset

    

train_final = pd.concat([train , train_scaled_df],axis = 1)

test_final = pd.concat([test  ,test_scaled_df],axis = 1)


#defining targer variable and features

target = 'Cover_Type'

features = [ col for col in train_final.columns if col not in ['Id','Cover_Type','Cover_type_description']]



X = train_final[features]

y = train_final[target]



#BUILDING THE MODEL



X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)



from sklearn.model_selection import KFold



Kfold = KFold(n_splits= 50,shuffle = True,  random_state= 1)



#GRID_SEARCH IN ORDER TO FIND OUT BEST PARAMETERS FOR THE MODEL

    

random_forest  = RandomForestClassifier()

params_decision_random_forest = {'n_estimators':[100,150,200], 'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2']}

grid_search_random_forest = GridSearchCV(random_forest, param_grid =params_decision_random_forest, cv=Kfold,scoring= 'neg_mean_squared_error', n_jobs=-1)

grid_search_random_forest.fit(X_train, y_train)

print('Best parameters for random forest classifier:', grid_search_random_forest.best_params_)



decision_random_forest = RandomForestClassifier(n_estimators = grid_search_random_forest.best_params_['n_estimators'],criterion = grid_search_random_forest.best_params_['criterion'], max_features = grid_search_random_forest.best_params_['max_features']) 

decision_random_forest.fit(X_train,y_train)

X_test = test_final[features]

y_pred = decision_random_forest.predict(X_test)



sub = pd.DataFrame({'Id': test.Id, 'Cover_Type': y_pred})

sub.to_csv('submission_csv', index = False)