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

import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing
train = pd.read_csv(r'../input/learn-together/train.csv')

test = pd.read_csv(r'../input/learn-together/test.csv')



print("Info training dataset :")

print(train.shape)

print("Null values for training dataset:")

print(train.isnull().sum())





print("Info testing dataset :")

print(test.shape)

print("Null values for testing dataset:")

print(test.isnull().sum())

# adding description to Cover_Type

cover_type = {1:'Spruce/Fir', 2:'Lodgepole Pine',3:'Ponderosa Pine',4:'Cottonwood/Willow',5:'Aspen',6:'Douglas-fir',7:'Krummholz'}

train['Cover_type_description'] = train['Cover_Type'].map(cover_type)
list_of_cover_type = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine','Cottonwood/Willow','Aspen','Douglas-fir','Krummholz']



#distribution of Aspect by cover type



for cover_type in list_of_cover_type:

    plt.title("Cover type by Aspect")

    sns.distplot(a = train[train['Cover_type_description'] == cover_type]['Aspect'], label = cover_type, rug = True)

    plt.legend()

plt.show()
#decoding of Aspect feature

#TRAINING

train['Azimut'] = 0

train['Azimut'].loc[(train['Aspect'] == 0)] = 'north' 

train['Azimut'].loc[(train['Aspect'] > 0) & (train['Aspect'] < 90)] = 'north_east' 

train['Azimut'].loc[(train['Aspect'] == 90)] = 'east' 

train['Azimut'].loc[(train['Aspect'] > 90) &(train['Aspect'] < 180)] = 'south_east' 

train['Azimut'].loc[(train['Aspect'] == 180)] = 'south'

train['Azimut'].loc[(train['Aspect'] > 180) & (train['Aspect'] < 270)] = 'south_west' 

train['Azimut'].loc[(train['Aspect'] == 270)] = 'west'

train['Azimut'].loc[(train['Aspect'] > 270) &(train['Aspect'] < 360)] = 'north_west' 

train['Azimut'].loc[(train['Aspect'] == 360)] = 'north'



#TEST

test['Azimut'] = 0

test['Azimut'].loc[(test['Aspect'] == 0)] = 'north' 

test['Azimut'].loc[(test['Aspect'] > 0) & (test['Aspect'] < 90)] = 'north_east' 

test['Azimut'].loc[(test['Aspect'] == 90)] = 'east' 

test['Azimut'].loc[(test['Aspect'] > 90) &(test['Aspect'] < 180)] = 'south_east' 

test['Azimut'].loc[(test['Aspect'] == 180)] = 'south'

test['Azimut'].loc[(test['Aspect'] > 180) & (test['Aspect'] < 270)] = 'south_west' 

test['Azimut'].loc[(test['Aspect'] == 270)] = 'west'

test['Azimut'].loc[(test['Aspect'] > 270) &(test['Aspect'] < 360)] = 'north_west' 

test['Azimut'].loc[(test['Aspect'] == 360)] = 'north'
for cover_type in list_of_cover_type:

    plt.title("Countplot by Azimut in training dataset")

    sns.countplot(x = train['Azimut'])

plt.show()
for cover_type in list_of_cover_type:

    plt.title("Cover type by Elevation")

    sns.distplot(a = train[train['Cover_type_description'] == cover_type]['Elevation'], label = cover_type)

    plt.legend()

plt.show()

#Elevation grouping  for testing and training



train['Elevation_bins'] = 0

train['Elevation_bins'].loc[train['Elevation'] <= 2000] = 'less than 2000'

train['Elevation_bins'].loc[(train['Elevation'] > 2000) & (train['Elevation'] <= 2500)] = 'between 2000 and 2500'

train['Elevation_bins'].loc[(train['Elevation'] > 2500) & (train['Elevation'] <= 3000)] = 'between 2000 and 3000'

train['Elevation_bins'].loc[(train['Elevation'] > 3000) & (train['Elevation'] <= 3500)] = 'between 3000 and 3500'

train['Elevation_bins'].loc[train['Elevation'] > 3500] = 'greater than 3500'





test['Elevation_bins'] = 0

test['Elevation_bins'].loc[test['Elevation'] <= 2000] = 'less than 2000'

test['Elevation_bins'].loc[(test['Elevation'] > 2000) & (test['Elevation'] <= 2500)] = 'between 2000 and 2500'

test['Elevation_bins'].loc[(test['Elevation'] > 2500) & (test['Elevation'] <= 3000)] = 'between 2000 and 3000'

test['Elevation_bins'].loc[(test['Elevation'] > 3000) & (test['Elevation'] <= 3500)] = 'between 3000 and 3500'

test['Elevation_bins'].loc[test['Elevation'] > 3500] = 'greater than 3500'
for cover_type in list_of_cover_type:

    plt.title("Cover Type by Slope")

    sns.distplot(a = train[train['Cover_type_description'] == cover_type]['Slope'], label = cover_type)

    plt.legend()

plt.show()
print("Max slope for training")

print(np.max(train['Slope']))



print("Max slope for testing")

print(np.max(test['Slope']))
# create category for slope for training and test dataset <=10, >10 and <= 20, > 20 and <=30 , > 30



test['Slope_category'] = 0

test['Slope_category'].loc[(test['Slope'] <= 10)] = 'slope less than 10'

test['Slope_category'].loc[(test['Slope'] > 10) & (test['Slope'] <= 20)] = 'slope between 10 and 20'

test['Slope_category'].loc[(test['Slope'] > 20) & (test['Slope'] <= 30)] = 'slope between 20 and 30'

test['Slope_category'].loc[(test['Slope'] > 30)] = 'slope greater than 30'



train['Slope_category'] = 0

train['Slope_category'].loc[(train['Slope'] <= 10)] = 'slope less than 10'

train['Slope_category'].loc[(train['Slope'] > 10) & (train['Slope'] <= 20)] = 'slope between 10 and 20'

train['Slope_category'].loc[(train['Slope'] > 20) & (train['Slope'] <= 30)] = 'slope between 20 and 30'

train['Slope_category'].loc[(train['Slope'] > 30)] = 'slope greater than 30'
train['mean_Hillshade'] = (train['Hillshade_9am']+ train['Hillshade_Noon']+train['Hillshade_3pm'])/3

test['mean_Hillshade'] = (test['Hillshade_9am']+ test['Hillshade_Noon']+test['Hillshade_3pm'])/3
def distance(a,b):

    return (np.sqrt(np.power(a,2)+np.power(b,2)))



train['Distance_to_hidrology'] = distance(train['Horizontal_Distance_To_Hydrology'],train['Vertical_Distance_To_Hydrology'])

test['Distance_to_hidrology'] = distance(test['Horizontal_Distance_To_Hydrology'],test['Vertical_Distance_To_Hydrology'])
for cover_type in list_of_cover_type:

    plt.title("Cover Type by Distance to hidrology  in training")

    sns.distplot(a = train[train['Cover_type_description'] == cover_type]['Distance_to_hidrology'], label = cover_type)

    plt.legend()

plt.show()
for cover_type in list_of_cover_type:

    plt.title("Cover Type by Distance to fire points")

    sns.distplot(a = train[train['Cover_type_description'] == cover_type]['Horizontal_Distance_To_Fire_Points'], label = cover_type)

    plt.legend()

plt.show()
#relationship for training dataset

columns = ['Elevation', 'Aspect','Slope', 'mean_Hillshade']

#for cover_type in list_of_cover_type:

sns.pairplot(train, hue = "Cover_type_description", vars = columns)

plt.legend()

plt.show()
columns_to_drop = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm']



train.drop(columns= columns_to_drop, inplace=True)

test.drop(columns= columns_to_drop, inplace=True)
columns_to_encode = ['Azimut','Elevation_bins','Slope_category']



train  = pd.get_dummies(train, columns = columns_to_encode)

test = pd.get_dummies(test, columns = columns_to_encode)


columns_to_drop_train = ['Cover_type_description']



train.drop(columns= columns_to_drop_train,inplace=True)
s_scaler = preprocessing.StandardScaler()

columns_to_scaled = ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points','mean_Hillshade', 'Distance_to_hidrology']

train_to_scaled = train[columns_to_scaled]

training_scaled = s_scaler.fit_transform(train_to_scaled)

training_scaled_df = pd.DataFrame(data = training_scaled, columns = columns_to_scaled)
df_training_concatenated = pd.concat([train[['Id','Cover_Type','Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40','Azimut_east',

       'Azimut_north', 'Azimut_north_east', 'Azimut_north_west',

       'Azimut_south', 'Azimut_south_east', 'Azimut_south_west', 'Azimut_west',

       'Elevation_bins_between 2000 and 2500',

       'Elevation_bins_between 2000 and 3000',

       'Elevation_bins_between 3000 and 3500',

       'Elevation_bins_greater than 3500', 'Elevation_bins_less than 2000',

       'Slope_category_slope between 10 and 20',

       'Slope_category_slope between 20 and 30',

       'Slope_category_slope greater than 30',

       'Slope_category_slope less than 10']],training_scaled_df],axis =1)
test_to_scaled = test[columns_to_scaled]

testing_scaled = s_scaler.fit_transform(test_to_scaled)

testing_scaled_df = pd.DataFrame(data = testing_scaled, columns = columns_to_scaled)    



df_testing_concatenated = pd.concat([test[['Id','Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40','Azimut_east',

       'Azimut_north', 'Azimut_north_east', 'Azimut_north_west',

       'Azimut_south', 'Azimut_south_east', 'Azimut_south_west', 'Azimut_west',

       'Elevation_bins_between 2000 and 2500',

       'Elevation_bins_between 2000 and 3000',

       'Elevation_bins_between 3000 and 3500',

       'Elevation_bins_greater than 3500', 'Elevation_bins_less than 2000',

       'Slope_category_slope between 10 and 20',

       'Slope_category_slope between 20 and 30',

       'Slope_category_slope greater than 30',

       'Slope_category_slope less than 10']],testing_scaled_df],axis =1)  

    
target = 'Cover_Type'

features = ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',

   'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

   'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

   'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

   'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

   'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

   'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

   'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

   'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

   'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

   'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

   'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',

   'mean_Hillshade', 'Distance_to_hidrology', 'Azimut_east',

   'Azimut_north', 'Azimut_north_east', 'Azimut_north_west',

   'Azimut_south', 'Azimut_south_east', 'Azimut_south_west', 'Azimut_west',

   'Elevation_bins_between 2000 and 2500',

   'Elevation_bins_between 2000 and 3000',

   'Elevation_bins_between 3000 and 3500',

   'Elevation_bins_greater than 3500', 'Elevation_bins_less than 2000',

   'Slope_category_slope between 10 and 20',

   'Slope_category_slope between 20 and 30',

   'Slope_category_slope greater than 30',

   'Slope_category_slope less than 10']

y = df_training_concatenated[target]

X = df_training_concatenated[features]



X_train, X_valid, y_train, y_valid = train_test_split(X, y,  test_size=0.3,random_state=0)
print("XGBOOST with grid search")

xgbclassifier = XGBClassifier()

params_xgbclassifier = {"n_estimators": [ 50, 100,150],"learning_rate":[0.01, 0.03,0.05]}

grid_search_xgboost = GridSearchCV(xgbclassifier, param_grid= params_xgbclassifier, cv=5, n_jobs=-1)

grid_search_xgboost.fit(X_train,y_train)

print("best parameter for xgboost_classifier ", grid_search_xgboost.best_params_)
model = XGBClassifier(n_estimators=grid_search_xgboost.best_params_['n_estimators'], random_state=0,learning_rate=grid_search_xgboost.best_params_['learning_rate'])

model.fit(X_train, y_train)

preds = model.predict(X_valid)

x_test = test[features]

y_prediction = model.predict(x_test)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state = 42).fit(X_valid,y_valid)

eli5.show_weights(perm, feature_names = X_valid.columns.tolist())

import shap



explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_valid)

shap.summary_plot(shap_values[1], X_valid)

    