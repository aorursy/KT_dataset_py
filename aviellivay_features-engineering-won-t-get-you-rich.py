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
train = pd.read_csv("../input/learn-together/train.csv", index_col=0)

test = pd.read_csv("../input/learn-together/test.csv", index_col=0)

X = train.copy()

X = X.drop('Cover_Type', 1)

y = train['Cover_Type']
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns

import matplotlib.pyplot as plt
model = RandomForestClassifier(n_estimators=100)
model = model.fit(X,y)
from sklearn.model_selection import KFold, cross_val_score



cv = KFold(n_splits=5, shuffle=True, random_state=1)



def cross_val(model, X=X, y=y):

    cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return cv_results



print("cross_val = {}".format(cross_val(model, X, y)))

def add_features(X_):

    X = X_.copy()

    

    X['Log_elevation'] = X['Elevation'].apply(np.log)

    

    X['Hill_Shade_Mean'] = X[['Hillshade_9am',

                              'Hillshade_Noon',

                              'Hillshade_3pm']

                            ].mean(axis='columns')

    

    X['Hillshade_9am_squared'] = X['Hillshade_9am'].apply(np.square)

    X['Hillshade_Noon_squared'] = X['Hillshade_Noon'].apply(np.square)

    X['Hillshade_3pm_squared'] = X['Hillshade_3pm'].apply(np.square)

    

    X['Cos_Slope'] = X['Slope'].apply(lambda a : np.cos(a/360*2*np.pi))

    

    X['Extremely_Stony'] = X[['Soil_Type' + soilN for soilN in ['24', '28', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40']]].sum(axis='columns')

    X['Rubbly'] = X[['Soil_Type' + soilN for soilN in ['3', '4', '5', '10', '11', '13']]].sum(axis='columns')

    X['Rock_Land'] = X[['Soil_Type' + soilN for soilN in ['12', '13', '30', '34' , '36']]].sum(axis='columns')

    X['Vanet'] = X[['Soil_Type' + soilN for soilN in ['2', '6']]].sum(axis='columns')

    X['Bullwark'] = X[['Soil_Type' + soilN for soilN in ['10', '11']]].sum(axis='columns')

    X['Hydro_Elevation_diff'] = X[['Elevation',

                                   'Vertical_Distance_To_Hydrology']

                                  ].diff(axis='columns').iloc[:, [1]]



    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 +

                                   X['Vertical_Distance_To_Hydrology']**2)



    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Fire_Points']

                            ].sum(axis='columns')



    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Fire_Points']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',

                             'Horizontal_Distance_To_Roadways']

                            ].sum(axis='columns')



    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',

                              'Horizontal_Distance_To_Roadways']

                             ].diff(axis='columns').iloc[:, [1]].abs()



    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',

                            'Horizontal_Distance_To_Fire_Points']

                           ].sum(axis='columns')



    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',

                             'Horizontal_Distance_To_Fire_Points']

                            ].diff(axis='columns').iloc[:, [1]].abs()

    

    cols = [

        'Horizontal_Distance_To_Roadways',

        'Horizontal_Distance_To_Fire_Points',

        'Hydro_Euclidean',

    ]

    X['distance_mean'] = X[cols].mean(axis=1)

    X['distance_road_fire'] = X[cols[:2]].mean(axis=1)

    X['distance_hydro_fire'] = X[cols[1:]].mean(axis=1)

    X['distance_road_hydro'] = X[[cols[0], cols[2]]].mean(axis=1)

    

    X['shade_noon_diff'] = X['Hillshade_9am'] - X['Hillshade_Noon']

    X['shade_3pm_diff'] = X['Hillshade_Noon'] - X['Hillshade_3pm']

    X['shade_mean'] = X[['Hillshade_9am', 

                           'Hillshade_Noon', 

                           'Hillshade_3pm']].mean(axis=1)

    

    X['ElevationHydro'] = X['Elevation'] - 0.25 * X['Hydro_Euclidean']

    X['ElevationV'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

    X['ElevationH'] = X['Elevation'] - 0.19 * X['Horizontal_Distance_To_Hydrology']

    X['Aspect_sin'] = np.sin(X.Aspect)

    X['Aspect_cos'] = np.cos(X.Aspect)

    X['Slope_sin'] = np.sin(X.Slope)

    X['Slope_cos'] = np.cos(X.Slope)

    

    return X





def drop_features(X_):

    X = X_.copy()

    drop_cols = ['Soil_Type1']

    

    X = X.drop(drop_cols, axis='columns')



    return X
X_copy = X.copy()

X_copy = add_features(X_copy)

X_copy = drop_features(X_copy)

test_copy = test.copy()

test_copy = add_features(test_copy)

test_copy = drop_features(test_copy)
print("cross_val = {}".format(cross_val(model, X_copy, y)))

fs_dict = {}

for trial in range(100):

    X_features = X_copy.copy()

    num_features = np.random.randint(10,len(X_copy.columns))

    for col in X_copy.columns:

        rand = np.random.randint(len(X_copy.columns))

        if rand > num_features:

            X_features = X_features.drop([col], axis='columns')

    cross_v = cross_val(model, X_features, y)

    print("Feature set # {}\n': X_features.columns = {}\n, cross_v = {}\n\n\n".format(trial, X_features.columns, cross_v))

    fs_dict[trial] = [np.mean(cross_v), X_features.columns]

best_features = sorted(fs_dict.items(), key=lambda kv: kv[1][0], reverse=True)[0][1][1].to_list()

best_features
sorted(fs_dict.items(), key=lambda kv: kv[1][0], reverse=True)
X_best = X_copy[best_features]

test_best = test_copy[best_features]
print("cross_val = {}".format(cross_val(model, X_best, y)))
def feature_importances(model, X, y, figsize=(18, 6)):

    model = model.fit(X, y)

    

    importances = pd.DataFrame({'Features': X.columns, 

                                'Importances': model.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=figsize)

    sns.barplot(x='Features', y='Importances', data=importances)

    plt.xticks(rotation='vertical')

    plt.show()



model.fit(X_copy, y)

feature_importances(model, X_copy, y)    
importances = pd.DataFrame({'Features': X_copy.columns, 

                            'Importances': model.feature_importances_})

    

importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
important_features_list = importances.iloc[0:50]['Features'].tolist()
X_important = X_copy[important_features_list]

test_important = test_copy[important_features_list]
print("cross_val = {}".format(cross_val(model, X_important, y)))
model.fit(X_best, y)

predicts = model.predict(test_best)



output = pd.DataFrame({'ID': test.index,

                       'Cover_Type': predicts})

output.to_csv('my_model.csv', index=False)