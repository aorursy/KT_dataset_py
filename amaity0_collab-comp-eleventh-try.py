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
train = pd.read_csv('../input/learn-together/train.csv', index_col='Id')

test = pd.read_csv('../input/learn-together/test.csv', index_col='Id')
import os

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings("ignore")
def addFeatures(df):

    #horizontal and vertical distance to hydrology can be easily combined

    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']

    df['distance_to_hydrology'] = df[cols].apply(np.linalg.norm, axis=1)

    

    #adding a few combinations of distance features to help enhance the classification

    cols = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',

            'Horizontal_Distance_To_Hydrology']

    df['distance_mean'] = df[cols].mean(axis=1)

    df['distance_sum'] = df[cols].sum(axis=1)

    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]

    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]

    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]

    

    #taking some factors influencing the amount of radiation

    df['cosine_of_slope'] = np.cos(np.radians(df['Slope']) )

    #X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))

    #X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))

    #X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))



    #sum of Hillshades

    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']

    #df['Sum_of_shades'] = df[shades].sum(1)

    weights = pd.Series([0.299, 0.587, 0.114], index=cols)

    df['hillshade'] = (df[shades]*weights).sum(1)



    df['elevation_vdh'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']

    print('Total number of features : %d' % (df.shape)[1])

    return df
def preprocessData(train, test):



    y_train = train['Cover_Type']

    

    classes = train.Cover_Type.unique()

    num_classes = len(classes)

    print("There are %i classes: %s " % (num_classes, classes))

    train.drop(['Cover_Type'], axis=1, inplace=True)



    train = addFeatures(train)    

    test = addFeatures(test)



    dtrn_first_ten = train.loc[:,:'Horizontal_Distance_To_Fire_Points']

    dtrn_wa_st = train.loc[:,'Wilderness_Area1':'Soil_Type40']

    dtrn_added_features = train.loc[:,'distance_to_hydrology':]

    dtrn_ = pd.concat([dtrn_first_ten,dtrn_added_features,dtrn_wa_st],axis=1)



    dtst_first_ten = test.loc[:,:'Horizontal_Distance_To_Fire_Points']

    dtst_wa_st = test.loc[:,'Wilderness_Area1':'Soil_Type40']

    dtst_added_features = test.loc[:,'distance_to_hydrology':]

    dtst_ = pd.concat([dtst_first_ten,dtst_added_features,dtst_wa_st],axis=1)

    

    # elevation was found to have very different distributions on test and training sets

    # lets just drop it for now to see if we can implememnt a more robust classifier!

    #train = train.drop('Elevation', axis=1)

    #test = test.drop('Elevation', axis=1)    



    return dtrn_, dtst_, y_train
X, test, y = preprocessData(train, test)
weights = [11.393577400361757, 1.4282825089634368, 0.6063107664752647, 1, 1.916980442614397, 

1.0945477432742674, 1.668754279754504, 1.7520168478233817, 8.207420802921982, 

0.7501841943847916, 1.9971420119714571, 2.72057743717325, 2.0, 1.575220244799055, 

2.0695773922466643, 2.536316322049836, 0.46168425088806536, 0.4420755307264942, 

10.660977569012896, 876.0230240897795, 795.52134403456]
from sklearn.neighbors import KNeighborsClassifier

clf1 = KNeighborsClassifier(n_neighbors=1,p=1)



X_copy = X.copy()

test_copy = test.copy()



for i in range(19):

    c = X.columns[i]

    X_copy[c] = weights[i]*X_copy[c]

    test_copy[c] = weights[i]*test_copy[c]

for i in range(19,23):

    c = X.columns[i]

    X_copy[c] = weights[19]*X_copy[c]

    test_copy[c] = weights[19]*test_copy[c]

for i in range(23,len(X.columns)):

    c = X.columns[i]

    X_copy[c] = weights[20]*X_copy[c]

    test_copy[c] = weights[20]*test_copy[c]
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=300, max_features='sqrt', bootstrap=False,max_depth=60,

                              min_samples_split=2,min_samples_leaf=1,random_state=1)

from sklearn.ensemble import ExtraTreesClassifier

clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=5,

                             min_samples_leaf=1,max_features=63,random_state=1)

from lightgbm import LGBMClassifier

clf4 = LGBMClassifier(num_leaves=109,objective='multiclass',num_class=7,

                       learning_rate=0.2,random_state=1)

from mlxtend.classifier import EnsembleVoteClassifier

eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4])

labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'LGBM', 'Ensemble']

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], labels):

    scores = cross_val_score(clf, X_copy, y, cv=5, scoring='accuracy')

    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
eclf.fit(X_copy,y)

preds_test = eclf.predict(test_copy)

print(preds_test[:10])
# Make the submission file

output = pd.DataFrame({'Id': test.index,'Cover_type': preds_test})

output.to_csv('submission.csv', index=False)