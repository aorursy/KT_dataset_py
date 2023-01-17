import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X = train.drop(['Id', 'Cover_Type'], axis=1)

y = train['Cover_Type']

X.head(), y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
from sklearn.metrics import classification_report,accuracy_score






def add_features(data):

    #data['Euclidean_Distance_To_Hydrology_'] = (data['Horizontal_Distance_To_Hydrology']**2 + data['Vertical_Distance_To_Hydrology']**2)**0.5

    #data['Mean_Distance_To_Amenities_'] = (data['Horizontal_Distance_To_Fire_Points'] + data['Horizontal_Distance_To_Hydrology'] + data['Horizontal_Distance_To_Roadways']) / 3.0

    #data['Elevation_Minus_Vertical_Distance_To_Hydrology_'] = data['Elevation'] - data['Vertical_Distance_To_Hydrology']

    

    data['HF1'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']

    data['HF2'] = abs(data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points'])

    data['HR1'] = abs(data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways'])

    data['HR2'] = abs(data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways'])

    data['FR1'] = abs(data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways'])

    data['FR2'] = abs(data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways'])



    # Pythagoras theorem

    data['slope_hyd'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5

    data.slope_hyd=data.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)



    # Means

    data['Mean_Amenities']=(data.Horizontal_Distance_To_Fire_Points + data.Horizontal_Distance_To_Hydrology + data.Horizontal_Distance_To_Roadways) / 3  

    data['Mean_Fire_Hyd']=(data.Horizontal_Distance_To_Fire_Points + data.Horizontal_Distance_To_Hydrology) / 2 

    

    #soil

    soil_types = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', \

                  'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', \

                  'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', \

                  'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', \

                  'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

    data[soil_types] = data[soil_types].multiply([i for i in range(1, 41)], axis=1)

    data['soil_type'] = data[soil_types].sum(axis=1)

    

    #wilderness

    data[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']] = data[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].multiply([1, 2, 3, 4], axis=1)

    data['Wilderness_Area'] = data[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']].sum(axis=1)

    

    

    #data = data.drop(['Soil_Type9', 'Soil_Type8', 'Soil_Type37', 'Soil_Type36', 'Soil_Type35', 'Soil_Type34', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type21', 'Soil_Type18', 'Soil_Type19', 'Soil_Type15', 'Soil_Type14', 'Soil_Type16' ], axis=1)

    return data



train = add_features(X)

test = add_features(test)
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.4, random_state=10)

X_train.head(), y_train.head()
#tuned_parameters = [{ 'max_depth': list([10,20,40,50]), 'n_estimators':list([200,300,400,500,600])}]

from scipy.stats import uniform

from scipy.stats import norm

n_estimators = np.random.uniform(500, 1000, 5).astype(int)

max_features = np.random.normal(30, 80, 5).astype(int)

 

# Check max_features>0 & max_features<=total number of features

max_features[max_features <= 0] = 1

max_features[max_features > X.shape[1]] = X.shape[1]

 



hyperparameters = {'n_estimators': list(n_estimators),

                   'max_features': list(max_features)}
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV





#exe=ExtraTreesClassifier()

#pipeline=Pipeline(exe,clf)

#print(clf)

clf=RandomForestClassifier(n_estimators=938,max_features=38)

#clf = ExtraTreesClassifier(criterion='gini',max_depth=60, n_estimators=700)



clf.fit(train, y)

#print("as")

y_pred = clf.predict(X_test)

n_correct = (y_pred == y_test).sum()

n_total = (y_pred == y_test).count()

print('Accuracy:', n_correct/n_total)



prediction_classes = pd.Series(clf.predict(test.drop('Id', axis=1))).rename('Cover_Type')

predictions = pd.concat([test['Id'], prediction_classes], axis=1).reset_index().drop('index', axis=1)

predictions.to_csv('submission.csv', index=False)

predictions.head()