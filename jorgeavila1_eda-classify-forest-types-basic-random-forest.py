import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('/kaggle/input/learn-together/train.csv')

test_data = pd.read_csv('/kaggle/input/learn-together/test.csv')
print (train_data.shape, test_data.shape)

print( 'data features')

print(train_data.dtypes.value_counts())

print( 'test_data features')

print(test_data.dtypes.value_counts())

print('Number of soils:',sum(['Soil' in column for column in train_data.columns]))

print('Number of Wilderness:',sum(['Wilderness' in column for column in train_data.columns]))

train_data.head()
#No nan values

train_data.isna().sum()
#Generate categorical features Soil and Wilderness_Area to improve the plots

train_data["Soil"] = np.nan

for n,col in enumerate(['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']):

    train_data.Soil[train_data[col] == 1] = n+1



train_data["Wilderness_Area"] = np.nan

for n,col in enumerate(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']):

    train_data.Wilderness_Area[train_data[col] == 1] = n+1

    

test_data["Soil"] = np.nan

for n,col in enumerate(['Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']):

    test_data.Soil[test_data[col] == 1] = n+1



test_data["Wilderness_Area"] = np.nan

for n,col in enumerate(['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']):

    test_data.Wilderness_Area[test_data[col] == 1] = n+1
#change aspect feature

def Compass(row):

    if row.Aspect < 45 or row.Aspect > 315:

        return 1

    elif row.Aspect < 135:

        return 2

    elif row.Aspect < 225:

        return 3

    else:

        return 4

    

train_data['Compass'] = train_data.apply(Compass, axis='columns')

test_data['Compass'] = test_data.apply(Compass, axis='columns')
feature = "Elevation"

sns.catplot(x="Cover_Type", y=feature, kind="violin", hue = 'Wilderness_Area',  data=train_data, height=7, aspect=3)

plt.show()
#Interesting features: 'Elevation' 'Soil' 'Wilderness_Area'

g = sns.PairGrid(train_data[["Cover_Type",'Elevation','Aspect','Soil','Wilderness_Area','Compass']])

g.map(plt.scatter)
for n in range(1,8):

    sns.distplot(train_data[train_data.Cover_Type == n].Hillshade_3pm)

plt.legend()
#We have the same amount of each type

train_data.Cover_Type.value_counts()
train_data.columns

columns_v1 = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points','Soil', 'Wilderness_Area']
train_data.columns
#Model with columns without modifications + Soil and Wilderness_Area with Compass

interest_columns = train_data.columns

interest_columns = interest_columns.drop(['Cover_Type','Id'])



X = train_data[interest_columns]

y = train_data.Cover_Type

X_test = test_data[interest_columns]

my_model = RandomForestClassifier(n_estimators = 100, random_state=42)

my_model = RandomForestClassifier(n_estimators = 719,

                                       max_features = 0.3,

                                       max_depth = 464,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state=42)



#CrossValidation

scores = cross_val_score(my_model, X, y, cv=5, scoring = 'accuracy')

print(scores)
#first trial de predictions

my_model.fit(X,y)

predictions = my_model.predict(X_test)



my_submission = pd.DataFrame({'Id': test_data.Id, 'Cover_Type': predictions})

my_submission.to_csv('submission.csv', index=False)
##PERMUTATION IMPORTANCE

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=1).fit(train_X, train_y)



import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



features_to_plot = ['Elevation', 'Wilderness_Area']

inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=val_X.columns, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()