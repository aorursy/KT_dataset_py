from datetime import datetime



print("last update: {}".format(datetime.now())) 
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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,  GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier

import lightgbm as lgb

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score

from sklearn.svm import SVC 

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

import numpy as np

np.random.seed(0)
# Read the data

X_original = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

X = X_original.drop('Cover_Type', axis = 1)

y = X_original['Cover_Type']
X.tail()
X_test.head()
print(X['Soil_Type7'].unique(), X['Soil_Type15'].unique())
X.shape[0]
X.drop(['Soil_Type7', 'Soil_Type15'], axis = 1, inplace = True)

X_test.drop(['Soil_Type7', 'Soil_Type15'], axis = 1, inplace = True)
#Create single soil field (reverse the one hot encoding)

soil_fields = [col for col in X if col.startswith('Soil_Type')]

train_soil = X[soil_fields]

test_soil = X_test[soil_fields]

X['Soil_Type'] = train_soil.idxmax(axis = 1).apply(lambda x: x.split('Type')[-1]).astype(int)

X_test['Soil_Type'] = test_soil.idxmax(axis = 1).apply(lambda x: x.split('Type')[-1]).astype(int)

X.drop(soil_fields, inplace = True, axis = 1)

X_test.drop(soil_fields, inplace = True, axis = 1)

#Create single wilderness area field (reverse the one hot encoding)

Wilderness_Area_Fields = [col for col in X if col.startswith('Wilderness_Area')]



train_wilderness = X[Wilderness_Area_Fields]

test_wilderness = X_test[Wilderness_Area_Fields]

X['Wilderness_Area'] = train_wilderness.idxmax(axis = 1).apply(lambda x: x.split('Area')[-1]).astype(int)

X_test['Wilderness_Area'] = test_wilderness.idxmax(axis = 1).apply(lambda x: x.split('Area')[-1]).astype(int)

X.drop(Wilderness_Area_Fields, inplace = True, axis = 1)

X_test.drop(Wilderness_Area_Fields, inplace = True, axis = 1)

X.head()

print(X['Soil_Type'].unique())

len(X['Soil_Type'].unique())
X_test.shape
set(X['Soil_Type'].unique()) == set(X_test['Soil_Type'].unique()) 
# O stand for missing values

dict = {}

for col in X_test.columns.tolist()[:10]:

    dict[col] = X_test[X_test[col] == 0].shape[0]

dict 
var = list(dict.keys())

height = list(dict.values())

print(var)

print(height)
plt.figure(figsize = (15,7))

pos = np.arange(len(var))

bars = plt.barh(pos, height)

plt.yticks(pos, var)

plt.xlabel("# of zeros values")

plt.title('check for zeros values')

plt.tick_params(top = False, left = False, bottom = False, labelbottom = False)

# remove frames

for spine in plt.gca().spines.values():

    spine.set_visible(False)



for bar in bars:

    plt.gca().text(bar.get_width()+1.2, bar.get_y()+bar.get_height()/2, str(int(bar.get_width())))

    

    

bars[4].set_color('red')

bars[3].set_color('gray')
X[['Soil_Type', 'Wilderness_Area']] = X[['Soil_Type', 'Wilderness_Area']].astype(int)

X_test[['Soil_Type', 'Wilderness_Area']] = X_test[['Soil_Type', 'Wilderness_Area']].astype(int)
X.dtypes
import featuretools as ft

from featuretools import selection

from featuretools.primitives import make_trans_primitive

from featuretools.variable_types import Numeric

import featuretools.variable_types as vtypes
# Merge forest data

forest = X.append(X_test, sort=False)
forest.head()
print(forest.shape)

print(f"total rows:{X.shape[0] + X_test.shape[0]}")
len(forest['Soil_Type'].unique())
def power(column):

    return column**2



Power = make_trans_primitive(function = power,

                             input_types = [Numeric],

                             return_type=Numeric)

# Create an entity set

es = ft.EntitySet(id='forest_data')
ID = ft.variable_types.Id
variable_types = {'Soil_Type': vtypes.Categorical, 'Wilderness_Area': vtypes.Categorical, 'Id': ID}
es.entity_from_dataframe(entity_id = 'data', dataframe = forest.reset_index(),

                         index = 'Id',variable_types = variable_types)
es['data']
es.normalize_entity(base_entity_id= 'data',

                    new_entity_id='Soil',

                    index='Soil_Type')
es['Soil'].df.shape
es.normalize_entity(base_entity_id= 'data',

                    new_entity_id='Wild_Area',

                    index='Wilderness_Area')
es['Wild_Area'].shape
es.plot()
ft.list_primitives()['type'].value_counts()
# Loop on differents aggregation primitives

#pd.options.display.max_rows = 10

func = ft.list_primitives()

func[func['type'] == 'aggregation']
# Loop on differents transformation primitives

# import random  

from random import sample 

func = ft.list_primitives()

func[func['type'] == 'transform'].sample(10)
trans_primitives = ['add_numeric', 'multiply_numeric']
# Run deep feature synthesis with transformation primitives

feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data',

                                      agg_primitives = ['median', 'std', 'mean', 'mode'],

                                      trans_primitives = ['add_numeric'],

                                      n_jobs = -1,

                                     verbose = True)



feature_matrix.head()
# Remove low information features

feature_matrix = selection.remove_low_information_features(feature_matrix)
feature_matrix.shape
from random import sample

feature_defs
# Check for missing values

feature_matrix.isnull().sum().sum()
# Split data

train_X = feature_matrix[:15120]

test_X = feature_matrix[15120:]
# Remove specific columns

#train_X.drop(['Soil_Type', 'Wilderness_Area'], axis = 1, inplace = True)

#test_X.drop(['Soil_Type', 'Wilderness_Area'], axis = 1, inplace = True)
train_X.tail()
test_X.head()
# Check of dataframe

test_X.iloc[:,:10].equals(X_test.iloc[:,:10])
from sklearn.multiclass import OneVsRestClassifier

#xgb_clf = OneVsRestClassifier(XGBClassifier(learning_rate =0.05, n_estimators=1000, n_jobs = -1))
# Meta Classifier

meta_cls = XGBClassifier(learning_rate =0.05, n_estimators=1000)
from sklearn.neighbors import KNeighborsClassifier
list_estimators = [RandomForestClassifier(n_estimators=500, max_features = 'sqrt',

                                random_state=1, n_jobs=-1), 

                   XGBClassifier(learning_rate =0.1, n_estimators=500, random_state=1, n_jobs=-1), 

                   LGBMClassifier(n_estimators=500,verbosity=0, random_state=1, n_jobs=-1),

                   KNeighborsClassifier(n_jobs = -1),

                  ExtraTreesClassifier(n_estimators=500, max_features = 'sqrt',random_state=1, n_jobs=-1)]

base_methods = ["RandomForestClassifier", "XGBClassifier", "LGBMClassifier", "KNeighborsClassifier", "ExtraTreesClassifier"]

#base_methods 
y.shape
from mlxtend.classifier import StackingCVClassifier



state = 1

stack = StackingCVClassifier(classifiers=list_estimators,

                             meta_classifier=meta_cls,

                             cv=3,

                             use_probas=True,

                             verbose=1, 

                             random_state=state,

                             n_jobs=-1)
stack.fit(train_X.values, y.values)
preds_test = stack.predict(test_X.values)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)