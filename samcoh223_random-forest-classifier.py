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
import json

from sklearn.cluster import KMeans

from sklearn.model_selection import cross_val_score #how good model is 

from sklearn.model_selection import cross_val_predict

import sklearn.ensemble as skens

from sklearn.decomposition import PCA

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from IPython.display import Image

from sklearn.metrics import classification_report, confusion_matrix

#visualization imports 
train = pd.read_json('/kaggle/input/whats-cooking/train.json')

train.head()
test = pd.read_json('/kaggle/input/whats-cooking/test.json')

test.head()
with open('/kaggle/input/whats-cooking/train.json') as json_file:

    data = json.load(json_file)

    

with open('/kaggle/input/whats-cooking/test.json') as json_file:

    test_data = json.load(json_file)
#train data 

lst = []

for x in data: 

    id_=x["id"]

    ingredients = x['ingredients']

    for ingred in ingredients:

        lst.append(ingred)
#train data ingredients 

unique_ingredients=list(set(lst))
#test data

lst2 = []

for x in test_data: 

    id_=x["id"]

    ingredients = x['ingredients']

    for ingred in ingredients:

        lst2.append(ingred)

#test data ingredients 

unique_ingredients_test =list(set(lst2))
ingred_value = {ingredient: [] for ingredient in unique_ingredients}



ids = []

ciusine = [] 

ingredient_lst = unique_ingredients

ingredients_for_id =[] 

for x in data: 

    id_=x["id"]

    ingredients_in_id = x['ingredients']

    ids.append(id_)

    ciusine.append(x["cuisine"])

    for ingred in ingred_value: 

            if ingred in ingredients_in_id:

                ingred_value[ingred].append(1)

            else: 

                 ingred_value[ingred].append(0)



                
# create a key in the ingred_value dictionary called cuisine 

# The value is now a list of cuisines in the dataset 

# the list called ciusine was created above 



ingred_value["cuisine"] = ciusine
train_df = pd.DataFrame(ingred_value)
train_df.head()
ingred_value_test = {ingredient: [] for ingredient in unique_ingredients_test}

## for test data: 

ids = []

ingredient_lst = unique_ingredients_test

ingredients_for_id =[] 

for x in test_data: 

    id_=x["id"]

    ingredients_in_id = x['ingredients']

    ids.append(id_)

    for ingred in ingred_value_test: 

            if ingred in ingredients_in_id:

                ingred_value_test[ingred].append(1)

            else: 

                 ingred_value_test[ingred].append(0)



                
# create a key in the ingred_value_test dictionary called id

# The value is now a list of ids in the dataset 

# the list called ids was created above 



ingred_value_test["id"] = ids
test_df = pd.DataFrame(ingred_value_test)
test_df.head()
y = train_df["cuisine"]

X = train_df.drop("cuisine", axis = 1) #all variables 



# first do model then PCA if not 

#first do PCA 
X_test = test_df.drop("id", axis = 1)
rf_model = skens.RandomForestClassifier(n_estimators=10,oob_score=True, criterion='entropy')

rf_model.fit(X,y)
scores = cross_val_score(rf_model, X,y, cv=5) #do this 5 times and withhold info each tim 

#score is the average score for each of the cross validaiton 

#Get an accuracy score using the mean and standard deviation multiplied by 2

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# X[~X.isin(X_test.columns())]

columns_in_train = set(X.columns)

columns_in_test = set(X_test.columns)

columns_in_both = list(columns_in_train & columns_in_test)

new_df_with_columns_in_both_train = X[columns_in_both]

new_df_with_columns_in_both_test = X_test[columns_in_both]

rf_model_for_test = skens.RandomForestClassifier(n_estimators=10,

                                                oob_score=True, 

                                                criterion='entropy')

rf_model_for_test.fit(new_df_with_columns_in_both_train,y)
predicted_y_for_test = rf_model_for_test.predict(new_df_with_columns_in_both_test)
test_df_predicted = test_df.copy()



test_df_predicted["predicted_labels"] = predicted_y_for_test
test_df_predicted.head()
scores_of_new_model = cross_val_score(rf_model_for_test, new_df_with_columns_in_both_train,y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_of_new_model.mean(), scores_of_new_model.std() * 2))
picking_a_label_at_random = list(np.random.random_integers(0, 19, size = len(X)))
#get a list of the unique category labels 

labels = list(y.unique())
#create a dictionary that maps 0-19 to their label names 

map_labels = {i : labels[i] for i in range(len(labels)) }
y_labels_for_random=[map_labels[x] for x in picking_a_label_at_random]
accuracy = accuracy_score(y, y_labels_for_random)
accuracy