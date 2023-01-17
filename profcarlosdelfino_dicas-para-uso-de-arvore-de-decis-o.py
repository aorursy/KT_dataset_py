# Code you have previously used to load data

import sys

import os

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

#  Say, all airline-safety files...

import zipfile

dataset_name = "sberbank-russian-housing-market"

working_train_file = "./train.csv"



import shutil

#shutil.rmtree('/kaggle/working')



#shutil.rmtree('/kaggle/working/pdf')



#os.remove('/kaggle/working/Model*')

#os.remove('/kaggle/working/train.csv')



# Will unzip the files so that you can see them..

with zipfile.ZipFile("../input/"+dataset_name+"/train.csv.zip","r") as z:

    z.extractall(".")



shutil.rmtree('./__MACOSX')

    

print("unziped train.csv.zip")
# Path of the file to read

train_data = pd.read_csv(working_train_file,index_col='id')

# Clean na data

train_data.dropna(inplace=True)



train_data.describe()
# Create X, seguindo a dica 1, escolhendo alguns *features* para evitar *overfit*

"""

price_doc: sale price (this is the target variable)

id: transaction id

timestamp: date of transaction

full_sq: total area in square meters, including loggias, balconies and other non-residential areas

life_sq: living area in square meters, excluding loggias, balconies and other non-residential areas

floor: for apartments, floor of the building

max_floor: number of floors in the building

material: wall material

build_year: year built

num_room: number of living rooms

kitch_sq: kitchen area

state: apartment condition

product_type: owner-occupier purchase or investment

sub_area: name of the district

"""

# features = ['num_room', 'max_floor', 'full_sq', 'life_sq', 'floor', 'material', 'build_year', 'kitch_sq']

# features = ['num_room', 'max_floor', 'kitch_sq', 'full_sq', 'life_sq', 'floor', 'material']

#

# Maxdepth: 05, Random State: 01, Validation MAE: 2,563,367, 

#    Features: ['num_room', 'max_floor', 'kitch_sq', 'full_sq', 'life_sq', 'floor', 'material', 'sub_area', 'product_type']

#

# Maxdepth: 05, Random State: 01, Validation MAE: 2,527,532, 

#    Features: ['num_room', 'max_floor', 'kitch_sq', 'full_sq', 'life_sq', 'floor', 'material']



features = ['num_room', 'max_floor', 'kitch_sq', 

            'full_sq', 'life_sq', 'floor', 

            'material']

targets = ['price_doc']



# Create target object and call it y

y = train_data[targets]

X = train_data[features]

X.head()
y.describe()
X.describe()
## Add these lines to turn off the warnings

import warnings

warnings.filterwarnings("ignore")



from sklearn.preprocessing import LabelEncoder



if 'sub_area' in X:

    sub_area_encoder = LabelEncoder()

    sub_area_encoder.fit(X['sub_area'].astype(str))

    X['sub_area'] = sub_area_encoder.transform(X['sub_area'].astype(str))

if 'product_type' in X:

    product_type_encoder = LabelEncoder()

    product_type_encoder.fit(X['product_type'].astype(str))

    X['product_type'] = product_type_encoder.transform(X['product_type'].astype(str))

X.describe()
def testDecisionTree(train_X, train_y, val_X, val_y, max_depth=3,random_state=1):

    train_model = DecisionTreeRegressor(max_depth=max_depth,random_state=random_state)

    # Fit Model

    train_model.fit(train_X, train_y)



    # Make validation predictions and calculate mean absolute error

    val_predictions = train_model.predict(val_X)

    val_mae = mean_absolute_error(val_predictions, val_y)

    #print("Maxdepth: {depth:2d}, Random State: {random:2d}, Validation MAE: {mae:,.0f}"

    #      .format(depth=max_depth, random=random_state, mae=val_mae))

    return train_model, val_predictions, val_mae
import graphviz 

from sklearn.datasets import load_iris

from sklearn import tree 

def showDecisionTree(model,features,targets, file_name='DecisionTree'):

    dot_data = tree.export_graphviz(model, out_file=None, 

                                        feature_names=features,  

                                        class_names=targets,  

                                        filled=True, rounded=True,  

                                        special_characters=True)  

    graph = graphviz.Source(dot_data)  

    #graph 

    #dot_data = tree.export_graphviz(model, out_file=None) 

    #graph = graphviz.Source(dot_data)  

    graph.render("pdf/"+file_name)

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



min_random_state = 0

max_random_state = 5

min_depth = 3

max_depth = len(features)* 2



#shutil.rmtree('/kaggle/working/pdf')



# Conforme a dica 3, usando inicialmente uma arvore raza para ajustar conforme as decisões.

# Iniciado com max_depth=3

# Specify Model

best_train_model = None

best_train_depth = 3

best_train_state = 1

best_train_mae = sys.float_info.max

for depth in range(min_depth,max_depth+1):

    for state in range(min_random_state, max_random_state+1):

        train_model, val_predictions, val_mae = testDecisionTree(train_X, train_y, val_X, val_y, max_depth=depth, random_state=state)

        if best_train_mae > val_mae:

            best_train_mae = val_mae

            best_train_depth = depth

            best_train_state = state

            best_train_model = train_model

        print("Maxdepth: {depth:2d}, Random State: {random:2d}, Validation MAE: {mae:,.0f}, Best MAE: {bmae:,.0f}"

            .format(depth=depth, random=state, mae=val_mae, bmae=best_train_mae))
showDecisionTree(train_model,features,targets,file_name="ModelDecisionTree_{depth:02d}_{random:02d}_{mae:0f}"

              .format(depth=best_train_depth, random=best_train_state, mae=best_train_mae))

print("Maxdepth: {depth:02d}, Random State: {random:02d}, Validation MAE: {mae:,.0f}, Features: {f}"

              .format(depth=best_train_depth, random=best_train_state, mae=best_train_mae, f=features))