import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Pokedex=pd.read_csv('/kaggle/input/pokemon/Pokemon.csv');


Pokedex.columns
y=Pokedex.HP
Pokedex_features=['Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
X=Pokedex[Pokedex_features]
X.describe()
X.head()
pokemon_model=DecisionTreeRegressor(random_state=1)
#fit the model
pokemon_model.fit(X,y)
print("Making HP predictions for the following 5 Pokemon:")
print(X.head())
print("The predictions are")
print(pokemon_model.predict(X.head()))
predicted_HP_stats=pokemon_model.predict(X)
mean_absolute_error(y, predicted_HP_stats)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

pokemon_model=DecisionTreeRegressor()

pokemon_model.fit(train_X, train_y)

val_predictions=pokemon_model.predict(val_X)

print(mean_absolute_error(val_y,val_predictions))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
forest_pokemon_model= RandomForestRegressor(random_state=1)
forest_pokemon_model.fit(train_X, train_y)
melb_preds=forest_pokemon_model.predict(val_X)
print(mean_absolute_error(val_y,melb_preds))