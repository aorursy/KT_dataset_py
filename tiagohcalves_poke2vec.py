# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import ast
import matplotlib
import numpy as np
import pandas as pd

from IPython.display import display
from scipy.spatial import distance
from sklearn.manifold import TSNE

%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
data = pd.read_csv("../input/pokemon.csv")
print("Shape: %d x %d" % data.shape)
print("Numeric Attributes")
for i in range(int(data.shape[1] / 10) + 1):
    n_cols = min(data.shape[1], (i+1)*10)
    display(data.iloc[:, 10*i:n_cols].describe(include=np.number))

print("Categorical Attributes")
display(data.describe(include=np.object))
# One hot encode abilities
abilities_list = []
for ab in data.abilities.values:
    # The list is as a string, but in Python syntax, so we can cast it to a proper type
    abilities = ast.literal_eval(ab)
    abilities_list.append(abilities)
    
ability_df = pd.DataFrame(abilities_list)
ability_df = ability_df.rename(columns=lambda x: "ability_" + str(x))
ability_df.head()
# Let's encode only the three first abilities
ability_oh_df = pd.get_dummies(ability_df.iloc[:, :3])
ability_oh_df.shape
# Join abilities OH data frame with the main frame
feat_data = data.drop("abilities", axis=1)
feat_data = feat_data.join(ability_oh_df)
feat_data.head()
# Cast capture date to int
cr_list = []
for cr in data["capture_rate"]:
    try:
        icr = int(cr)
        cr_list.append(icr)
    except:
        cr_list.append(0)
        
feat_data["capture_rate_int"] = cr_list
feat_data["capture_rate_int"].head()
# One-hot encode types
type_df = pd.get_dummies(data[["type1", "type2"]])

feat_data = feat_data.join(type_df)

feat_data.drop("type1", axis=1)
feat_data.drop("type2", axis=1)
feat_data.head()
# Drop irrelevant columns
feat_data = feat_data.drop("capture_rate", axis=1)
feat_data = feat_data.drop("classfication", axis=1)
feat_data = feat_data.drop("japanese_name", axis=1)
feat_data = feat_data.drop("name", axis=1)
feat_data = feat_data.drop("type1", axis=1)
feat_data = feat_data.drop("type2", axis=1)
feat_data.shape
# Remove NaN
poke_matrix = feat_data.as_matrix()
poke_matrix = np.nan_to_num(poke_matrix)
poke_matrix.shape
# Let's create a index map for the pokemon
poke_names = data["name"].as_matrix()
poke_map = {name:idx for (idx, name) in enumerate(poke_names)}
print(poke_names[:25])
# The distance function

# Calculate the cosine distance for all pokemon
poke_distance_matrix = np.nan_to_num(distance.cdist(poke_matrix, poke_matrix))

def print_top_close(name, n=15):
    # Find the smaller distances
    dist_array = poke_distance_matrix[[poke_map[name]], :]
    top_k_closest = dist_array.argsort(axis=1)[:, 1:n]
    print(poke_names[top_k_closest])
print_top_close("Articuno")
print_top_close("Charizard")
print_top_close("Pikachu")
print_top_close("Tyranitar")
# Cast the features to 2 dimensions
poke_embedded = TSNE(n_components=2).fit_transform(poke_matrix)

x = poke_embedded[:, 0]
y = poke_embedded[:, 1] 
n = poke_embedded.shape[0]
scatter_data = [go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    text = poke_names
)]

py.iplot(scatter_data, filename='poke-scatter')
def poke_vec(name):
    return poke_matrix[[poke_map[name]], :]
def print_top_close_vec(vec, n=15):
    dist = distance.cdist(poke_matrix, vec, metric='cosine')
    dist = np.nan_to_num(dist)

    top_k_dense = dist.argsort(axis=0)[:n, 0]
    print(poke_names[top_k_dense])
target_vec = poke_vec("Gengar") - poke_vec("Gastly") + poke_vec("Charmander")
print_top_close_vec(target_vec)
target_vec = poke_vec("Pupitar") - poke_vec("Larvitar") + poke_vec("Gible")
print_top_close_vec(target_vec)