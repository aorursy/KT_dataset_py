# set up environment 

import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
# import data
penguins = pd.read_csv('/kaggle/input/palmer-penguins/penguins.csv')
# basic EDA
penguins.head()
# basic EDA
# we have NAs
penguins.describe()
# do penguins co-habitate on an island?
penguins.groupby(['island', 'species']).size().reset_index(name='counts')
# set up variables for decision tree

penguins_full = penguins.dropna()

# prediction target
y = penguins_full.body_mass_g

# feature selection
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
X = penguins_full[features]
# Define and fit model
penguin_model = DecisionTreeRegressor(random_state=406)

# Fit model
model_fit = penguin_model.fit(X, y)
# HOML export graphic of trained decision tree
from sklearn.tree import export_graphviz

export_graphviz(
  model_fit, 
  out_file='/kaggle/working/penguins_dt.dot',
  feature_names=features,
  rounded=True,
  filled=True)
# convert dot file to png
!dot -Tpng penguins_dt.dot -o penguins_dt.png
# show full decision tree
from IPython.display import Image
Image(filename = 'penguins_dt.png')