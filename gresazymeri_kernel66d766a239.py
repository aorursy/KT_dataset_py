netflix_titles_data.head()
movies =netflix_titles_data.loc[netflix_titles_data.type == 'Movie']['type']
rate_movies = len(movies)

print("Number of movies :", rate_movies)

shows= netflix_titles_data.loc[netflix_titles_data.type== 'TV Show']['type']
rate_shows =len(shows)

print("Number of shows :", rate_shows)
import pandas as pd
netflix_titles_data=pd.read_csv("../input/netflix-shows/netflix_titles.csv")

netflix_titles_data.columns
netflix_titles_data.describe()
netflix_titles_features=["show_id"]
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = '../input/netflix-shows/netflix_titles.csv'

netflix_titles = pd.read_csv(iowa_file_path)

X =netflix_titles_data[netflix_titles_features]
y =netflix_titles_data.show_id
iowa_model = DecisionTreeRegressor()
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those movies:", y.head().tolist())

from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")




X.head()
print("Making predictions for the following 5 id :")
print(X.head())


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))