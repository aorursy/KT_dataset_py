import numpy as np 
import pandas as pd 
import matplotlib

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline 
test = pd.read_csv("../input/test.csv", index_col="row_id")
test.head()
train = pd.read_csv("../input/train.csv", index_col="row_id")
train.head()
print(train.shape) 
print(test.shape)
train_descriptive_stats = train.describe()
train_descriptive_stats
test_descriptive_stats = test.describe()
test_descriptive_stats
train.corr()
test.corr()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train[train.columns.difference(["place_id"])], train["place_id"], test_size=0.33, random_state=42)
print(X_train.shape)
train.tail(100000).plot.hexbin(x='x', y='y');
places_vs_datarows_distribution = train["place_id"].value_counts()
places = places_vs_datarows_distribution.index 
places_vs_datarows_distribution = places_vs_datarows_distribution.to_frame("nbr_of_rows") 
places_vs_datarows_distribution["place_id"] = places
del places 
places_vs_datarows_distribution.head()
len(pd.unique(train["place_id"].values.ravel()))
less_than_100_place_id = places_vs_datarows_distribution[places_vs_datarows_distribution["nbr_of_rows"] <= 100]
len(pd.unique(less_than_100_place_id["place_id"].values.ravel()))
less_than_10_place_id = places_vs_datarows_distribution[places_vs_datarows_distribution["nbr_of_rows"] <= 10]
len(pd.unique(less_than_10_place_id["place_id"].values.ravel()))
places_vs_datarows_distribution.hist(column="nbr_of_rows",bins=100,figsize=(10,10), xlabelsize=15, ylabelsize=15)

from scipy import stats
no_outliers = train[(np.abs(stats.zscore(train)) < 3).all(axis=1)]
print(train.shape)
print(no_outliers.shape)
train_descriptive_stats["time"]
test_descriptive_stats["time"]
#assuming time is in minutes  
#how long does it span over train set? 
train_nbr_of_minutes = (train_descriptive_stats["time"].loc["max"]-train_descriptive_stats["time"].loc["min"]) 
print("train hours {0}".format(train_nbr_of_minutes/(60)))
print("train days {0}".format(train_nbr_of_minutes/(60*24)))
#how long does time span in test data
test_nbr_of_minutes = (test_descriptive_stats["time"].loc["max"]-test_descriptive_stats["time"].loc["min"])
print("test hours {0}".format(test_nbr_of_minutes/(60)))
print("test days {0}".format(test_nbr_of_minutes/(60*24)))
#split the to grids of 2kmx2km
from itertools import product 
def split_to_grids(grid_size, data):
    analysis_per_grid = pd.DataFrame()
    grid_id = []
    grid_places_count = []
    for i in product(range(grid_size,11,grid_size),repeat=2):
        grid_id.append(i)
        dt = data[(data["x"]>=i[0]-grid_size)&(data["x"]<i[0])&(data["y"]>=i[1]-grid_size)&(data["y"]<i[1])]
        count = dt["place_id"].count()
        grid_places_count.append(count) 
        print("nbr of places {0}".format(count))
    analysis_per_grid["grid_id"] = grid_id
    analysis_per_grid["places_count"] = grid_places_count
    return analysis_per_grid
analysis_per_grid = split_to_grids(2,train)
analysis_per_grid
