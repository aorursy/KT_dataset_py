import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
%matplotlib inline
salary = pd.read_csv("../input/nba_2017_salary.csv");
br_2017 = pd.read_csv("../input/nba_2017_br.csv");
players =pd.read_csv("../input/nba_2017_players_stats_combined.csv");
print(br_2017.columns.values)
br_2017.head(5)
corr = br_2017.corr()
a = plt.subplots(figsize=(15,9))
a = sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.lmplot(x="Age", y="PS/G", data=br_2017)
sns.lmplot(x="AST", y="PS/G", data=br_2017)
selected_player = br_2017[br_2017["Player"] == "LeBron James"].iloc[0]
distance_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
import math
def euclidean_distance(row):
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)
lebron_distance = br_2017.apply(euclidean_distance, axis=1)
print("Each player's Euclidean Distance with LeBron James:")
print(lebron_distance)
br_2017_numeric = br_2017[distance_columns]
br_2017_normalized = (br_2017_numeric - br_2017_numeric.mean()) / br_2017_numeric.std()
br_2017_normalized.fillna(0, inplace=True)
from scipy.spatial import distance
lebron_normalized = br_2017_normalized[br_2017["Player"] == "LeBron James"].iloc[0]
euclidean_distances = br_2017_normalized.apply(lambda row: distance.euclidean(row, lebron_normalized), axis=1)
distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_lebron = br_2017.loc[int(second_smallest)]["Player"]
import random
from numpy.random import permutation
br_2017.fillna(0, inplace=True)
br_2017.head(5)
random_indices = permutation(br_2017.index)
random_indices = permutation(br_2017.index)
test_cutoff = math.floor(len(br_2017)/3)
test_data = br_2017.loc[random_indices[1:test_cutoff]]
train_data = br_2017.loc[random_indices[test_cutoff:]]
x_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PS/G']
y_column = ["PS/G"]
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(train_data[x_columns], train_data[y_column])
predictions = knn.predict(test_data[x_columns])
print(predictions)
actual = test_data[y_column]
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
print("***** Model Assessment *****")
print("MSE:",mean_squared_error(actual,predictions))
print("MAE:",mean_absolute_error(actual,predictions))
print("R_Square:",r2_score(actual,predictions))