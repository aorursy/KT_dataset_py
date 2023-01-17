# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Search_df = pd.read_csv("/kaggle/input/citysearchstatedataset/City_Search_Data/Total_Search_df.csv")
Search_df = Search_df.drop(['index'], axis=1)
Search_df.head()
Search_df.columns
def score_calculator(Value, weight):
    score = np.dot(Value,weight)
    return score

def matrix_ranker(matrix):
    matrix= np.argsort(matrix, axis=0)
    print(matrix.shape)
    return matrix
Powerty_weights = [1,2]
Poverty_matrix = (Search_df[['Poverty','ChildPoverty']]).to_numpy()
Poverty_scores = score_calculator(Poverty_matrix, Powerty_weights)
print(Poverty_scores.shape)
US_Values = [18.5,60.6,13.4,1.3,5.9,0.3]
Diversity_matrix = (Search_df[['Hispanic','White','Black','Native','Asian','Pacific']]).to_numpy()
Diversity_scores = score_calculator(Diversity_matrix, US_Values)
Diversity_scores.shape
Employment_weight = [60.8, 39.2]
Employment_matrix = (Search_df[['Employed','Unemployment']]).to_numpy()
Employment_scores = score_calculator(Employment_matrix, Employment_weight)
Employment_scores.shape
Industry_matrix= (Search_df[['Professional','Service', 'Office', 'Construction','Production']]).to_numpy()
Industry_ranked = matrix_ranker(Industry_matrix)
print(Industry_ranked)
Transportation_matrix= (Search_df[['Drive','Carpool', 'Transit', 'Walk','OtherTransp', 'WorkAtHome', 'MeanCommute']]).to_numpy()
Transportation_ranked = matrix_ranker(Transportation_matrix)
print(Transportation_ranked)
Employment_ratio_matrix= (Search_df[['PrivateWork','PublicWork', 'SelfEmployed', 'FamilyWork']]).to_numpy()
Employment_ratio_ranked = matrix_ranker(Employment_ratio_matrix)
print(Employment_ratio_ranked)
Income_per_capita_matrix= (Search_df[['Mean','Income', 'Stdev', 'Income']]).to_numpy()
Income_per_capita_ranked = matrix_ranker(Income_per_capita_matrix)
print(Income_per_capita_ranked)
Columns_names = ['City','State', 'population','Poverty','Diversity','Employment',
                 'Professional','Service', 'Office', 'Construction','Production',
                 'Drive','Carpool', 'Transit', 'Walk','OtherTransp', 'WorkAtHome', 'MeanCommute',
                 'PrivateWork','PublicWork', 'SelfEmployed','FamilyWork', 
                 'Mean','Income', 'Stdev', 'Income']
Total_matrix = np.concatenate((((Search_df['City'].values).reshape(-1,1)), 
                               (Search_df['State'].values).reshape(-1,1)),axis=-1)
Total_matrix = np.concatenate((Total_matrix, (Search_df['population'].values).reshape(-1,1)), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Poverty_scores.reshape(-1,1)), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Diversity_scores.reshape(-1,1)), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Employment_scores.reshape(-1,1)), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Industry_ranked), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Transportation_ranked), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Employment_ratio_ranked), axis=-1)
Total_matrix = np.concatenate((Total_matrix, Income_per_capita_ranked), axis=-1)
Recommendation_df = pd.DataFrame(data=Total_matrix, columns=Columns_names)
print(final_df.shape)
Recommendation_df.head(5)
Recommendation_df.to_csv("Recommendation_df.csv",index=False )
Recommendation_df.head()
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

Numarical_data = Recommendation_df.drop(['City','State'], axis = 1)
scaler = StandardScaler()
Scaled_Traing_data = scaler.fit_transform(Numarical_data.to_numpy())
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}
dbscan = DBSCAN(eps=0.3)
sse = []
n = 50
for k in range(1, n):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(Scaled_Traing_data)
    sse.append(kmeans.inertia_)
plt.style.use("seaborn")
plt.plot(range(1, n), sse)
plt.xticks(range(1, n))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()