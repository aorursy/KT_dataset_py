import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
attractions_gor = pd.read_csv("../input/tripadvisor-attractions-reviews-nearby-locations/attractions_gor.csv")

attractions_scc = pd.read_csv("../input/tripadvisor-attractions-reviews-nearby-locations/attractions_scc.csv")
attractions_gor.head()
attractions_gor.tail()
attractions_gor.info()
attractions_gor.shape
attractions_scc.head()
attractions_scc.tail()
attractions_scc.info()
attractions_scc.shape
attractions_gor_most_visited = attractions_gor["region_name"].value_counts().sort_values()[-1::-1]

for name,i in zip(attractions_gor_most_visited.index,range(len(attractions_gor_most_visited))):

    attractions_gor_most_visited = attractions_gor_most_visited.rename(index= {attractions_gor_most_visited.index[i] : name.strip("Attractions")})

attractions_gor_most_visited
attractions_scc_most_visited = attractions_scc["region_name"].value_counts().sort_values()[-1::-1]

attractions_scc_most_visited
plt.figure(figsize = (15,8))

sns.barplot(attractions_gor_most_visited.index,attractions_gor_most_visited.values)

plt.xticks(rotation = 90,size = 15)

plt.yticks(size = 15)

plt.title("attractions_GOR_most_visited",size = 20)

plt.show()
plt.figure(figsize = (15,8))

sns.barplot(attractions_scc_most_visited.index,attractions_scc_most_visited.values)

plt.xticks(rotation = 90,size = 15)

plt.yticks(size = 15)

plt.title("attractions_scc_most_visited",size = 20)

plt.show()