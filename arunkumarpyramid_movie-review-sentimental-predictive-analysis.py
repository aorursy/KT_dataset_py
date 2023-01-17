import pandas as pd

dataset=pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

dataset.head()
dataset.shape
dataset.info()
dataset.isna().sum()
# This logic will help to find missing ratio.

missing_ratio=(dataset.isna().sum()/len(dataset))*100.0



# To know all missing value ratio for each attribute....

print(missing_ratio.sort_values(ascending=False))
# Visualising the missing ration in barplot



import seaborn as sb

import matplotlib.pyplot as plt



sb.barplot(x=missing_ratio.index,y=missing_ratio)

plt.title("Missing Value ratio in %")

plt.xlabel('Missing value attributes')

plt.ylabel('Missing value in percentage')

plt.show()
dataset['sentiment'].value_counts()