# I'll use pandas



import pandas as pd

import numpy as np
# It's time to read the dataset



dataset_path = "../input/bangla-news-datasets-from-bdpratidin/1000DaysNews.csv"

dataframe = pd.read_csv(dataset_path)



print(f"There are {len(dataframe)} articles in the dataset.")
dataframe.head()
print(f"There are news from {dataframe['date'][0]} to {dataframe['date'][len(dataframe)-1]} in the dataset.")
print(f"There are news from {dataframe['category'].nunique()} categories in the dataset.")
dataframe['category'].value_counts()
dataframe['category'].value_counts().plot(kind='bar')