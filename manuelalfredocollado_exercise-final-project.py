import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
my_filepath = "../input/avatar-the-last-air-bender/avatar_data.csv"



my_data = pd.read_csv(my_filepath, index_col="true_chapt")



my_data.head()
# Create a plot

plt.figure(figsize=(17,6))

num_data = my_data[['imdb_rating','series_rating']]

sns.lineplot(data=num_data)



sns.barplot(x=num_data.index, y=num_data['imdb_rating'])
plt.figure(figsize=(25,20))

sns.heatmap(data=num_data, annot=True)
plt.figure(figsize=(17,7))

sns.regplot(x=my_data.index, y=my_data['imdb_rating'])
plt.figure(figsize=(17,7))

sns.swarmplot(x=my_data['book'],

              y=my_data['imdb_rating'])
sns.distplot(a=my_data['imdb_rating'], kde=False)

sns.kdeplot(data=my_data['imdb_rating'], shade=True)

# Read the files into variables 

my_data_water = my_data.loc[my_data['book'].isin(['Water'])]

my_data_earth = my_data.loc[my_data['book'].isin(['Earth'])]

my_data_fire = my_data.loc[my_data['book'].isin(['Fire'])]



sns.distplot(a=my_data_water['imdb_rating'], label="Water", kde=False)

sns.distplot(a=my_data_earth['imdb_rating'], label="Earth", kde=False, color="k")

sns.distplot(a=my_data_fire['imdb_rating'], label="Fire", kde=False, color="r")

plt.legend()

sns.kdeplot(data=my_data_water['imdb_rating'],shade=True, label="Water")

sns.kdeplot(data=my_data_earth['imdb_rating'],shade=True, label="Earth", color="k")

sns.kdeplot(data=my_data_fire['imdb_rating'],shade=True, label="Fire", color="r")

plt.legend()