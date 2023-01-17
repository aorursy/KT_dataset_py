import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

my_filepath_personality = "../input/top-personality-dataset/2018-personality-data.csv"



# Read the file into a variable my_data

my_data_personality  = pd.read_csv(my_filepath_personality)



# Path of the file to read

my_filepath_ratings = "../input/top-personality-dataset/2018_ratings.csv"



# Read the file into a variable my_data

my_data_ratings  = pd.read_csv(my_filepath_ratings)







# Line chart 

#plt.figure(figsize=(12,6))

#sns.lineplot(data=my_data_personality)
#View smaple data

my_data_personality.head()
#View smaple data

my_data_ratings.head()