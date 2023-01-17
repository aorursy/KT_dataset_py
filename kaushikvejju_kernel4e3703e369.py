

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#PART I: Specifying File Path
netflix_filepath = "../input/netflix-shows/netflix_titles.csv"

#PART II: Assign a variable to the content of the file
netflix_data =  pd.read_csv(netflix_filepath, index_col="show_id")

#PART III: Print specific components of the dataset, such as the columns and the first five rows
netflix_data.columns #Columns
netflix_data.head() #First 5 rows
#PART IV: Develop visualizations based on the contents of the data set:
#Visualization I: Categorical Scatter Plot --> Takes two categories: TV Shows and Movies and displays their frequency in a certain year
plt.figure(figsize=(14,6))
plt.title("Movies and TV Shows Released on Netflix")
sns.swarmplot(x= netflix_data['type'],
              y=netflix_data['release_year'])
#Visualization II: Histogram --> Displays the Frequency of Age Ratings of Netflix Entries
plt.figure(figsize=(14, 6))
plt.title("Age Ratings on Netflix")
sns.set(style="darkgrid")
ax = sns.countplot(x='rating',  data= netflix_data)
plt.show()
