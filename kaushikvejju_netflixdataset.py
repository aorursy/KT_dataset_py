
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
netflix_filepath = "../input/netflix-shows/netflix_titles.csv"
netflix_data =  pd.read_csv(netflix_filepath, index_col="show_id")
netflix_data.columns
netflix_data.head() 
plt.title("Movies and TV Shows Released on Netflix")
sns.swarmplot(x= netflix_data['type'],
              y=netflix_data['release_year'])

netflix_count = netflix_data['release_year'].value_counts()
netflix_count = pd.DataFrame(netflix_count).reset_index()
netflix_count.columns = ['release_year','Number of Entries',]
netflix_count

ax = sns.lineplot(x="release_year", y="Number of Entries",
                  data= netflix_count)
plt.figure(figsize=(14,6))
plt.title("Age Ratings Movies and TV Shows Released on Netflix")

ax = sns.countplot(data=netflix_data,y= "rating")

netflix_count_2 = netflix_data['rating'].value_counts()
netflix_count_2 = pd.DataFrame(netflix_count_2).reset_index()
netflix_count_2.columns = ['rating','count']
netflix_count_2
rating_data = netflix_count_2["rating"]
count_data = netflix_count_2["count"]
colors = ["b", "r", "y", "m", "c","g", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "powderblue", "navy", "darkcyan"]
explode = (0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005) 
plt.figure(figsize=(24,9.5))
plt.pie(count_data, labels=rating_data, explode=explode, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=140)
plt.show()
netflix_specific = netflix_data[netflix_data['rating']== 'NC-17']
num = netflix_specific.shape[0]
print ("There are " + str(num) + " movies/tv shows that are rated NC-17 on Netlfix. They are displayed below")
netflix_specific 
netflix_specific_2 = netflix_data[netflix_data['rating']== 'UR']
num2 = netflix_specific_2.shape[0]
print ("There are " + str(num2) + " movies/tv shows that are unrated on Netlfix. They are displayed below")
netflix_specific_2
num_total = netflix_data.shape[0]
percent = 100 * (9/ (num_total))
print (str(percent) + "% of all the content in Netflix is listed in the NC-17 and Unrated categories. That is 9 out of " + str(num_total) + " entries.")