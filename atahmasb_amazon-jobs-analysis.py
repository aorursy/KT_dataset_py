# pandas for handling our dataset
import pandas as pd
# numpy for numeric operations
import numpy as np
from collections import defaultdict

# matplotlib for plotting
import matplotlib.pyplot as plt
# use ggplot style
plt.style.use('ggplot')
# seaborn for beautiful visualizations
import seaborn as sns
# regualar expression
import re
# print inline in this notebook
%matplotlib inline

# read the data set using pandas .read_csv() method
df_job_description = pd.read_csv('../input/amazon_jobs_dataset.csv')
# print the top 5 row from the dataframe
df_job_description.head()
#list of programming langauages and technologies
languages_list = ['swift','matlab','mongodb','hadoop','cosmos', 'mysql','spark', 'pig', 'python', 'java', 'c++', 'php', 'javascript', 'objectivec', 'ruby', 'perl','c','c#']
# get our BASIC QUALIFICATIONS and PREFERRED QUALIFICATIONS columns and and convert all of the values to a list
qualifications = df_job_description['BASIC QUALIFICATIONS'].tolist()+df_job_description['PREFERRED QUALIFICATIONS'].tolist()
# joining the list to a single string and lower case the letter
qualifications_string = "".join(re.sub('[·,-/’()]', '', str(v)) for v in qualifications).lower()
wordcount = dict((x,0) for x in languages_list)
for w in re.findall(r"[[\w'+#-]+|[.!?;’]", qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
# print
print(wordcount)

# sort the dict
programming_language_popularity = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
# make a new dataframe from programming languages and their popularity
df_popular_programming_lang = pd.DataFrame(programming_language_popularity,columns=['Language','Popularity'])
# Capitalize each programming language first letter
df_popular_programming_lang['Language'] = df_popular_programming_lang.Language.str.capitalize()
df_popular_programming_lang = df_popular_programming_lang[::-1]
df_popular_programming_lang
# plot
df_popular_programming_lang.plot.bar(x='Language',y='Popularity',figsize=(15,15), legend=False)
# add a suptitle
plt.suptitle("Programming Languages popularity at Amazon Jobs", fontsize=20)
# set xlabel to ""
plt.xlabel("")
# change xticks fontsize to 14
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# finally show the plot
plt.show()
# get our BASIC QUALIFICATIONS and PREFERRED QUALIFICATIONS columns and and convert all of the values to a string

basic_qualifications_string = "".join(re.sub('[·,-/’()]', '', str(v)) for v in qualifications)

degree_list = ["BA", "BS", "Bachelor's", "PhD","MS","Master's"]
wordcount = dict((x,0) for x in degree_list)
for w in re.findall(r"[\w']+|[.,!?;’]", basic_qualifications_string):
    if w in wordcount:
        wordcount[w] += 1
# print
print(wordcount)
degree_wanted = sorted(wordcount.items(), key=lambda kv: kv[1], reverse=True)
df_degree_popular = pd.DataFrame(degree_wanted,columns=['Degree','Popularity'])

df_degree_popular
num=list([('Bachelor of Science',1471),('Master of Science',850),('PhD',489),('BA',2)])
df_degree_popular = pd.DataFrame(num,columns=['Degree','Popularity'])
df_degree_popular
df_degree_popular = df_degree_popular[::-1] 
# plot
df_degree_popular.plot.bar(x='Degree',y='Popularity',figsize=(15,10), stacked=True)
# add a suptitle
plt.suptitle("Popularity of academic degree at Amazon Jobs ", fontsize=18)
# set xlabel to ""
plt.xlabel("")

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# finally show the plot
plt.show()
years = defaultdict(lambda: 0)

for w in re.findall(r'([0-9]) ', basic_qualifications_string):
     years[w] += 1
        
print(years)
years = sorted(years.items(), key=lambda kv: kv[1], reverse=True)

df_years = pd.DataFrame(years,columns=['Years of experience','Popularity'])
df_years = df_years[::-1] 
df_years.plot.bar(x='Years of experience',y='Popularity',figsize=(10, 8), legend=False,stacked=True)
# add a suptitle
plt.title("Years of experiences needed for Amazon Jobs", fontsize=18)
# set xlabel to ""
plt.xlabel("Popularity", fontsize=14)
plt.ylabel("Years of experiences",fontsize=18)
# change xticks fontsize to 14
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# finally show the plot
plt.show()
# where is most job located
threshold = 10
location_value_counts = df_job_description.location.value_counts()
to_remove = location_value_counts[location_value_counts <= threshold].index
df_job_description['location'].replace(to_remove, np.nan, inplace=True)
location_value_counts = df_job_description.location.value_counts()
location_value_counts = location_value_counts[::-1]
location_value_counts.plot.barh(figsize=(15, 15))
# add a suptitle
plt.title("Amazon Jobs Location", fontsize=24)
# set xlabel to ""
plt.xlabel("Popularity", fontsize=20)
plt.ylabel("Location",fontsize=20)
# change xticks fontsize to 14
plt.xticks(fontsize=24)
plt.yticks(fontsize=20)
# finally show the plot
plt.show()