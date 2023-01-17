# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# pandas for handling our dataset 
# remove warnings
import warnings
warnings.filterwarnings('ignore')
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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read the data set using pandas .read_csv() method
#job_skills = pd.read_csv('job_skills.csv') #if you load from your local pc 
job_skills = pd.read_csv('../input/job_skills.csv')
# print the top 5 row from the dataframe
print(job_skills.shape)
job_skills.head()

# plotting function
def plotting(df):
    df.plot.barh(x=df.columns[0],y=df.columns[1],figsize=(25,20), legend=True,stacked=True )
    # add a suptitle
    plt.suptitle( df.columns[0] +" "+ df.columns[1]+" "+ "at Google Jobs", fontsize=25)
    
    plt.xlabel(df.columns[1], fontsize=25)
    plt.ylabel(df.columns[0],fontsize=25)
    # change xticks fontsize to 14
    plt.yticks(fontsize=20)
    plt.gca().invert_yaxis()
    # finally show the plot
    plt.show()
job_skills['Country_Location']=job_skills['Location'].str.split(',').str[-1]
country_value_counts = job_skills.Country_Location.value_counts()
country_value_counts=country_value_counts[country_value_counts.values>=10] #Values less than 10 are ignored for better visualisation
#location_value_counts = location_value_counts[::-1]
df_popular_country=pd.Series.to_frame(country_value_counts).reset_index()
df_popular_country.columns=['Country_Location', 'Preference']
df_popular_country.head(8)
plotting(df_popular_country)
job_skills['City_Location']=job_skills['Location'].str.split(',').str[0]
city_value_counts = job_skills.City_Location.value_counts()
city_value_counts=city_value_counts[city_value_counts.values>=10] #Values less than 10 are ignored for better visualisation
#location_value_counts = location_value_counts[::-1]
df_popular_city=pd.Series.to_frame(city_value_counts).reset_index()
df_popular_city.columns=['City', 'Preference']
df_popular_city.head(8)
plotting(df_popular_city)
# most popular language list 
programing_language_list = ['go','r', 'sas', 'matlab','stata','python', 'java','net', 'c++','html','css', 'php', \
                            'javascript', 'objective-c', 'ruby', 'perl','c','c#', 'sql','mysql','mapreduce','hadoop','kotlin']
min_qualifications = job_skills['Minimum Qualifications'].tolist()
min_qualifications_string = ''.join(map(str, min_qualifications)).lower()
# this portion of code is taken from https://www.kaggle.com/djcarlos/are-you-experienced-enough-to-work-at-google 
# find out which language occurs in most in minimum Qualifications string
skillcount = dict((keys,0) for keys in programing_language_list)
for w in re.findall(r"[\w'+#-]+|[.!?;’]", min_qualifications_string):
    if w in skillcount:
        skillcount[w] += 1

print(skillcount)
#Converting dictionary to Dataframe in ascending
df_popular_programming_lang = pd.DataFrame.from_dict(skillcount, orient='index').sort_values(by=0,ascending=False).reset_index()
df_popular_programming_lang.columns=['Programming_Language', 'Popularity'] #Assigning column
# Capitalize each programming language first letter
df_popular_programming_lang['Programming_Language'] = df_popular_programming_lang.Programming_Language.str.capitalize()
#df_popular_programming_lang = df_popular_programming_lang[::-1] 
df_popular_programming_lang.head(10)
plotting(df_popular_programming_lang)
degree_list = ["ba", "bs", "bachelor's", "phd",'mba','bachelor']
degree_count = dict((x,0) for x in degree_list)
for w in re.findall(r"[\w']+|[.,!?;’]", min_qualifications_string):
    if w in degree_count:
        degree_count[w] += 1
# print
print(degree_count)

#Converting dictionary to Dataframe in ascending
df_degree_popular = pd.DataFrame.from_dict(degree_count, orient='index').sort_values(by=0,ascending=False).reset_index()
df_degree_popular.columns=['Degree', 'Popularity'] #Assigning column name
# Capitalize each programming language first letter
df_degree_popular['Degree'] = df_degree_popular.Degree.str.upper() 
#df_degree_popular = df_degree_popular[::-1] 
df_degree_popular
plotting(df_degree_popular)
# this portion of code is taken from https://www.kaggle.com/djcarlos/are-you-experienced-enough-to-work-at-google 
years_exp = defaultdict(lambda: 0)

for w in re.findall(r'([0-9]+) year', min_qualifications_string):
     years_exp[w] += 1
        
print(years_exp)
#Converting dictionary to Dataframe in ascending
df_years_exp = pd.DataFrame.from_dict(years_exp, orient='index').sort_values(by=0,ascending=False).reset_index()
df_years_exp.columns=['Years of experience', 'Popularity'] #Assigning column name
#df_years_exp = df_years_exp[::-1] 
df_years_exp.head(10)

plotting(df_years_exp)
category_value_counts = job_skills.Category.value_counts()
#category_value_counts = category_value_counts[::-1]
df_popular_category=pd.Series.to_frame(category_value_counts).reset_index()
df_popular_category.columns=['Popular Category', 'Preference']
df_popular_category.head(10)
plotting(df_popular_category)
job_skills['Experience'] = job_skills['Minimum Qualifications'].str.extract(r'([0-9]+) year')
dff = job_skills[['Experience','Category']]
dff = dff.dropna()
dff.Experience.value_counts().iloc[:10].sort_values()

plt.figure(figsize=(20,25))
plt.title('Experiences needed in different job category', fontsize=24)
sns.countplot(y='Category', hue='Experience', data=dff, hue_order=dff.Experience.value_counts().iloc[:5].index.sort_values())
plt.yticks(fontsize=18)
plt.xlabel('Popularity', fontsize=24)
plt.show()
job_skills.Country_Location.value_counts().iloc[:6].index

plt.figure(figsize=(10,25))
plt.title('Google job categories popularity in different Countries', fontsize=24)
sns.countplot(y='Category', hue='Country_Location', data=job_skills, hue_order=job_skills.Country_Location.value_counts().iloc[:6].index)
plt.yticks(fontsize=18)
plt.show()