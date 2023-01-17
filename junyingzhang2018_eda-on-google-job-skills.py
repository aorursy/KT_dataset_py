import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import scipy.stats as stats
import re
# Input data files are available in the "../input/" directory.
# os.listdir() Return a list containing the names of the files in the directory.
import os
print(os.listdir('../input'))

skills=pd.read_csv('../input/job_skills.csv')
# Use .head() and .tail() to check the first and last five observations
skills.head()
skills.tail()
# Use.info() to check the number of observations, column types and NAN value
skills.info()
skills['Company'].value_counts()
top_10_category=skills['Category'].value_counts().head(10)
print(top_10_category)

# Use seaborn.barplot to visualize the top 10 Job Categories
sns.barplot(x=top_10_category.values, y=top_10_category.index, orient='h')
# Check the top 10 location
top_10_location=skills['Location'].value_counts().head(10)
print(top_10_location)
sns.barplot(x=top_10_location.values, y=top_10_location.index, orient='h')
# The top 3 Job locations are Mountain View which is the headquarter, Sunnyvalue, and Durbin.
# Check why Sunnyvale has a lot of job opening
job_sunnyvale=skills[skills['Location']=='Sunnyvale, CA, United States']
# it looks like many of the jobs at the Sunnyvale location are related to Google Cloud
print(job_sunnyvale.shape)
job_sunnyvale.head()
# Let's check how many jobs has Title or Responsibilities mentions Google Cloud for job_sunnyvale
cloudrelated=[]
job_sunnyvale.reset_index(inplace=True)
title=job_sunnyvale['Title'].str.lower()
res=job_sunnyvale['Responsibilities'].str.lower()

for i in range(len(title)):
        if re.findall('google cloud', title[i], flags=0): 
            cloudrelated.append(1)
        elif re.findall('google cloud', res[i], flags=0):
            cloudrelated.append(1)
        else:
            cloudrelated.append(0)
cloudrelated=pd.Series(cloudrelated)  

job_sunnyvale['cloudrelated']=cloudrelated
print(job_sunnyvale['cloudrelated'].value_counts())
print(job_sunnyvale['cloudrelated'].sum()/job_sunnyvale['cloudrelated'].count())
# As we can see 39% of the jobs at Google Sunnyvale location mentioned Google cloud in the Job Title or description
language_list = ['python', 'java', 'c++', 'php', 'javascript', 'objective-c', 'ruby', 'perl','c','c#', 
                            'sql','kotlin', 'swift', 'ios','fortran', 'go', 'haskell', 'html', 'r','sas','scala','stata']
language_dict_mini=dict((language, 0) for language in language_list)
language_dict_prefer=dict((language, 0) for language in language_list)

mini=skills['Minimum Qualifications'].str.lower().tolist()
prefer=skills['Preferred Qualifications'].str.lower().tolist()

mini_string=' '.join(str(word) for word in mini)
prefer_string=' '.join(str(word) for word in prefer)
for language in re.findall(r"[\w'+#-]+|[.!?;’]", mini_string):
    if language in language_list:
        language_dict_mini[language] += 1
language_mini=pd.Series(language_dict_mini)
language_mini.sort_values(ascending=False)
for language in re.findall(r"[\w'+#-]+|[.!?;’]", prefer_string):
    if language in language_list:
        language_dict_prefer[language] += 1
language_prefer=pd.Series(language_dict_prefer)
language_prefer.sort_values(ascending=False)
# Visualize the top 10 languages in the minimum qualifications and preferred qualifications
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.barplot(x=language_mini.sort_values(ascending=False).head(10).values, y=language_mini.sort_values(ascending=False).head(10).index, orient='h')
plt.title('Top 10 languages mentioned in Minimum Qualifications')
plt.xlabel('Count')
plt.ylabel('Programming Language')

plt.subplot(1,2,2)
sns.barplot(x=language_prefer.sort_values(ascending=False).head(10).values, y=language_prefer.sort_values(ascending=False).head(10).index, orient='h')
plt.title('Top 10 languages mentioned in Prefered Qualifications')
plt.xlabel('Count')
plt.ylabel('Programming Language')
degree_mini=dict((val, 0) for val in ['phd', 'master', 'bachelor', 'associate', 'no degree'])
degree_mini
# Create list for the degrees
phd=['phd', 'ph.d', 'doctor', 'm.d.', 'dds']
master=['master', 'ma', 'm.a.', 'ms', 'm.s.', 'mba', 'mfa']
bachelor=['bachelor', 'bs', 'b.s.', 'ba', 'b.a.', 'bfa', 'bas']
associate=['associate', 'a.a.', 'a.s.', 'aas', 'high school']

degree_mini=dict((val, 0) for val in ['phd', 'master', 'bachelor', 'associate'])
degree_prefer=dict((val, 0) for val in ['phd', 'master', 'bachelor', 'associate'])

mini=skills['Minimum Qualifications'].str.lower()
prefer=skills['Preferred Qualifications'].str.lower()


for i in range(len(mini)):
    for degree in re.findall(r"[\w'+#-]+|[.!?;’]", str(mini[i])):
        if degree in phd:
            degree_mini['phd']+=1
        elif degree in master:
            degree_mini['master']+=1
        elif degree in bachelor:
            degree_mini['bachelor']+=1
        elif degree in associate:
            degree_mini['associate']+=1
        
       
degree_mini=pd.Series(degree_mini)  
degree_mini
for i in range(len(prefer)):
    for degree in re.findall(r"[\w'+#-]+|[.!?;’]", str(prefer[i])):
        if degree in phd:
            degree_prefer['phd']+=1
        elif degree in master:
            degree_prefer['master']+=1
        elif degree in bachelor:
            degree_prefer['bachelor']+=1
        elif degree in associate:
            degree_prefer['associate']+=1
        
       
degree_prefer=pd.Series(degree_prefer)  
degree_prefer
# Visualize the degree in the minimum qualifications and preferred qualifications
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.barplot(x=degree_mini.sort_values(ascending=False).values, y=degree_mini.sort_values(ascending=False).index, orient='h')
plt.title('Degrees mentioned in Minimum Qualifications')
plt.xlabel('Count')
plt.ylabel('Degree')

plt.subplot(1,2,2)
sns.barplot(x=degree_prefer.sort_values(ascending=False).values, y=degree_prefer.sort_values(ascending=False).index, orient='h')
plt.title('Degrees mentioned in Prefered Qualifications')
plt.xlabel('Count')
plt.ylabel('Degree')
mini_string=" ".join(str(i) for i in mini)
prefer_string=" ".join(str(i) for i in prefer)
# the code to find number of years refer to  https://www.kaggle.com/niyamatalmass/what-you-need-to-get-a-job-at-google

from collections import defaultdict
years_exp_mini = defaultdict(lambda: 0)
years_exp_prefer = defaultdict(lambda: 0)
for w in re.findall(r'([0-9]+) year', mini_string):
     years_exp_mini[w] += 1
for w in re.findall(r'([0-9]+) year', prefer_string):
     years_exp_prefer[w] += 1
        
print(years_exp_mini)
print(years_exp_prefer)
years_exp_mini=pd.Series(years_exp_mini).sort_values(ascending=False)  
years_exp_prefer=pd.Series(years_exp_prefer).sort_values(ascending=False)  
# Visualize the years of experience in the minimum qualifications and preferred qualifications
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
years_exp_mini.plot.barh(x=years_exp_mini.values, y=years_exp_mini.index)
plt.title('Years of Experience mentioned in Minimum Qualifications')
plt.xlabel('Count')
plt.ylabel('Years of Experience')

plt.subplot(1,2,2)
years_exp_prefer.plot.barh(x=years_exp_prefer.values, y=years_exp_prefer.index)
plt.title('Years of Experience mentioned in Prefered Qualifications')
plt.xlabel('Count')
plt.ylabel('Years of Experience')
# Example for how to use the defaultdict
from collections import defaultdict
s = 'mississippi'
d = defaultdict(int)
for k in s:
    d[k]+=1
d.items()    
