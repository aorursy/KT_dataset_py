# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import re
import string
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def generate_wordcloud(column, stopwords):
    wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white',  
                stopwords = stopwords,
                min_font_size = 10).generate(str(df[column]))
    return wordcloud
    
def plot_wordcloud(wordcloud):
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.show()
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')

df.info()
df.head()
df['Job Description'] = df['Job Description'].apply(lambda x: re.sub('\n', ' ', str(x)))
df['Company Name'] = df['Company Name'].apply(lambda x: re.sub('\n...', '', str(x))) #replace all instances of the new line and company rating from 
df[['Job Description', 'Company Name']]
punc_string = string.punctuation #define list of punctuation
df['Job Description'] = df['Job Description'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) #remove all punctuataion using
df['Company Name'] = df['Company Name'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) #remove all punctuataion using
df[['Job Description', 'Company Name']]
wordcloud = WordCloud(width = 1000, height = 1000, 
                background_color ='white',  
                stopwords = STOPWORDS,
                min_font_size = 10).generate(str(df['Job Description']))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
new_stopwords = list(STOPWORDS) + ['Fanduel', 'group'] #stopwords is imported as a set. need to transform to a list in order to make it mutable
clean_cloud = generate_wordcloud('Job Description', new_stopwords)
plot_wordcloud(clean_cloud)
Location_count = df.Location.value_counts() >= 50
temp_df = pd.DataFrame(Location_count).reset_index()
temp_df.columns = ['Location', '50 Listings']
df = df.merge(temp_df)
df[df['50 Listings'] == True]['Location'].value_counts().plot(kind='bar')
plt.xlabel('Cities')
plt.ylabel('Count of job listings')
plt.title('Cities with >= 50 listings')
plt.show()
df_sector = df[~(df['Sector'] == '-1')]
sec_count = df_sector['Sector'].value_counts() 
sec_count_20 = sec_count[sec_count > 20]
sec_count_20.plot(kind='bar')
plt.xlabel('Sector')
plt.ylabel('Count of job listings')
plt.title('Amount of listings per sector')
plt.show()
df['Salary Estimate'].value_counts()
df = df[~(df['Salary Estimate'] == '-1')]
df[['Salary Min','Salary Max']] = df['Salary Estimate'].apply(lambda x: str(x).split('-')).tolist()
df['Salary Min'] = df['Salary Min'].apply(lambda x: str(x).strip('$K')).astype('int')
df['Salary Max'] = df['Salary Max'].apply(lambda x: str(x).strip('$K(Glassdoor est.)')).astype('int')
df_salary_50 = df[df['50 Listings'] == True]
df_salary_50_group = df_salary_50.groupby('Location').mean()[['Salary Min', 'Salary Max']]
df_salary_50_group.reset_index(inplace=True)
plt.style.use('seaborn')
sns.pointplot(data=df_salary_50_group, x='Location', y='Salary Min', color='red', label='Salary Min')
sns.pointplot(data=df_salary_50_group, x='Location', y='Salary Max', color='blue', label='Salary Max')
plt.xticks(rotation=45)
plt.xlabel('Cities (>=50 listings)')
plt.ylabel('Salary')
plt.title('Salary min/max for cities with most listings')
plt.legend()
plt.show()
#sal_for_box = df_salary_50[['Location', 'Salary Min', 'Salary Max']]
#sal_for_box.boxplot(by='Location', column = ['Salary Min', 'Salary Max'])
#sns.boxplot(data=df_salary_50, x='Location', y=[['Salary Min', 'Salary Max']])
#sns.boxplot(data=df_salary_50, x='Location', y='Salary Max', color='blue', order=['New York, NY'])
max_list= [0.75, 2.75, 4.75, 6.75, 8.75, 10.75, 12.75, 14.75, 16.75, 18.75]
min_list= [1.25, 3.25, 5.25, 7.25, 9.25, 11.25, 13.25, 15.25, 17.25, 19.25]
tick_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
tick_labels = list(df_salary_50.Location.unique()).sort()
ax = df_salary_50.boxplot(column='Salary Max', by='Location', positions=max_list)
df_salary_50.boxplot(column='Salary Min', by='Location',positions=min_list, ax=ax)
plt.xticks(tick_list,labels=tick_labels ,rotation=45)
plt.xlabel('Cities (>=50 listings)')
plt.ylabel('Salary')
plt.title('Salary min/max for cities with most listings')
#plt.legend()
plt.show()
sec_count_bool = sec_count >= 20
temp_df = pd.DataFrame(sec_count_bool).reset_index()
temp_df.columns = ['Sector', 'sector count >=20']
df = df.merge(temp_df)
df_sector = df[df['sector count >=20'] == True]
df_sector = df_sector.groupby('Sector').mean()[['Salary Min', 'Salary Max']]
df_sector.reset_index(inplace=True)
df_sector.head()
fig, ax = plt.subplots()

ax2= ax.twinx()

ax = sec_count_20.plot(kind='bar', ax=ax)
sns.pointplot(data=df_sector, x='Sector', y='Salary Max', color='red', ax=ax2)
sns.pointplot(data=df_sector, x='Sector', y='Salary Min', color='green', ax=ax2)
plt.xlabel('Sector')
plt.ylabel('Count of job listings')
plt.title('Amount of listings per sector')
plt.show()
