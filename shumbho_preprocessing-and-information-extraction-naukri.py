# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns



from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv")

df.head(3)
df.info()
df[df.duplicated(keep=False)]
df[df.duplicated(['Uniq Id'])]
df.drop(['Uniq Id','Crawl Timestamp'], axis=1, inplace=True)
print('Percentage of missing values :')

print(df.isnull().sum()*100/(df.notnull().sum()+df.isnull().sum()))



plt.figure(figsize=(12,6))

df.isnull().sum().plot(kind='bar', colormap='Accent')

plt.title('Missing Values Plot')

plt.xlabel('Features -->')

plt.ylabel('No. of missing values -->')

plt.show()

df.dropna(subset=np.delete(df.columns.values, 4),inplace=True)

df.isnull().sum()*100/(df.notnull().sum()+df.isnull().sum())
print(df.nunique())

plt.figure(figsize=(15,4))

plt.plot(df.nunique())

plt.grid()

plt.title('Number of unique values')

plt.show()
### Pre-process Job Title
job_title = df['Job Title'].value_counts()
plt.figure(figsize=(18,6))

plt.plot(job_title.values)

plt.xlabel('No. of samples -->')

plt.ylabel('No. of job titles -->')

plt.title('Number of job titles plot')

plt.grid()

plt.show()
df['Job Title'] = [re.sub('[^a-zA-Z]+', ' ', i.lstrip()) for i in df['Job Title']]

df['Job Title'].value_counts().iloc[:10]
df['Job Salary'] = [re.sub('[^0-9,-]+', '', i) if 'PA' in i else 'Not Disclosed by Recruiter' for i in df['Job Salary']]

print(df['Job Salary'].value_counts().iloc[:10])

plt.figure(figsize=(12,5))

df['Job Salary'].value_counts().iloc[1:20].plot(kind='barh', colormap='Accent')

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('Salaries -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Job Salaries Plot')

plt.show()
df['Job Experience Required'] = [re.sub('[^0-9,-]+', '', i) for i in df['Job Experience Required']]

print(df['Job Experience Required'].value_counts().iloc[:10])

plt.figure(figsize=(12,5))

df['Job Experience Required'].value_counts().iloc[:20].plot(kind='bar', colormap='Accent')

plt.xticks(fontsize=14, rotation=0)

plt.yticks(fontsize=14)

plt.xlabel('Experience req. (in years) -->',fontsize=14)

plt.ylabel('No. of jobs -->',fontsize=14)

plt.title('Job Exp. Required Plot',fontsize=14)

plt.show()
df['Key Skills'] = df['Key Skills'].map(lambda x: ' '.join(['_'.join(i.lstrip().split()) for i in x.split('|') if len(i) > 1]).lower())

df['Key Skills'].value_counts().index.ravel()


common_words = ' '.join(df['Key Skills'].value_counts().index.ravel())

  

wordcloud = WordCloud(width = 1200, height = 600, 

                background_color ='white',

                min_font_size = 10).generate(common_words) 





plt.figure(figsize = (16, 6), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
print('Median length of strings:', np.median(df['Role Category'].map(lambda x: len(str(x))).tolist()))

print('Mean length of strings: %.2f' % np.array(df['Role Category'].map(lambda x: len(str(x))).tolist()).mean())

print('Std. Deviation: %.2f' % np.array(df['Role Category'].map(lambda x: len(str(x))).tolist()).std())
df['Role Category'].value_counts().iloc[:15]
df['Role Category'] = df['Role Category'].map(lambda x: np.where(len(str(x)) > 80, 'N.A.', x))

df['Role Category'] = df['Role Category'].map(lambda x: re.sub('\(.*?\)', '', x))

df['Role Category'] = df['Role Category'].map(lambda x: ' '.join(['_'.join(j.lstrip().rstrip().split()) if len(j.lstrip().rstrip().split()) != 1 else j.lstrip().rstrip() for j in x.split('/')]))
df.loc[df['Role Category'].str.contains('Ford', case=False), 'Role Category'] = 'Operations'

df.loc[df['Role Category'].str.contains('Telecom', case=False), 'Role Category'] = 'Telecom'

df.loc[df['Role Category'].str.contains('HR', case=False), 'Role Category'] = 'HR Recruitment IR'

df.loc[df['Role Category'].str.contains('LOGISTICS', case=False), 'Role Category'] = 'Logistics'

df.loc[df['Role Category'].str.contains('Bank', case=False), 'Role Category'] = 'Retail Personal_Banking'

df.loc[df['Role Category'].str.contains('Sales', case=False), 'Role Category'] = 'Sales'

df.loc[df['Role Category'].str.contains('System_Design', case=False), 'Role Category'] = 'Programming_&_Design'

df.loc[df['Role Category'].str.contains('nan', case=False), 'Role Category'] = 'N.A.'

print(df['Role Category'].value_counts().iloc[:10])

plt.figure(figsize=(12,6))

df['Role Category'].value_counts().iloc[:20].plot(kind='barh', colormap='Accent')

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('Role -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Role for jobs Plot',fontsize=14)

plt.show()
df['Location'] = df['Location'].map(lambda x: re.sub('\(.*?\)|[^\w]|More', ' ',  x))

df['Location'] = df['Location'].map(lambda x: ' '.join(set(x.strip().split())).upper())
print(df['Location'].value_counts().iloc[:10])

plt.figure(figsize=(12,6))

df['Location'].value_counts().iloc[:20].plot(kind='barh', colormap='Accent')

plt.yticks(fontsize=14)

plt.xticks(fontsize=14)

plt.ylabel('Locations -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Top Locations with max jobs Plot',fontsize=14)

plt.show()
master = list()

for i in df.Location:

    master.extend(i.split(' '))

  

wordcloud = WordCloud(width = 1200, height = 600, 

                background_color ='white',

                min_font_size = 10).generate(' '.join(master)) 





plt.figure(figsize = (16, 6), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
df['Functional Area'] = df['Functional Area'].map(lambda x: np.where(len(str(x)) > 80, 'N.A.', x))

df['Functional Area'] = df['Functional Area'].map(lambda x: x.replace('/',','))

df['Functional Area'] = df['Functional Area'].map(lambda x: ' '.join(['_'.join(i.split()) if len(i.strip().split(' ')) > 1 else i.strip() for i in x.split(',')]))
df.loc[df['Functional Area'].str.contains('IT_Software_', case=False), 'Functional Area'] = 'IT_Software'
print(df['Functional Area'].value_counts().iloc[:10])

plt.figure(figsize=(12,6))

df['Functional Area'].value_counts().iloc[:20].plot(kind='barh', colormap='Accent')

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('Functional Area -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Function area jobs Plot',fontsize=14)

plt.show()


functional_area = ' '.join(df['Functional Area'].values.ravel())

  

wordcloud = WordCloud(width = 1200, height = 600, 

                background_color ='white',

                min_font_size = 10).generate(functional_area) 



plt.figure(figsize = (16, 6), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show()
df['Industry'].value_counts()
df.loc[df['Industry'].str.contains('Allegis', case=False), 'Industry'] = 'IT-Software, Software Services'

df.loc[df['Industry'].str.contains('TEKsystems', case=False), 'Industry'] = 'IT-Software, Software Services'

df.loc[df['Industry'].str.contains('Solugenix', case=False), 'Industry'] = 'IT-Software, Software Services'

df.loc[df['Industry'].str.contains('Laurus Labs', case=False), 'Industry'] = 'Medical, Healthcare, Hospitals'

df.loc[df['Industry'].str.contains('Ford', case=False), 'Industry'] = 'Automobile, Auto Anciliary, Auto Components'

df['Industry'] = df['Industry'].str.replace('/',',')
df['Industry'] = df['Industry'].map(lambda x: ' '.join(['_'.join(i.strip().split(' ')) if len(i.strip().split(' ')) > 1 else i.strip() for i in x.split(',')]))
print(df['Industry'].value_counts())

plt.figure(figsize=(12,6))

df['Industry'].value_counts().iloc[:20].plot(kind='barh', colormap='Accent')

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('Industry -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Industry wise jobs Plot',fontsize=14)

plt.show()
df['Role'].value_counts()
df['Role'] = df['Role'].map(lambda x: np.where(len(str(x)) > 80, 'Other', x))
print(df['Role'].value_counts().iloc[:10])

plt.figure(figsize=(12,6))

df['Role'].value_counts().iloc[:20].plot(kind='barh', colormap='Accent')

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.ylabel('Roles -->',fontsize=14)

plt.xlabel('No. of jobs -->',fontsize=14)

plt.title('Role wise jobs Plot',fontsize=14)

plt.show()
data = df.copy()


for i in data['Job Title'].value_counts().index.tolist()[:10]:

    df = data[(data['Job Title'] == i) & (data['Job Salary'] != 'Not Disclosed by Recruiter')][['Job Salary','Job Experience Required']]

    fig = px.bar(df, x='Job Salary', y='Job Experience Required',

                 hover_data=['Job Salary', 'Job Experience Required'], color='Job Salary',

                 labels={'pop':'population of Canada'}, height=400,

                 title = 'Top 10 most popular Job Titles Exp and Salary : ' + i)

    fig.show()

    print('-'*80)

df = data[data['Job Salary'] != 'Not Disclosed by Recruiter'].sort_values('Job Salary',ascending=False)[['Job Salary','Role Category']]

fig = px.sunburst(df, path=['Role Category','Job Salary'])

fig.show()
data['Average_Job_Exp_Req'] = data['Job Experience Required'].map(lambda x: re.sub('[^0-9]',' ',x))

data['Average_Job_Exp_Req'] = data['Average_Job_Exp_Req'].map(lambda x: np.mean([int(i) for i in x.split()]))
df = data.sort_values(by = "Average_Job_Exp_Req", ascending = False)[['Job Title','Role','Average_Job_Exp_Req']].reset_index(drop=True)
fig = px.sunburst(df, path=['Role','Average_Job_Exp_Req'])

fig.show()
for i in data.Location.value_counts().index.tolist()[:10]:

    fig = px.pie(data[data.Location == i], names='Role Category',title = 'Job Roles in ' + i)

    fig.show()
data['Average_Salary'] = data[data['Job Salary'] != 'Not Disclosed by Recruiter']['Job Salary'].str.replace('-',' ')

data['Average_Salary'] = data['Average_Salary'].str.replace(',','')

data.loc[data[data.Average_Salary.notnull()]['Average_Salary'].index,'Average_Salary'] = data[data.Average_Salary.notnull()].Average_Salary.map(lambda x: np.mean([int(i) for i in x.split()]))
data[data.Average_Salary.notnull()][['Key Skills','Average_Salary']].sort_values('Average_Salary',ascending=False).reset_index(drop=True)[:50]