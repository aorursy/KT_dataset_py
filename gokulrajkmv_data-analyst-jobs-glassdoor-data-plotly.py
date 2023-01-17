#importing packages for data Analysis

import numpy as np # linear algebra
import pandas as pd # data processing

#importing wordcloud for visual

from wordcloud import WordCloud, STOPWORDS
#importing packages for data visuals

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#importing packages for interactive data visuals

from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


from IPython.display import HTML
#importing data

df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv',index_col = 0)
df.info()
df.tail(3)
# we can see in above tail of the data columns nun value is filled with -1,-1.0 and "-1" so we can replace with np.nun

df.replace(to_replace =-1,  value = np.nan,inplace=True)

df.replace(to_replace =-1.0,value = np.nan,inplace=True)

df.replace(to_replace ='-1',value = np.nan,inplace=True)
#let us look at Missing data

missing_data = df.isnull().sum()/len(df)*100

missing_data.iplot(kind='bar',color='#F52D5E',  title='Missing values in each column', yTitle='In percentage')

# since competitors and Easy Apply columns are almost 70% of data is missing so it is not required for analysis
 
df.drop(['Easy Apply','Competitors'],1,inplace = True)


# we can continue cleaning for Salary, Company Rating and Location
                 
# cleaning Salary Estimate column
# let us remove the unwanted 'strings'

df['Salary Estimate'] = df['Salary Estimate'].str.replace('Glassdoor est.','')
df['Salary Estimate'] = df['Salary Estimate'].str.replace('$', '')
df['Salary Estimate'] = df['Salary Estimate'].str.replace('K','')
df['Salary Estimate'] = df['Salary Estimate'].str.strip('()')

#let us split the Salary Estimate into two parts Min and Max

Salary_min_max = df['Salary Estimate'].str.split('-',expand=True)
# join the columns
df['Salary_Min'] = Salary_min_max[0]
df['Salary_Max'] = Salary_min_max[1]

# convert the column into float64
df['Salary_Min'] = pd.to_numeric(df['Salary_Min'])
df['Salary_Max'] = pd.to_numeric(df['Salary_Max'])
# cleaning company name column

df['Company Name'] = df['Company Name'].str.replace(r'\W'," ")
df['Company Name'] = df['Company Name'].str.replace('\d+', '')
# cleaning location column

df['Location'] = df['Location'].str.replace(r'\W'," ")
df['Location'] = df['Location'].str.replace('\d+', '')
# cleaning Job Title column

df['Job Title'] = df['Job Title'].str.replace(r'\W'," ")
df['Job Title'] = df['Job Title'].str.replace('\d+', '')
#df['Job Description']


df['Job Description'] = df['Job Description'].str.replace(r'\W'," ")
df['Job Description'] = df['Job Description'].str.replace('\d+', '')
df.info()
# ploting data 

plt.figure(figsize=(10,6))
sns.set_context(context='notebook', font_scale=1)
sns.set_style('whitegrid')
sns.pairplot(df)
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df['Job Title']: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1200, height = 1200, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = '#9CC9AD') 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Job Title')  
plt.show() 
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df['Job Description']: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for text in range(len(tokens)): 
        tokens[text] = tokens[text].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1200, height = 1200, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 20).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = '#6897BB') 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Job Description')  
plt.show() 


# data for plot

df[df['Job Title']== 'Data Analyst']['Location'].value_counts()[0:10].iplot(kind='bar',color='#FFC300',title='Top 10 location with Data Analyst job')
    
# ploting the rating

plt.figure(figsize=(14,6))
sns.countplot(df['Rating'])
plt.title('Rating')
# top 10 rated company 

df[df['Rating'] >= 4 ]['Company Name'].dropna().value_counts()[0:10].iplot(kind='bar',color='green',title='Top 10 company with greater than 4 rating')
print('*******************************************')
print('The Average rating is Data Analyst',round(df[df['Job Title'] == 'Data Analyst']['Rating'].mean(),2)) 
print('*******************************************')
print('*****************************************************')
print('Minimum mean salary for Data Analyst role is',round(df[df['Job Title'] == 'Data Analyst']['Salary_Min'].mean()*1000,2))
print('*****************************************************')
print('Maximum mean salary for Data Analyst role is',round(df[df['Job Title'] == 'Data Analyst']['Salary_Max'].mean()*1000,2))
print('*****************************************************')
# let us plot the distribution of salary

x = df['Salary_Min']
y = df['Salary_Max']


fig, ax = plt.subplots(1,2,figsize=(15, 6))

plt.figure(figsize=(10,6))

sns.set_style('dark')
sns.set_context(context = 'notebook',font_scale=1)
sns.distplot(x, ax = ax[0],color='red',bins=5,kde=False)
sns.distplot(y, ax = ax[1],color='blue',bins=5,kde=False)

ax[0].title.set_text('Minimum Salary')
ax[1].title.set_text('Maximum Salary')

plt.tight_layout()


# plotting location where above mean salary of Data Analyst''


df[(df['Salary_Max'] >= 95)]['Location'].value_counts()[0:10].iplot(kind='bar',color='#52E2FE',title='Top 10 location where salary of Data Analyst above mean salary')

df[(df['Rating'] == 4.5)]['Job Title'].value_counts()[0:10].iplot(kind='bar',color='#61E6A8',title='Top rated Job title')
rating_job_title = df[(df['Rating'] == 4.5)]['Job Title']

comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in rating_job_title: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for text in range(len(tokens)): 
        tokens[text] = tokens[text].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1200, height = 1200, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 20).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = '#52E2FE') 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Top Rated job title')  
plt.show() 
# top 10 Industry

job_industry = df['Industry'].value_counts()[0:10]


job_industry.iplot(kind='bar',color='#61E6A8',title='Top 10 Industry')
# let us compare top 10 Industry in jobs with wordcloud

comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df['Industry']: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for text in range(len(tokens)): 
        tokens[text] = tokens[text].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1200, height = 1200, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 5).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = '#61E6A8') 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Industry')  
plt.show() 
#top 10 Company with Data analyst role

df['Company Name'].value_counts()[0:10].iplot(kind='bar',color='purple',title='top 10 Company with Data analyst role') 
# what is mean salary of Data Analyst which is posted.

print('what is mean salary of Data Analyst which is posted ?')
print('******************************')
print('Minimum mean salary is',round(df[df['Job Title'] == 'Data Analyst']['Salary_Min'].mean()*1000,2))
print('******************************')
print('Maximum mean salary is',round(df[df['Job Title'] == 'Data Analyst']['Salary_Max'].mean()*1000,2))
print('******************************')

print('\n')

print('what is average rating for the job title Data Analyst ?')
print('******************************')
print('The Average rating is',round(df[df['Job Title'] == 'Data Analyst']['Rating'].mean(),2)) 