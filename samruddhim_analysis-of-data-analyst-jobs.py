import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#importing the dataset

data_analyst_jobs = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data_analyst_jobs
print(data_analyst_jobs.isnull().sum()) #checking for null values in the dataset
print(data_analyst_jobs.info()) #checking the general information of the dataset: non-null count, d-type, etc
data_analyst_jobs['Easy Apply'] = data_analyst_jobs['Easy Apply'].fillna(False).astype(bool) #As seen in dataset, Easy Apply column has -1 values, replacing them with boolean value False
data_analyst_jobs['Easy Apply'].value_counts() # Checking for value count of Easy Apply column
#removing unwanted columns
data_analyst_jobs.drop(['Unnamed: 0', 'Competitors', 'Easy Apply'], axis = 1, inplace = True) #Removing unwanted columns as they are not important for my further analysis
data_analyst_jobs.replace(['-1'], [np.nan], inplace=True)
data_analyst_jobs.replace(['-1.0'], [np.nan], inplace=True)
data_analyst_jobs.replace([-1], [np.nan], inplace=True)
data_analyst_jobs.isnull().sum()  #After replacing -1 with nan, we can see that there are null values in the dataset
data_analyst_salary = data_analyst_jobs['Salary Estimate'].str.split("-",expand=True,)

minimum_salary = data_analyst_salary[0]
minimum_salary = minimum_salary.str.replace('K',' ')


maximum_salary = data_analyst_salary[1].str.replace('(Glassdoor est.)', ' ')
maximum_salary = maximum_salary.str.replace('(', ' ')
maximum_salary = maximum_salary.str.replace(')', ' ')
maximum_salary = maximum_salary.str.replace('K', ' ')

maximum_salary = maximum_salary.str.replace('$', ' ').fillna(0).astype('int')
minimum_salary = minimum_salary.str.replace('$', ' ').fillna(0).astype('int')
data_analyst_jobs['Minimum Salary'] = minimum_salary
data_analyst_jobs['Maximum Salary'] = maximum_salary

data_analyst_jobs.drop('Salary Estimate',axis = 1,inplace = True)
data_analyst_jobs['Company Name'] = data_analyst_jobs['Company Name'].str.replace('\n.*', ' ')
Location = data_analyst_jobs['Location'].str.split(",",expand=True,)
Location_City = Location[0]
Location_State = Location[1]
data_analyst_jobs['Location City'] = Location_City
data_analyst_jobs['Location State'] = Location_State
data_analyst_jobs.drop('Location',axis = 1, inplace = True)

HQ = data_analyst_jobs['Headquarters'].str.split(",",expand=True)
Headquarters_City = HQ[0]
Headquarters_State = HQ[1]
data_analyst_jobs['Headquarters City'] = Headquarters_City
data_analyst_jobs['Headquarters State'] = Headquarters_State
data_analyst_jobs.drop('Headquarters',axis = 1, inplace = True)

department = data_analyst_jobs['Job Title'].str.split(',', expand = True)
#data_analyst_jobs['Job Title'], data_analysu_jobs['Department']
data_analyst_jobs['Job Title'], data_analyst_jobs['Department'] = department[0],department[1]
data_analyst_jobs.drop('Department',1, inplace = True)
data_analyst_jobs['Job Title'].value_counts()

data_analyst_jobs['Job Title'] = data_analyst_jobs['Job Title'].str.replace('Sr.', 'Senior')
data_analyst_jobs.info()
data_analyst_jobs['Type of ownership'].value_counts()
data_analyst_jobs['Industry'].value_counts()
data_analyst_jobs['Sector'].value_counts()
data_analyst_jobs['Revenue'].value_counts()
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].replace('Unknown / Non-Applicable', None)
# data['Revenue']=data['Revenue'].replace('Unknown / Non-Applicable', None)
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('$', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('(USD)', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('(', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace(')', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace(' ', '')
data_analyst_jobs['Revenue'].value_counts()
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('2to5billion', '2billionto5billion')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('5to10billion ', '5billionto10billion ')

data_analyst_jobs['Revenue'].value_counts()
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].replace('million', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].replace('10+billion', '10billionto11billion')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('Lessthan1million', '0millionto1million')
data_analyst_jobs['Revenue'].value_counts()
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('million', ' ')
data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].str.replace('billion', '000 ')
data_analyst_jobs['Revenue'].value_counts()

Revenue = data_analyst_jobs['Revenue'].str.split("to",expand=True)
Revenue[0].value_counts()
Revenue[1].value_counts()
data_analyst_jobs['Revenue'].value_counts()
data_analyst_jobs['Minimum Revenue'] = Revenue[0]
data_analyst_jobs['Maximum Revenue'] = Revenue[1]

data_analyst_jobs['Maximum Revenue'] = pd.to_numeric(data_analyst_jobs['Maximum Revenue'])
data_analyst_jobs['Minimum Revenue'] = pd.to_numeric(data_analyst_jobs['Minimum Revenue'])
data_analyst_jobs.drop('Revenue',1,inplace=True)
data_analyst_jobs
data_analyst_jobs['Size'].value_counts()
data_analyst_jobs['Size'] = data_analyst_jobs['Size'].str.replace('employees', '')

data_analyst_jobs['Size'] = data_analyst_jobs['Size'].str.replace('+', 'plus')
data_analyst_jobs['Size'] = data_analyst_jobs['Size'].replace('Unknown', None)


data_analyst_jobs['Size'] = data_analyst_jobs['Size'].str.replace('10000plus', '10000 to 10001')
size = data_analyst_jobs['Size'].str.split("to",expand=True)
data_analyst_jobs['Minimum Size'] = size[0]
data_analyst_jobs['Maximum Size'] = size[1]
data_analyst_jobs
data_analyst_jobs.drop('Size',1,inplace = True)
# def contains_word(s, w):
#     return f' {w} ' in f' {s} '

# # def rev(text):
# #     #if contains_word(text,'billion') is True:
# #     text.str.replace('billion','')
         
# #     return text

# def revenue(text):
#     if contains_word(text,'billion') is True:
#         max_rev = float(data_analyst_jobs['Maximum Revenue'].replace("billion", " ").strip())*1000
#         #revenue = float(maxRev[0].replace('+','').strip())*100
#     return max_rev

# data_analyst_jobs['Revenue'] = data_analyst_jobs['Revenue'].apply(lambda text: clean_revenue(text))
f, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True)
sns.despine(left=True)
sns.distplot(data_analyst_jobs['Minimum Salary'],color = 'r',ax = axes[0])
sns.distplot(data_analyst_jobs['Maximum Salary'],ax = axes[1])
plt.legend();
sns.boxplot(x = data_analyst_jobs['Rating']);
data_analyst_jobs['Minimum Size'] = data_analyst_jobs['Minimum Size'].astype('float')
data_analyst_jobs['Maximum Size'] = data_analyst_jobs['Maximum Size'].astype('float')


f, axes = plt.subplots(1, 2, figsize=(20, 5), sharex=True)
sns.boxplot(x = data_analyst_jobs['Minimum Size'], ax = axes[0],palette='Set1');
sns.boxplot(x = data_analyst_jobs['Maximum Size'], ax = axes[1],palette='Set2');
plt.subplots(figsize=(10,10))
splot = sns.barplot(x=data_analyst_jobs['Job Title'].value_counts()[0:20].index,y=data_analyst_jobs['Job Title'].value_counts()[0:20], palette = 'winter_r')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.xlabel('Job Title',fontsize=15)
plt.ylabel('Job Count',fontsize=15)
plt.xticks(rotation=90)
plt.yticks(fontsize=15)
plt.title('Top 20 Job Title Counts',fontsize=25);

# for index, row in data_analyst_jobs.iterrows():
#     splot.text(row.name,row.tip, round('Job Title',2), color='black', ha="center")

plt.subplots(figsize=(15,15))
splot = sns.barplot(x = data_analyst_jobs['Company Name'][0:20], y = data_analyst_jobs['Maximum Revenue'][0:20], data = data_analyst_jobs, palette = 'spring')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')


plt.xlabel('Company Name',fontsize=15)
plt.ylabel('Maximum revenue in million dollars',fontsize=15)
plt.xticks(rotation=90)
plt.yticks(fontsize=20)
plt.title('Maximum Revenue of top 20 Companies',fontsize=25);
data_analyst_jobs['Average Revenue'] = data_analyst_jobs[['Minimum Revenue','Maximum Revenue']].mean(axis=1)

avg_rev = data_analyst_jobs['Average Revenue'][0:20]
avg_rev
plt.subplots(figsize=(20,15))
splot = sns.barplot(x = data_analyst_jobs['Company Name'][0:20], y = data_analyst_jobs['Average Revenue'][0:20], data = data_analyst_jobs, palette = 'summer')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.xlabel('Company Name')
plt.ylabel('Average revenue in million dollars')
plt.xticks(rotation=90)
plt.yticks(fontsize=20)
plt.title('Average Revenue of top 20 Companies',fontsize=25);

data = data_analyst_jobs.groupby('Location City')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
data
import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Bar(
   x = data.index,
   y = data['Minimum Salary'],
   name = 'Minimum Salary'
))

fig.add_trace(go.Bar(
   x = data.index,
   y = data['Maximum Salary'],
   name = 'Maximum Salary'
))

#data1 = [plot1,plot2]
fig.update_layout(title = 'Minimum and Maximum salaries of top 25 cities', barmode = 'group')
#fig = go.Figure(data = data, layout = layout)

fig.show()
data1 = data_analyst_jobs.groupby('Job Title')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
data1
import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Bar(
   x = data1.index,
   y = data1['Minimum Salary'],
   name = 'Minimum Salary'
))

fig.add_trace(go.Bar(
   x = data1.index,
   y = data1['Maximum Salary'],
   name = 'Maximum Salary'
))

#data1 = [plot1,plot2]
fig.update_layout(title = 'Minimum and Maximum salaries of top 25 job titles', barmode = 'stack')
#fig = go.Figure(data = data, layout = layout)

fig.show()
data_analyst_jobs['Average Salary'] = data_analyst_jobs[['Minimum Salary', 'Maximum Salary']].mean(axis = 1)
import plotly.express as px
fig = px.scatter(data_analyst_jobs, x=data_analyst_jobs['Rating'], y= data_analyst_jobs['Average Salary'])
fig.update_layout(title = 'Relation between average salary and rating of companies')
fig.show()

data2 = data_analyst_jobs.groupby('Founded')[['Average Revenue']].mean().sort_values(['Average Revenue'],ascending=False).head(25)
data2
fig = px.line(x=data2['Average Revenue'], y=data2.index, labels={'x':'Average Revenue', 'y':'Year founded'})
fig.update_layout(title = 'Relation between the average revenue and year the company was founded')
fig.show()
data3 = data_analyst_jobs.groupby('Founded')[['Average Revenue']].mean().sort_values(['Average Revenue'],ascending=False).tail(25)
data3
fig = px.line(x=data3['Average Revenue'], y=data3.index, labels={'x':'Average Revenue', 'y':'Year founded'})
fig.update_layout(title = 'Relation between the average revenue and year the company was founded')
fig.show()
data4 = pd.DataFrame(data_analyst_jobs['Sector'].value_counts())
data4
import plotly.express as px
fig = px.pie(data4, values=data4['Sector'], names=data4.index)
fig.update_layout(title = 'Percentage of Different Sectors with requirement of Data Analyst Roles')
fig.show()

data5 = pd.DataFrame(data_analyst_jobs['Industry'].value_counts().head(25))
data5
import plotly.express as px
fig = px.pie(data5, values=data5['Industry'], names=data5.index)
fig.update_layout(title = 'Percentage of top 25 Industries with requirement of Data Analyst Roles')
fig.show()


data6 = pd.DataFrame(data_analyst_jobs['Type of ownership'].value_counts())
data6

import plotly.express as px
fig = px.pie(data6, values=data6['Type of ownership'], names=data6.index)
fig.update_layout(title = 'Type of ownership')
fig.show()



data7 = pd.DataFrame(data_analyst_jobs['Headquarters City'].value_counts().head(25))
data7

import plotly.express as px
fig = px.pie(data7, values=data7['Headquarters City'], names=data7.index)
fig.update_layout(title = 'Top 25 Headquarter City')
fig.show()




data8 = pd.DataFrame(data_analyst_jobs['Location City'].value_counts().head(25))
data8

import plotly.express as px
fig = px.pie(data8, values=data8['Location City'], names=data8.index)
fig.update_layout(title = 'Top 25 Job Locations')
fig.show()





from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
plt.subplots(figsize=(15,15))
wc = WordCloud()
text = data_analyst_jobs['Job Title']
wc.generate(str(' '.join(text)))
plt.imshow(wc)
plt.axis("off")
plt.show()
# import nltk
# from nltk.corpus import stopwords
# import re
# from nltk.stem.porter import PorterStemmer
# print(stopwords.words('english'))

# stop_words = set(stopwords.words('english'))
# jobdes = data_analyst_jobs['Job Description'].to_csv()
# jobdes = jobdes.split(' ')
# jobdes = jobdes.lower()
# jobdes

# skills = ['python', 'java','c', 'r','c++', 'hadoop', 'communication']

# for word in all_words:
#     print(word)
usa_map = data_analyst_jobs.groupby('Location City')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False)
usa_map = usa_map.reset_index()
usa_map.head(20)

cities = usa_map['Location City']
cities.head(20)

['Daly City','Marin City', 'Los Gatos', 'Berkeley', 'San Jose', 'Cupertino','Santa Clara', 'Pico Rivera', 'Whittier','Far Rockaway', 'Secaucus', 'Sunnyvale', 'Menlo Park', 'Elk Grove Village', 'Glenview', 'Maywood', 'Northfield', 'Stanford', 'San Francisco', 'El Cajon']
usa_maps = data_analyst_jobs.groupby('Location State')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False)
usa_maps = usa_maps.reset_index()

usa_maps = usa_maps.drop([3, 0])
usa_maps
import plotly.express as px

fig = px.choropleth(locations= ['AZ','NJ','NY','CO','IL','NC','VA','SC','WA','PA','DE','TX','KS','FL','IN','OH','GA','UT'], 
                    locationmode="USA-states", 
                    color=[94.494845, 90.232558, 89.026087, 89.022727, 88.829268,85.233333, 85.125000, 83.000000, 82.759259, 77.824561, 75.909091, 74.116751, 67.000000, 66.666667, 61.000000, 58.800000, 56.000000, 48.454545],
                    labels={'color':'Maximum Salary', 'locations':'State'},
                    scope="usa") 


fig.update_layout(
    
    title_text = 'Top 20 States with Maximum Salary',
    geo_scope='usa'
)
fig.show()