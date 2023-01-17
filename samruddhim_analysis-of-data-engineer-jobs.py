import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#importing the dataset

df = pd.read_csv('../input/data-engineer-jobs/DataEngineer.csv')
df
print(df.isnull().sum()) #checking for null values in the dataset
print(df.info()) #checking the general information of the dataset: non-null count, d-type, etc
df['Easy Apply'] = df['Easy Apply'].fillna(False).astype(bool) #As seen in dataset, Easy Apply column has -1 values, replacing them with boolean value False
df['Easy Apply'].value_counts() # Checking for value count of Easy Apply column
df.replace(['-1'], [np.nan], inplace=True)
df.replace(['-1.0'], [np.nan], inplace=True)
df.replace([-1], [np.nan], inplace=True)
df.isnull().sum()  #After replacing -1 with nan, we can see that there are null values in the dataset
df_salary = df['Salary Estimate'].str.split("-",expand=True,)

minimum_salary = df_salary[0]
minimum_salary = minimum_salary.str.replace('K',' ')


maximum_salary = df_salary[1].str.replace('(Glassdoor est.)', ' ')
maximum_salary = maximum_salary.str.replace('(', ' ')
maximum_salary = maximum_salary.str.replace(')', ' ')
maximum_salary = maximum_salary.str.replace('K', ' ')
maximum_salary = maximum_salary.str.replace('Employer est.', ' ')

maximum_salary = maximum_salary.str.replace('$', ' ').fillna(0).astype(int)
minimum_salary = minimum_salary.str.replace('$', ' ').fillna(0).astype(int)
df['Minimum Salary'] = minimum_salary
df['Maximum Salary'] = maximum_salary

df.drop('Salary Estimate',axis = 1,inplace = True)
df['Company Name'] = df['Company Name'].str.replace('\n.*', ' ')
Location = df['Location'].str.split(",",expand=True,)
Location_City = Location[0]
Location_State = Location[1]
df['Location City'] = Location_City
df['Location State'] = Location_State
df.drop('Location',axis = 1, inplace = True)

HQ = df['Headquarters'].str.split(",",expand=True)
Headquarters_City = HQ[0]
Headquarters_State = HQ[1]
df['Headquarters City'] = Headquarters_City
df['Headquarters State'] = Headquarters_State
df.drop('Headquarters',axis = 1, inplace = True)

department = df['Job Title'].str.split(',', expand = True)
df['Job Title'], df['Department'] = department[0],department[1]
df.drop('Department',1, inplace = True)
df['Job Title'].value_counts()

df['Job Title'] = df['Job Title'].str.replace('Sr.', 'Senior')
df.info()
df['Type of ownership'].value_counts()
df['Industry'].value_counts()
df['Sector'].value_counts()
df['Revenue'].value_counts()
df['Revenue'] = df['Revenue'].replace('Unknown / Non-Applicable', None)
# data['Revenue']=data['Revenue'].replace('Unknown / Non-Applicable', None)
df['Revenue'] = df['Revenue'].str.replace('$', ' ')
df['Revenue'] = df['Revenue'].str.replace('(USD)', ' ')
df['Revenue'] = df['Revenue'].str.replace('(', ' ')
df['Revenue'] = df['Revenue'].str.replace(')', ' ')
df['Revenue'] = df['Revenue'].str.replace(' ', '')
df['Revenue'].value_counts()
df['Revenue'] = df['Revenue'].str.replace('2to5billion', '2billionto5billion')
df['Revenue'] = df['Revenue'].str.replace('5to10billion ', '5billionto10billion ')

df['Revenue'].value_counts()
df['Revenue'] = df['Revenue'].replace('million', ' ')
df['Revenue'] = df['Revenue'].replace('10+billion', '10billionto11billion')
df['Revenue'] = df['Revenue'].str.replace('Lessthan1million', '0millionto1million')
df['Revenue'].value_counts()
df['Revenue'] = df['Revenue'].str.replace('million', ' ')
df['Revenue'] = df['Revenue'].str.replace('billion', '000 ')
df['Revenue'].value_counts()

Revenue = df['Revenue'].str.split("to",expand=True)
Revenue[0].value_counts()
Revenue[1].value_counts()
df['Revenue'].value_counts()
df['Minimum Revenue'] = Revenue[0]
df['Maximum Revenue'] = Revenue[1]

df['Maximum Revenue'] = pd.to_numeric(df['Maximum Revenue'])
df['Minimum Revenue'] = pd.to_numeric(df['Minimum Revenue'])
df.drop('Revenue',1,inplace=True)
df
df['Size'].value_counts()
df['Size'] = df['Size'].str.replace('employees', '')

df['Size'] = df['Size'].str.replace('+', 'plus')
df['Size'] = df['Size'].replace('Unknown', None)


df['Size'] = df['Size'].str.replace('10000plus', '10000 to 10001')
size = df['Size'].str.split("to",expand=True)
df['Minimum Size'] = size[0]
df['Maximum Size'] = size[1]
df
df.drop('Size',1,inplace = True)
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
sns.distplot(df['Minimum Salary'],color = 'r',ax = axes[0])
sns.distplot(df['Maximum Salary'],ax = axes[1])
plt.legend();
sns.boxplot(x = df['Rating']);
df['Minimum Size'] = df['Minimum Size'].astype('float')
df['Maximum Size'] = df['Maximum Size'].astype('float')


f, axes = plt.subplots(1, 2, figsize=(20, 5), sharex=True)
sns.boxplot(x = df['Minimum Size'], ax = axes[0],palette='Set1');
sns.boxplot(x = df['Maximum Size'], ax = axes[1],palette='Set2');
plt.subplots(figsize=(10,10))
splot = sns.barplot(x=df['Job Title'].value_counts()[0:20].index,y=df['Job Title'].value_counts()[0:20], palette = 'winter_r')
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
splot = sns.barplot(x = df['Company Name'][0:20], y = df['Maximum Revenue'][0:20], data = df, palette = 'spring')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')


plt.xlabel('Company Name',fontsize=15)
plt.ylabel('Maximum revenue in million dollars',fontsize=15)
plt.xticks(rotation=90)
plt.yticks(fontsize=20)
plt.title('Maximum Revenue of top 20 Companies',fontsize=25);
df['Average Revenue'] = df[['Minimum Revenue','Maximum Revenue']].mean(axis=1)

avg_rev = df['Average Revenue'][0:20]
avg_rev
plt.subplots(figsize=(20,15))
splot = sns.barplot(x = df['Company Name'][0:20], y = df['Average Revenue'][0:20], data = df, palette = 'summer')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 15), textcoords = 'offset points')

plt.xlabel('Company Name')
plt.ylabel('Average revenue in million dollars')
plt.xticks(rotation=90)
plt.yticks(fontsize=20)
plt.title('Average Revenue of top 20 Companies',fontsize=25);

data = df.groupby('Location City')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
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
data1 = df.groupby('Job Title')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False).head(25)
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
df['Average Salary'] = df[['Minimum Salary', 'Maximum Salary']].mean(axis = 1)
import plotly.express as px
fig = px.scatter(df, x=df['Rating'], y= df['Average Salary'])
fig.update_layout(title = 'Relation between average salary and rating of companies')
fig.show()

data2 = df.groupby('Founded')[['Average Revenue']].mean().sort_values(['Average Revenue'],ascending=False).head(25)
data2
fig = px.line(x=data2['Average Revenue'], y=data2.index, labels={'x':'Average Revenue', 'y':'Year founded'})
fig.update_layout(title = 'Relation between the average revenue and year the company was founded')
fig.show()
data3 = df.groupby('Founded')[['Average Revenue']].mean().sort_values(['Average Revenue'],ascending=False).tail(25)
data3
fig = px.line(x=data3['Average Revenue'], y=data3.index, labels={'x':'Average Revenue', 'y':'Year founded'})
fig.update_layout(title = 'Relation between the average revenue and year the company was founded')
fig.show()
data4 = pd.DataFrame(df['Sector'].value_counts())
data4
import plotly.express as px
fig = px.pie(data4, values=data4['Sector'], names=data4.index)
fig.update_layout(title = 'Percentage of Different Sectors with requirement of Data Engineer Roles')
fig.show()

data5 = pd.DataFrame(df['Industry'].value_counts().head(25))
data5
import plotly.express as px
fig = px.pie(data5, values=data5['Industry'], names=data5.index)
fig.update_layout(title = 'Percentage of top 25 Industries with requirement of Data Analyst Roles')
fig.show()


data6 = pd.DataFrame(df['Type of ownership'].value_counts())
data6

import plotly.express as px
fig = px.pie(data6, values=data6['Type of ownership'], names=data6.index)
fig.update_layout(title = 'Type of ownership')
fig.show()



data7 = pd.DataFrame(df['Headquarters City'].value_counts().head(25))
data7

import plotly.express as px
fig = px.pie(data7, values=data7['Headquarters City'], names=data7.index)
fig.update_layout(title = 'Top 25 Headquarter City')
fig.show()




data8 = pd.DataFrame(df['Location City'].value_counts().head(25))
data8

import plotly.express as px
fig = px.pie(data8, values=data8['Location City'], names=data8.index)
fig.update_layout(title = 'Top 25 Job Locations')
fig.show()





from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
plt.subplots(figsize=(15,15))
wc = WordCloud()
text = df['Job Title']
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
usa_map = df.groupby('Location City')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False)
usa_map = usa_map.reset_index()
usa_map.head(20)

cities = usa_map['Location City']
cities.head(20)

['Daly City','Marin City', 'Los Gatos', 'Berkeley', 'San Jose', 'Cupertino','Santa Clara', 'Pico Rivera', 'Whittier','Far Rockaway', 'Secaucus', 'Sunnyvale', 'Menlo Park', 'Elk Grove Village', 'Glenview', 'Maywood', 'Northfield', 'Stanford', 'San Francisco', 'El Cajon']
usa_maps = df.groupby('Location State')[['Minimum Salary', 'Maximum Salary']].mean().sort_values(['Maximum Salary','Minimum Salary'],ascending=False)
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