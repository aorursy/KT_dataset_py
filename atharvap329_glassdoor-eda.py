!pip install plotly



!pip install seaborn

!pip install nltk

!pip install gensim

!pip install yellowbrick

#importing libraries

import pandas as pd 

import numpy as np

import nltk 

import plotly.express as px

import gensim

import gc

import string

import re

import yellowbrick

#import plotly.plotly as py

import plotly.graph_objs as go

#from plotly.offline import iplot, init_notebook_mode

from plotly.subplots import make_subplots



#cufflinks.go_offline(connected=True)

#init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline









pd.set_option('display.max_colwidth', None)

pd.options.display.max_columns = None
df_ny = pd.read_csv('../input/glassdoor-data-science-job-data/Data_Job_NY.csv')

df_sf = pd.read_csv('../input/glassdoor-data-science-job-data/Data_Job_SF.csv')

df_tx = pd.read_csv('../input/glassdoor-data-science-job-data/Data_Job_TX.csv')

df_wa = pd.read_csv('../input/glassdoor-data-science-job-data/Data_Job_WA.csv')

#Concatenating the data files



data_df = pd.concat([df_ny , df_sf , df_tx,df_wa] , axis = 0 , ignore_index = True)

del df_ny , df_sf , df_tx ,df_wa

gc.collect()
#Beginning the Cleaning and analysis of the data

#data_df.head(1)
#data_df.tail(1)
data_df.shape
data_df.info()
#First let's convert min_salary and max_salary columns to int

data_df['Min_Salary'] = data_df['Min_Salary'].apply(lambda x : int(x))

data_df['Max_Salary'] =data_df['Max_Salary'].apply(lambda x : int(x))
#Extracting date and day from Date_Posted : data is the format y-m-d

import calendar

data_df['Month'] = data_df['Date_Posted'].apply(lambda x : calendar.month_abbr[int(str(x).split('-')[1])]) 

#data_df['Month'] = data_df['Month'].apply(lambda x : calendar.month_abbr[int(x)])

def Convert_to_Day(x):

    sl = x.split('-')

    

    return calendar.day_abbr[int(calendar.weekday(int(sl[0]) , int(sl[1]) , int(sl[2])))]



data_df['Day'] = data_df['Date_Posted'].apply(lambda x : Convert_to_Day(x))
#While collecting the data if no salary is found I replaced the value by -1 so lets store that data in different data frame

index_missing = data_df[(data_df['Min_Salary'] == -1)].index

test_df = data_df.iloc[index_missing, :].reset_index(drop = True)

data_df.drop(index_missing , axis = 0 , inplace = True)

data_df = data_df.reset_index(drop = True)

#We will use this data as our test set.
#Now that we have train and test set there are duplicates in the data becasue our scraper was not perfect and could havea assimilated multiple entries

cols = [col for col in data_df.columns if col not in ['Day' , 'Month']]

#For training data 

train_series = data_df.duplicated(cols , keep = 'first')

data_df =data_df[~train_series].reset_index(drop = True)

test_series = test_df.duplicated(cols , keep = 'first')

test_df = test_df[~test_series].reset_index(drop = True)
#Unique States



print(data_df['State'].unique())
#Let's explore the top 5 cites in which most job lisitngs are there

for state in data_df['State'].unique():

    print(f"State of {state}")

    print(data_df[data_df['State'] == state]['City'].value_counts()[:5])



#Pie Chart of CA and NY



max_state = ['CA' , 'TX']

fig = make_subplots(rows = 1 , cols =2 , specs=[[{'type':'domain'}, {'type':'domain'}]])

for i,state in enumerate(max_state,1):

    cities = data_df[data_df['State'] == state]['City'].value_counts()[:5].index.to_list()

    counts = data_df[data_df["State"] == state]['City'].value_counts()[:5].to_list()

    fig.add_trace(go.Pie(labels = cities ,values = counts  ,name = state),1,i)

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_layout(

    title_text="States with most number of jobs",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text= 'CA', x=0.20, y=0.5, font_size=25, showarrow=False),

                 dict(text='TX', x=0.82, y=0.5, font_size=25, showarrow=False)])

fig.show()
#Dropping the states where value of number of jobs will be one as they'll be outliers

index = data_df[(data_df['State'] =='NC') | (data_df['State'] =='TN') | (data_df['State'] =='KY')].index

data_df.drop(index , inplace = True)
#Let's the avg minimal salaries for states 

import numpy as np

states = data_df['State'].unique().tolist()

fig = go.Figure()

min_sal =  data_df.groupby('State')['Min_Salary']

max_sal =  data_df.groupby('State')['Max_Salary']

fig.add_trace(go.Bar(x = states,

                    y = min_sal.mean(),

                    name = 'Min Salary' , marker_color = 'Magenta'))



fig.add_trace(go.Bar(x = states,

                    y = max_sal.mean(),

                    name = 'Max Salary' , marker_color = 'SkyBlue'))

fig.update_layout(template = 'ggplot2', barmode = 'group')

fig.show()

#Let's see avg minimal salaries according to top  5 cities 

states = ["CA",'TX','DC']

fig = make_subplots(rows = 3 , cols = 1,specs = [[{"type": "xy"}],[{"type": "xy"}],[{"type": "xy"}]])

colors = ['#2e9dd4' ,'#e76969' ,'#4fd882' ,'#f22dea' , '#e7468f']

for i,state in enumerate(states,1):



    cities = data_df[data_df['State'] == state]['City'].value_counts()[:5].index.to_list()

    avg_min_sals = []

    for city in cities:

        



        avg_min_sals.append(int(data_df[data_df['City'] == city]['Min_Salary'].mean()))

    fig.add_trace(go.Bar(x = cities , y = avg_min_sals  ,marker_color = colors ,name = state),i,1)

fig.update_layout(template = 'ggplot2' , title = "Average Minimal Salaries per city in states with most number of Jobs")

fig.show()





#Job Types in States with Max number of Jobs

for state in states:

    print(f"Type of Jobs in state of {state}")

    print(data_df[data_df['State'] == state]['Job_Type'].value_counts())
#Let's see the day on which most number of jobs are posted

day_fig = go.Figure([go.Bar(x = data_df['Day'].value_counts().index.to_list() ,

                    y = data_df['Day'].value_counts().to_list() , marker_color = 'skyblue')])

day_fig.update_layout(template = 'ggplot2' , title = 'Days with max number of jobs')
#Now's  let's explore the industry column

#This column has Nan Values



ind = data_df[~data_df['Industry'].isnull()]

print(f"Number of Unique Industries : {ind.Industry.nunique()}")
ind.Industry.value_counts()
#top 8 industries with max number of jobs



fig = go.Figure()

fig.add_traces(go.Pie(values = ind.Industry.value_counts()[:8].to_list(),

                    labels= ind.Industry.value_counts()[:8].index.to_list(),

                    name = 'Industry',textposition = 'inside' , textinfo = 'percent+label'))

fig.update_layout(template = 'plotly_white',title = 'Industries with most number of Data Science Related jobs' )

fig.show()
#Let's see which industries dominate the states 

for state in ind.State.unique():

    print(f"State of {state}")

    print(ind[ind['State'] == state]['Industry'].value_counts()[:8])





#Lets take a look at minimal average salary for the top 8 industries

fig = go.Figure()

fig.add_trace(go.Bar(x = ind.groupby("Industry")['Min_Salary'].mean().to_list(),

y = ind.groupby("Industry")['Min_Salary'].mean().index.to_list(), marker_color = 'goldenrod',

orientation = 'h' , name = "Min Avg Salary"

))

fig.add_trace(go.Bar(x = ind.groupby("Industry")['Max_Salary'].mean().to_list(),

y = ind.groupby("Industry")['Max_Salary'].mean().index.to_list(), marker_color = 'deepskyblue'

,orientation = 'h' ,name = "Max Avg Salary"))

fig.update_layout( template = 'plotly_dark',

    title = 'Minimal And Maximal Average Annual Salaries according to industries' ,barmode = 'group')

fig.show()
#Now let's explore companies 



print(f"Number of Unique Company Names : {data_df['Company'].nunique()}")

# Companies which have most number of job postings



fig = go.Figure()

fig.add_trace(go.Bar(y = data_df['Company'].value_counts()[:20].to_list(),

x= data_df['Company'].value_counts()[:20].index.to_list(),

marker_color = 'deepskyblue' , name = "Company"))

fig.update_layout(title= 'Companies with Max Number of Job Postings related to data science',

                template = 'plotly_dark')

fig.show()
#Let's take a look at Avg Minimal and Maximal salaries for companies 

def Plot_Company_salaries(companies,title):

    fig = go.Figure()

    min_sal = []

    max_sal = []

    for company in companies:

        min_sal.append(data_df[data_df['Company'] == company]['Min_Salary'].mean())

        max_sal.append(data_df[data_df['Company'] == company]['Max_Salary'].mean())







    fig.add_trace(go.Bar(x = min_sal ,y = companies , marker_color = 'deepskyblue' 

    , name  = 'Minimal Salary' , orientation = 'h'))

    fig.add_trace(go.Bar( x= max_sal,y = companies , marker_color = 'red' , 

    name = 'Maximal Salary', orientation = 'h'))



    fig.update_layout(title = title,

    barmode = 'group' , template = 'plotly_dark')

    fig.show()

    

    
#Top 5 companies in CA

states = ['CA' ,'TX' ,'DC' ,'MD' ,'VA']

companies = []

titles = []

for state in states:

    companies.append(data_df[data_df['State'] == state]['Company'].value_counts()[:5].index.to_list())

    titles.append(f'{state} : Minimal And Maximal Annual Average Salaries for top 5 companies')



for i in range(len(states)):

    Plot_Company_salaries(companies[i] , titles[i])

#Distribution of ratings of companies



ratings =data_df[~data_df['Rating'].isnull()]['Rating']

sns.distplot(ratings,kde = True , rug = True)

plt.axvline(np.median(ratings),color='r', linestyle='--')

plt.grid(True)

plt.title("Distribution of Ratings")

plt.show()
#Minimal Salaries distribution

sns.distplot(data_df['Min_Salary'] , kde = True , rug = True)

plt.axvline(np.median(data_df['Min_Salary']),color='r', linestyle='--')

plt.axvline(np.mean(data_df['Min_Salary']),color='g', linestyle='--')

plt.grid(True)

plt.title("Distribution of minimal Salaries")

plt.show()



#Box plot for minimal salaries

fig = px.box(data_df , y = 'Min_Salary' ,points = 'all')

fig.show()
#Maximal Salaries distribution

sns.distplot(data_df['Max_Salary'] , kde = True , rug = True)

plt.axvline(np.median(data_df['Max_Salary']),color='r', linestyle='--')

plt.axvline(np.mean(data_df['Max_Salary']),color='g', linestyle='--')

plt.grid(True)

#plt.figure(figsize=(100,100))

plt.title("Distribution of Maximum Salaries")

plt.show()
#Box plot for maximal salaries

fig = px.box(data_df , y = 'Max_Salary' ,points = 'all')

fig.show()
#unique Job titles

data_df['Job_title'].nunique()
#Top 8 job titles with max jobs

fig = go.Figure()

fig.add_traces(go.Pie(values = data_df.Job_title.value_counts()[:8].to_list(),

                    labels= data_df.Job_title.value_counts()[:8].index.to_list(),

                    name = 'Job Title',textposition = 'inside' , textinfo = 'percent+label'))

fig.update_layout(template = 'plotly_white',title = 'Job Titles with most number of  jobs',

                showlegend = False )

fig.show()

titles = ['Data Scientist' ,'Data Analyst' ,'Data Engineer']

min_sal = []

max_sal = []

for title in titles:

    min_sal.append(data_df[data_df['Job_title'] == title]['Min_Salary'].mean())

    max_sal.append(data_df[data_df['Job_title'] == title]['Max_Salary'].mean())



fig = go.Figure()

fig.add_trace(go.Bar(x = min_sal ,y = titles , marker_color = 'deepskyblue',

orientation = 'h' , name = 'Min Salary'))

fig.add_trace(go.Bar(x = max_sal ,y = titles , marker_color = 'magenta',

orientation = 'h' , name = 'Max Salary'))

fig.update_layout(title = 'Annual Avergae Salaries for Job titles having most jobs',

barmode = 'group' ,template = 'plotly_white')

fig.show()
#Let's See how the description actually looks

x = data_df.Job_Desc[0].replace('\n\n' , '\n')

x = x.split('\n')



print(*x , sep = '\n')
#Let's clean \n

data_df['Job_Desc'] = data_df['Job_Desc'].replace('\n\n' , " " , regex = True)

data_df['Job_Desc'] = data_df['Job_Desc'].replace('\n' , " " , regex = True)



test_df['Job_Desc'] = test_df['Job_Desc'].replace('\n\n' , " " , regex = True)

test_df['Job_Desc'] = test_df['Job_Desc'].replace('\n' , " " , regex = True)

#Let's remove punctuation and Stopwords



from gensim.parsing.preprocessing import remove_stopwords

def Remove_puncutations_stopwords(s):



    s = ''.join([i for i in s if i not in string.punctuation])

    s = remove_stopwords(s)

    return s



data_df['Job_Desc'] = data_df['Job_Desc'].apply(lambda x : Remove_puncutations_stopwords(x))



test_df['Job_Desc'] = test_df['Job_Desc'].apply(lambda x : Remove_puncutations_stopwords(x))
#Let's try to visualize counts of the tokens

from sklearn.feature_extraction.text import CountVectorizer

from yellowbrick.text import FreqDistVisualizer



vec = CountVectorizer(min_df= 2 , stop_words = 'english' , ngram_range = (2,2))

docs = vec.fit_transform(data_df.Job_Desc)

features = vec.get_feature_names()



visualizer = FreqDistVisualizer(features=features, orient='h' , size = (800,800))

visualizer.fit(docs)

visualizer.show()
#Let's test it out for job_titles



vec_title = CountVectorizer(min_df= 2 , stop_words = 'english' , ngram_range = (2,2))

docs_titles = vec.fit_transform(data_df.Job_title)

features_title = vec.get_feature_names()



visualizer = FreqDistVisualizer(features=features_title, orient='h' , size = (800,800))

visualizer.fit(docs_titles)

visualizer.show()
#Now let's take average of minimal and maximal salary find its median



data_df['avg_sal'] = (data_df['Min_Salary'] + data_df['Max_Salary'])//2

#Median avg annual salary

print(f"Median average annual salary is {data_df['avg_sal'].median()}")

median_sal = data_df['avg_sal'].median()

data_df['is_higher'] = [1 if i > median_sal else 0 for i in data_df.avg_sal]

data_df.to_csv("train_data.csv" , index = False)

test_df.to_csv('test_data.csv' , index = False)