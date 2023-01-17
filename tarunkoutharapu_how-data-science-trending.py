import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from plotly.subplots import make_subplots

import seaborn as sns

import plotly.offline as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as ply

import plotly.graph_objs as go

import plotly.express as px

import folium 

from folium import plugins

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#Importing the 2019,2018,2017 Datasets

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')



df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')



df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')

 #df_2017.head()

# df_2018.head()

df_2019.head()
df_2019.columns =df_2019.iloc[0]

df_2019.drop([0], inplace =True)



df_2018.columns =df_2018.iloc[0]

df_2018.drop([0], inplace =True)



def return_count(data,question_part):

    """Counts occurences of each value in a given column"""

    counts_df = data[question_part].value_counts().to_frame()

    return counts_df



def return_percentage(data,question_part):

    """Calculates percent of each value in a given column"""

    total = data[question_part].count()

    counts_df= data[question_part].value_counts().to_frame()

    percentage_df = (counts_df*100)/total

    return percentage_df





    

def plot_graph(data,question,title,x_axis_title,y_axis_title):

    """ plots a percentage bar graph"""

    df = return_percentage(data,question)

    

    trace1 = go.Bar(

                    x = df.index,

                    y = df[question],

                    #orientation='h',

                    marker = dict(color='dodgerblue',

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500,

                       xaxis=dict(type='category',categoryorder='array',categoryarray=salary_order,title=y_axis_title),

                       yaxis= dict(title=x_axis_title))

                       

    fig = go.Figure(data = data, layout = layout)

    iplot(fig) 
df_all_surveys = pd.DataFrame(data = [len(df_2017),len(df_2018),len(df_2019)],

                          columns = ['Number of responses'], index = ['2017','2018','2019'])

df_all_surveys.index.names = ['Year of Survey']





x = df_all_surveys['Number of responses'].index

y = df_all_surveys['Number of responses'].values





# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=['Year 2017','Year 2018','Year 2019'],

            y=y,

            text=y,

            width=0.5,

            textposition='auto',

            marker=dict(color='orangered')

 )])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.update_layout(yaxis=dict(title='Number of Participants'),width=700,height=500,

                  title='Total number of respondents over last 3 years',

                  xaxis=dict(title='Year'))

fig.show()
def plot_graph(data,question,title,x_axis_title):

    df = return_percentage(data,question)

    

    trace1 = go.Bar(

                    y = df.index,

                    x = df[question][0:20],

                    orientation='h',

                    marker = dict(color='lawngreen',line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500, 

                       xaxis= dict(title=x_axis_title),

                       yaxis=dict(autorange="reversed"),

                       showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    fig.show()

     

response_count = df_2019['In which country do you currently reside?'].count()

plot_graph(df_2019,'In which country do you currently reside?','Top 20 countries of respondents in 2019','Percentage of Respondents')
plot_graph(df_2018,'In which country do you currently reside?','Top 20 countries of respondents in 2018','Percentage of Respondents')
plot_graph(df_2017,'Country','Top 20 countries of respondents','Percentage of Respondents')
country_dist=df_2019.iloc[:,4].value_counts()

fig = px.choropleth(country_dist.values, locations=country_dist.index,

                    locationmode='country names',

                    color=country_dist.values,

                    color_continuous_scale=px.colors.sequential.OrRd)

fig.update_layout(title="Countrywise Distribution of data science Participatens in 2019")

fig.show()
country_dist=df_2018.iloc[:,4].value_counts()

fig = px.choropleth(country_dist.values, locations=country_dist.index,

                    locationmode='country names',

                    color=country_dist.values,

                    color_continuous_scale=px.colors.sequential.OrRd)

fig.update_layout(title="Countrywise Distribution of data science Participatens in 2018")

fig.show()
country_dist=df_2017.iloc[:,1].value_counts()

fig = px.choropleth(country_dist.values, locations=country_dist.index,

                    locationmode='country names',

                    color=country_dist.values,

                    color_continuous_scale=px.colors.sequential.OrRd)

fig.update_layout(title="Countrywise Distribution of data science Participatens in 2017")

fig.show()
df_2019['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)





df_2018['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)

df_2018['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)







# Splitting all the datasets genderwise

male_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Male']

female_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Female']



male_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Male']

female_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Female']



male_2017 = df_2017[df_2017['GenderSelect']=='Male']

female_2017 = df_2017[df_2017['GenderSelect']=='Female']





# Top-10 Countries with Respondents in 2019

topn = 10

count_male = male_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()

count_female = female_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()



pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.4,domain={'x': [0,0.46]})

pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Top-10 Countries with Respondents in 2019', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_men, pie_women], layout=layout)

py.iplot(fig)
def get_name(code):

    '''

    Translate code to name of the country

    '''

    try:

        name = pycountry.countries.get(alpha_3=code).name

    except:

        name=code

    return name



country_number = pd.DataFrame(female_2019['In which country do you currently reside?'].value_counts())

country_number['country'] = country_number.index

country_number.columns = ['number', 'country']

country_number.reset_index().drop(columns=['index'], inplace=True)

country_number['country'] = country_number['country'].apply(lambda c: get_name(c))

country_number.head(5)







worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',

                 z = country_number['number'], colorscale = "Blues", reversescale = True, 

                 marker = dict(line = dict( width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of respondents'))]



layout = dict(title = 'The Nationality of Female Respondents in 2019', geo = dict(showframe = False, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

py.iplot(fig, validate=False)
colors1 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 

counts = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)

labels = counts.index

values = counts.values



pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors1,line=dict(color='#000000', width=1)))

layout = go.Layout(title='Gender Distribution in 2019')



fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
colors2 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 

gender_count_2019 = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)

gender_count_2018 = df_2018['What is your gender? - Selected Choice'].value_counts(sort=True)

gender_count_2017 = df_2017['GenderSelect'].value_counts(sort=True)





labels = ["Male ", "Female", "Prefer not to say ", "Prefer to self-describe"]

labels1 = ["Male ", "Female","A different identity", "Non-binary","genderqueer, or gender non-conforming"]

# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels1, values=gender_count_2017.values, name="2017",marker=dict(colors=colors2)),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=gender_count_2018.values, name="2018",marker=dict(colors=colors2)),

              1, 2)

fig.add_trace(go.Pie(labels=labels, values=gender_count_2019.values, name="2019",marker=dict(colors=colors2)),

              1, 3)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.5, hoverinfo="label+percent+name")



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.data[1].marker.line.width = 1

fig.data[1].marker.line.color = "black"

fig.data[2].marker.line.width = 1

fig.data[2].marker.line.color = "black"



fig.update_layout(

    title_text="Gender Distribution over the years",font=dict(size=12), legend=dict(orientation="h"),

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='2017', x=0.11, y=0.5, font_size=20, showarrow=False),

                 dict(text='2018', x=0.5, y=0.5, font_size=20, showarrow=False),

                 dict(text='2019', x=0.88, y=0.5, font_size=20, showarrow=False)])

fig.show()
female_country_2019 = female_2019['In which country do you currently reside?']

female_country_2018 = female_2018['In which country do you currently reside?']

female_country_2017 = female_2017['Country']

                                                                  

f_2019 = female_country_2019[(female_country_2019 == 'India') | (female_country_2019 == 'United States')].value_counts()

f_2018 = female_country_2018[(female_country_2018 == 'India') | (female_country_2018 == 'United States')].value_counts()

f_2017 = female_country_2017[(female_country_2017 == 'India') | (female_country_2017 == 'United States')].value_counts()                                                                  

                                         

female_country_count = pd.DataFrame(data = [f_2017,f_2018,f_2019],index = ['2017','2018','2019'])    



female_country_count['total'] = [len(female_2017),len(female_2018),len(female_2019)]

female_country_count['US%'] = female_country_count['United States']/female_country_count['total']*100

female_country_count['India%'] = female_country_count['India']/female_country_count['total']*100



female_country_count[['India%','US%']].plot(kind='bar',color=['dodgerblue','skyblue'],linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Pattern of US and Indian Female respondents over the years', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Year of Survey',fontsize=15)

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left",labels=['India','US'])

plt.show()
df_2017['Age in years'] = pd.cut(x=df_2017['Age'], bins=[18,21,25,29,34,39,44,49,54,59,69,79], 

                                                        labels=['18-21',

                                                                '22-24',

                                                                '25-29',

                                                                '30-34',

                                                                '35-39',

                                                                '40-44',

                                                                '45-49',

                                                                '50-54',

                                                                '55-59',

                                                                '60-69',

                                                                '70+'])

                                                                                                  





x = df_2017['Age in years'].value_counts()

y = df_2018['What is your age (# years)?'].value_counts()

z = df_2019['What is your age (# years)?'].value_counts()





w = pd.DataFrame(data = [x,y,z],index = ['2017','2018','2019'])

w.fillna(0,inplace=True)



w.loc['2017'] = w.loc['2017']/len(df_2017)*100

w.loc['2018'] = w.loc['2018']/len(df_2018)*100

w.loc['2019'] = w.loc['2019']/len(df_2019)*100



w.T[['2019']].plot(subplots=True, layout=(1,1),kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(10,8)

plt.title('Age wise Distribution of  Respondents in 2019',fontsize=15)

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of  Respondents',fontsize=15)

plt.show()
w.T[['2018']].plot(subplots=True,layout=(1,2),color='dodgerblue',kind='bar',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(15,6)

#plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)

#plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

#plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of  Respondents',fontsize=15)

plt.show()
w.T[['2017']].plot(subplots=True,layout=(1,2),color='dodgerblue',kind='bar',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(15,6)

#plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)

#plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

#plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of  Respondents',fontsize=15)

plt.show()
female_2017['Age in years'] = pd.cut(x=female_2017['Age'], bins=[18,21,25,29,34,39,44,49,54,59,69,79], 

                                                        labels=['18-21',

                                                                '22-24',

                                                                '25-29',

                                                                '30-34',

                                                                '35-39',

                                                                '40-44',

                                                                '45-49',

                                                                '50-54',

                                                                '55-59',

                                                                '60-69',

                                                                '70+'])

                                                                                                  





x = female_2017['Age in years'].value_counts()

y = female_2018['What is your age (# years)?'].value_counts()

z = female_2019['What is your age (# years)?'].value_counts()





w = pd.DataFrame(data = [x,y,z],index = ['2017','2018','2019'])

w.fillna(0,inplace=True)



w.loc['2017'] = w.loc['2017']/len(female_2017)*100

w.loc['2018'] = w.loc['2018']/len(female_2018)*100

w.loc['2019'] = w.loc['2019']/len(female_2019)*100



w.T[['2019']].plot(subplots=True, layout=(1,1),kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(10,8)

plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)

plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.show()
w.T[['2017','2018']].plot(subplots=True,layout=(1,2),color='dodgerblue',kind='bar',linewidth=1,edgecolor='k',legend=False)

plt.gcf().set_size_inches(15,6)

#plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)

#plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

#plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.show()