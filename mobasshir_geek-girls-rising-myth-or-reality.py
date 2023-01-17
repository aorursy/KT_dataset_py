# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Graphics in retina format are more sharp and legible

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

plt.rcParams['image.cmap'] = 'viridis'





import plotly.offline as py

import pycountry



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from plotly.offline import init_notebook_mode, iplot 

init_notebook_mode(connected=True)



import folium 

from folium import plugins

#Importing the 2019 Dataset

df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

df_2019.columns = df_2019.iloc[0]

df_2019=df_2019.drop([0])



df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
print('Total respondents in 2019:',df_2019.shape[0])



male_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Male']

female_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Female']

female_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Female']

female_2017 = df_2017[df_2017['GenderSelect']=='Female']







df_2019['In which country do you currently reside?'].replace({'United States of America':'United States','Viet Nam':'Vietnam','China':"People 's Republic of China","United Kingdom of Great Britain and Northern Ireland":'United Kingdom',"Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)



topn = 10



count_male = male_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()

count_female = female_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()



pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.5,domain={'x': [0,0.46]})

pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Top-10 countries with respondents', font=dict(size=12), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])



fig = dict(data=[pie_men, pie_women], layout=layout)

py.iplot(fig)
colors = ['#1BA1E2', '#AA00FF', '#F0A30A','#8c564b'] #gold,bronze,silver,chestnut brown

counts = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)

labels = counts.index

values = counts.values



pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors))

layout = go.Layout(title='Gender Distribution in 2019')

fig = go.Figure(data=[pie], layout=layout)

py.iplot(fig)
from plotly.subplots import make_subplots



counts1 = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)

counts2 = df_2018['What is your gender? - Selected Choice'].value_counts(sort=True)

counts3 = df_2017['GenderSelect'].value_counts(sort=True)





labels = ["Male ", "Female", "Prefer not to say ", "Prefer to self-describe"]

labels3 = ["Male ", "Female","A different identity", "Non-binary","genderqueer, or gender non-conforming"]

# Create subplots: use 'domain' type for Pie subplot

fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])

fig.add_trace(go.Pie(labels=labels3, values=counts3.values, name="2017"),

              1, 1)

fig.add_trace(go.Pie(labels=labels, values=counts2.values, name="2018"),

              1, 2)

fig.add_trace(go.Pie(labels=labels, values=counts1.values, name="2019"),

              1, 3)

# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.5, hoverinfo="label+percent+name")



fig.update_layout(

    title_text="Gender Distribution over the years",font=dict(size=12), legend=dict(orientation="h"),

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='2017', x=0.11, y=0.5, font_size=20, showarrow=False),

                 dict(text='2018', x=0.5, y=0.5, font_size=20, showarrow=False),

                 dict(text='2019', x=0.88, y=0.5, font_size=20, showarrow=False)])

fig.show()
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

                 z = country_number['number'], autocolorscale = True, reversescale = False, 

                 marker = dict(line = dict(color = 'rgb(100,100,100)', width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of respondents'))]



layout = dict(title = 'The Nationality of Female Respondents', geo = dict(showframe = False, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

py.iplot(fig, validate=False)
def return_percentage(data,question_part,response_count):

    """Calculates percent of each value in a given column"""

    counts = data[question_part].value_counts()

    total = response_count

    percentage = (counts*100)/total

    value = [percentage]

    question = [data[question_part]][0]

    percentage_df = pd.DataFrame(data=value).T     

    return percentage_df





def plot_multiple_choice(data,question,title,x_axis_title):

    df = return_percentage(data,question,response_count)

    

    trace1 = go.Bar(

                    y = df.index,

                    x = df[question][0:20],

                    orientation='h',

                    name = "Kaggle Survey 2019",

                    marker = dict(color='#00C9E0',

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=1000, height=500, 

                       xaxis= dict(title=x_axis_title),

                       yaxis=dict(autorange="reversed"),

                       showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)

    

response_count = len(female_2019['In which country do you currently reside?'])

plot_multiple_choice(female_2019,'In which country do you currently reside?','Top 20 countries of female respondents','Percentage of Respondents')

    
female_2018['In which country do you currently reside?'].replace({'United States of America':'United States','Viet Nam':'Vietnam',

                                                                  'China':"People 's Republic of China",

                                                                  "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                                  "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)

female_2019['In which country do you currently reside?'].replace({'United States of America':'United States','Viet Nam':'Vietnam',

                                                                  'China':"People 's Republic of China",

                                                                  "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                                  "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)







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



female_country_count[['India%','US%']].plot(kind='bar',cmap='tab10')

plt.gcf().set_size_inches(10,8)

plt.title('Pattern of US and Indian Female respondents over the years', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Years',fontsize=15)

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left",labels=['India','US'])



plt.figure(figsize=(10,8))

sns.countplot(x="What is your age (# years)?", data=female_2019,palette ='Blues',order = df_2019['What is your age (# years)?'].value_counts().index.sort_values())

plt.title('Age wise Distribution of Female Respondents',fontsize=15)

plt.xticks( rotation=45, fontweight='bold', fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Count',fontsize=15)

## Grouping the Ages

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['18-21']),'18-21','')

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['25-29','22-24']),'22-29',df_2019['Age Group'])

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['30-34','35-39']),'30-39',df_2019['Age Group'])

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['40-44','45-49']),'40-49',df_2019['Age Group'])

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['50-54','55-59']),'50-59',df_2019['Age Group'])

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['60-69']),'60-69',df_2019['Age Group'])

df_2019['Age Group']=np.where(df_2019['What is your age (# years)?'].isin(['70+']),'70s and above',df_2019['Age Group'])



count_age=df_2019.groupby(['In which country do you currently reside?','Age Group'])['What is your age (# years)?'].count().reset_index()

count_age.columns=['Country','Age Group','Count']

count_age=count_age[count_age['Country'].isin(df_2019['In which country do you currently reside?'].value_counts()[:10].index)]

count_age=count_age[count_age['Country']!='Other']

count_age.pivot('Country','Age Group','Count').plot.bar(stacked=True,width=0.8)

plt.gcf().set_size_inches(16,8)

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.title('Country wise Age Distribution', fontsize = 15)

plt.xlabel('Countries',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")

plt.show()
from wordcloud import WordCloud

female_title = female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].dropna()

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='black',

                          width=512,

                          height=384

                         ).generate(" ".join(female_title))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
fig = plt.figure()

plt.figure(figsize=(10,8))

sns.countplot(x="What is the highest level of formal education that you have attained or plan to attain within the next 2 years?", data=female_2019,palette="Set3",

             order = female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().index)

plt.xticks(rotation=90)

plt.title("Females'Formal Education level",fontsize=15)

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.xlabel('Age in years',fontsize=15)

plt.ylabel('Count',fontsize=15)

df_edu_temp = pd.crosstab(female_2019['In which country do you currently reside?'],

              female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])





df_edu = df_edu_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')

                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "People 's Republic of China")

                    | (df_edu_temp.index == 'United Kingdom')].drop('I prefer not to answer',axis=1)



df_edu.plot(kind='bar',width=1)

plt.gcf().set_size_inches(16,8)

plt.title('Country wise Age Distribution', fontsize = 15)

plt.xlabel('Countries',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")

plt.show()
plt.figure(figsize=(8,8))

ax=female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()[:10].plot.barh(width=0.9,color=sns.color_palette('Set3',25))

plt.gca().invert_yaxis()

plt.title('Current Roles2019')

plt.show()


df_roles_temp = pd.crosstab(female_2019['In which country do you currently reside?'],

            female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'])

df_roles_temp['Data Engineer'] = df_roles_temp['DBA/Database Engineer']+df_roles_temp['Data Engineer']

df_roles_temp['Data/Business Analyst'] = df_roles_temp['Data Analyst']+df_roles_temp['Business Analyst']

df_roles = df_roles_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')

                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "People 's Republic of China")

                    | (df_edu_temp.index == 'United Kingdom')].drop(['Other','DBA/Database Engineer','Data Engineer','Data Analyst','Business Analyst'],axis=1)





ax = df_roles.plot(kind='bar',width=1)

plt.gcf().set_size_inches(20,8)

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.title('Country wise Age Distribution', fontsize = 15)

plt.xlabel('Countries',fontsize=15)

plt.ylabel('Count',fontsize=15)

plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")

plt.show()
plt.figure(figsize=(8,8))

female['What is the size of the company where you are employed?'].value_counts()[:10].plot.barh(width=0.9,color=sns.color_palette('Set3',25))

plt.gca().invert_yaxis()

plt.title('Size of Company')

plt.show()