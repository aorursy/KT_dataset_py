#importing libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from chart_studio.plotly import iplot

import base64

import io

from scipy.misc import imread

import codecs

from IPython.display import HTML

from matplotlib_venn import venn2

from subprocess import check_output
#importing 2019 dataset

response19=pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',encoding='ISO-8859-1')

response19.columns = response19.iloc[0]

response19=response19.drop([0])





#importing 2018 dataset

response18=pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

response18.columns = response18.iloc[0]

response18=response18.drop([0])



#importing 2017 dataset



response17=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')



# Helper functions



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

    

response19.head()
#Basic Interpretations



print('The total number of respondents:',response19.shape[0])

print('Total number of Countries with respondents:',response19['In which country do you currently reside?'].nunique())

print('Country with highest respondents:',response19['In which country do you currently reside?'].value_counts().index[0],'with',response19['In which country do you currently reside?'].value_counts().values[0],'respondents')
#Gender Split



plt.subplots(figsize=(14,7))

sns.countplot(y=response19['What is your gender? - Selected Choice'],order=response19['What is your gender? - Selected Choice'].value_counts().index)

plt.title("Gender Split - 2019 Kaggle Survey")

plt.show()
plt.subplots(figsize=(14,7))

sns.countplot(y=response18['What is your gender? - Selected Choice'],order=response18['What is your gender? - Selected Choice'].value_counts().index)

plt.title("Gender split - 2018 Kaggle Suvey")

plt.show()
plt.subplots(figsize=(15,5))

sns.countplot(y=response17['GenderSelect'],order=response17['GenderSelect'].value_counts().index)

plt.title("Gender Split - 2017 Kaggle survey")

plt.show()
# Replacing the ambigious countries name with Standard names



response19['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)





response18['In which country do you currently reside?'].replace(

                                                   {'United States of America':'United States',

                                                    'Viet Nam':'Vietnam',

                                                    "People 's Republic of China":'China',

                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',

                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)



# Splitting all the datasets genderwise



male_2019 = response19[response19['What is your gender? - Selected Choice']=='Male']

female_2019 = response19[response19['What is your gender? - Selected Choice']=='Female']



male_2018 = response18[response18['What is your gender? - Selected Choice']=='Male']

female_2018 = response18[response18['What is your gender? - Selected Choice']=='Female']



male_2017 = response17[response17['GenderSelect']=='Male']

female_2017 = response17[response17['GenderSelect']=='Female']



# Top-10 Countries with Respondents in 2019



topn = 10

count_male = male_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()

count_female = female_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()



pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.4,domain={'x': [0,0.46]})

pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Top-10 Countries with Respondents in 2019', font=dict(size=10), legend=dict(orientation="h"),

              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),

                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])





#Female age distribution



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
#Country wise distribution 



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

                 z = country_number['number'], colorscale = "Greens", reversescale = True, 

                 marker = dict(line = dict( width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of respondents'))]



layout = dict(title = 'Female Respondents Distributions over world in 2019', geo = dict(showframe = False, showcoastlines = True, 

                                                                projection = dict(type = 'Mercator')))



fig = dict(data=worldmap, layout=layout)

py.iplot(fig, validate=False)
#Female over the years



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
#Obstacles over african countries females



African_2019 = female_country_2019[(female_country_2019 == 'Algeria') | 

                                   (female_country_2019 == 'Nigeria') |

                                   (female_country_2019 == 'Egypt')   |

                                   (female_country_2019 == 'Kenya')   |

                                   (female_country_2019 == 'South Africa')].value_counts()



African_2018 = female_country_2018[(female_country_2018 == 'Algeria') | 

                                   (female_country_2018 == 'Nigeria') |

                                   (female_country_2018 == 'Egypt')   |

                                   (female_country_2018 == 'Kenya')   |

                                   (female_country_2018 == 'South Africa')].value_counts()



African_2017 = female_country_2017[(female_country_2017 == 'Algeria') | 

                                   (female_country_2017 == 'Nigeria') |

                                   (female_country_2017 == 'Egypt')   |

                                   (female_country_2017 == 'Kenya')   |

                                   (female_country_2017 == 'South Africa')].value_counts()

African_subcontinent_count = pd.DataFrame(data = [African_2017,African_2018,African_2019],index = ['2017','2018','2019']) 





African_subcontinent_count.fillna(0,inplace=True)

African_subcontinent_count.loc[:,'Sum'] = African_subcontinent_count.sum(axis=1)





x = African_subcontinent_count['Sum'].index

y = African_subcontinent_count['Sum'].values



# Use textposition='auto' for direct text

fig1 = go.Figure(data=[go.Bar(

            x=['Year 2017','Year 2018','Year 2019'],

            y=y,

            text=y,

            width=0.4,

            textposition='auto',

            marker=dict(color='dodgerblue'))])



fig1.data[0].marker.line.width = 1

fig1.data[0].marker.line.color = "black"

fig1.update_layout(yaxis=dict(title='Number of Female Respondents'),

                   title='Total African Females respondents over the years',width=800,height=500,

                   xaxis=dict(title='Years'))



#Africal country wise respondents count



countries = ['South Africa','Egypt','Kenya','Nigeria','Algeria']

colors = ["steelblue","dodgerblue","lightskyblue","deepskyblue","darkturquoise","paleturquoise","turquoise"]



African_subcontinent_count[countries].plot(kind='bar',color=colors,linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Country wise Female respondents from Africa ', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Years',fontsize=15)

plt.ylabel('Number of Female Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left")

plt.show()
#Education qualification of female respondents



import textwrap

from  textwrap import fill



x_axis=range(7)

qualification = female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()

qualification = qualification/len(female_2019)*100

labels = qualification.index



qualification.plot(kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=None)

plt.gcf().set_size_inches(10,8)

plt.title('Educational Qualifications of the Females respondents in 2019', fontsize = 15)

plt.xticks(x_axis, [textwrap.fill(label, 10) for label in labels], 

           rotation = 0, fontsize=12, horizontalalignment="right")

#plt.xlabel('Education Qualification',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.xlabel('Qualification',fontsize=15)

plt.show()
!pip install wordcloud
#Roles of female in the field of data science



from wordcloud import WordCloud

female_title_2017 = female_2017['CurrentJobTitleSelect'].dropna()

female_title_2018 = female_2018['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].dropna()

female_title_2019 = female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].dropna()



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[26, 8])

wordcloud1 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(female_title_2017))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Job Titles in 2017',fontsize=20);



wordcloud2 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(female_title_2018))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Job Titles in 2018',fontsize=20);



wordcloud3 = WordCloud( background_color='white',

                        width=600,

                        height=400).generate(" ".join(female_title_2019))

ax3.imshow(wordcloud3)

ax3.axis('off')

ax3.set_title('Job Titles in 2019',fontsize=20);
#Country wise job roles of female



df_edu_temp = pd.crosstab(female_2019['In which country do you currently reside?'],

              female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])





df_edu = df_edu_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')

                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "China")

                    |(df_edu_temp.index == 'United Kingdom')].drop('I prefer not to answer',axis=1)







df_roles_temp = pd.crosstab(female_2019['In which country do you currently reside?'],

                            female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'])

df_roles_temp['Data Engineer'] = df_roles_temp['DBA/Database Engineer']+df_roles_temp['Data Engineer']

df_roles_temp['Data/Business Analyst'] = df_roles_temp['Data Analyst']+df_roles_temp['Business Analyst']

df_roles = df_roles_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')

                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "China")

                    |(df_edu_temp.index == 'United Kingdom')].drop(['Other','DBA/Database Engineer','Data Engineer','Data Analyst','Business Analyst'],axis=1)





df_roles = (df_roles/len(female_2019))*100



ax = df_roles.plot(kind='bar',width=1,cmap='tab20',linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(20,8)

plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.title('Country wise Job Role Distribution of Females in 2019', fontsize = 15)

#plt.xlabel('Countries',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')

plt.yticks( fontsize=10)

plt.legend(fontsize=15,bbox_to_anchor=(0.5, -0.22), loc="upper center",ncol=6)

plt.show()

#Percentage of female scientists over the years



def return_percentage1(data,question_part):

    """Calculates percent of each value in a given column"""

    total = data[question_part].count()

    counts_df= data[question_part].value_counts()

    percentage_df = (counts_df*100)/total

    return percentage_df







female_DS_2017 = return_percentage1(female_2017,'CurrentJobTitleSelect').loc['Data Scientist']

female_DS_2018 = return_percentage1(female_2018,'Select the title most similar to your current role (or most recent title if retired): - Selected Choice').loc['Data Scientist']

female_DS_2019 = return_percentage1(female_2019,'Select the title most similar to your current role (or most recent title if retired): - Selected Choice').loc['Data Scientist']

ds = pd.DataFrame(data = [female_DS_2017,female_DS_2018,female_DS_2019],

                          columns = ['Percentage of total roles'], index = ['2017','2018','2019'])

ds.round(1)

ds.index.names = ['Year of Survey']



x = ds['Percentage of total roles'].index

y = np.round(ds['Percentage of total roles'].values,1)





# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=['Year 2017','Year 2018','Year 2019'],

            y=y,

            text=y,

            width=0.4,

            textposition='auto',

            marker=dict(color='dodgerblue')

 )])



fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "black"

fig.update_layout(yaxis=dict(title='Percentage of Female Respondents'),width=700,height=500,

                  title='Female Data Scientists in the survey',xaxis=dict(title='Years'))

fig.show()