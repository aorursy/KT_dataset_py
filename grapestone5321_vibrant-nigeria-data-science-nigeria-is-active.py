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

%matplotlib inline

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



#Importing the 2018 Dataset

df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

df_2018.columns = df_2018.iloc[0]

df_2018=df_2018.drop([0])



#Importing the 2017 Dataset

df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
male_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Male']

female_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Female']

female_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Female']

female_2017 = df_2017[df_2017['GenderSelect']=='Female']





df_2019['In which country do you currently reside?'].replace({'United States of America':'United States','Viet Nam':'Vietnam','China':"People 's Republic of China","United Kingdom of Great Britain and Northern Ireland":'United Kingdom',"Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)



topn = 15

count_male = male_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()

count_female = female_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()



pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.3,domain={'x': [0,0.46]})

pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})



layout = dict(title = 'Top-15 countries with respondents', font=dict(size=10), legend=dict(orientation="h"),

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

                 z = country_number['number'], colorscale = "Reds", reversescale = True, 

                 marker = dict(line = dict( width = 0.5)), 

                 colorbar = dict(autotick = False, title = 'Number of respondents'))]



layout = dict(title = 'The Nationality of Female Respondents in 2019', geo = dict(showframe = False, showcoastlines = True, 

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





def plot_graph(data,question,title,x_axis_title):

    df = return_percentage(data,question,response_count)

    

    trace1 = go.Bar(

                    y = df.index,

                    x = df[question][0:20],

                    orientation='h',

                    name = "Kaggle Survey 2019",

                    marker = dict(color='salmon',

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500, 

                       xaxis= dict(title=x_axis_title),

                       yaxis=dict(autorange="reversed"),

                       showlegend=False)

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)

    

response_count = len(female_2019['In which country do you currently reside?'])

plot_graph(female_2019,'In which country do you currently reside?','Top 20 countries of female respondents','Percentage of Female Respondents')

    
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



female_country_count[['India%','US%']].plot(kind='bar',cmap='tab20b',linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Pattern of US and Indian Female respondents over the years', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Year of Survey',fontsize=15)

plt.ylabel('Percentage of Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left",labels=['India','US'])

plt.show()
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

African_subcontinent_count['Total_Females'] = [len(female_2017),len(female_2018),len(female_2019)]

African_subcontinent_count['Africa'] = African_subcontinent_count['Sum']/African_subcontinent_count['Total_Females']*100



African_subcontinent_count['Africa'].plot(kind='bar',color='r',linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Total African Females respondents over the years', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Years',fontsize=15)

plt.ylabel('Percentage of Female Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left",labels=['Africa'])

plt.show()
countries = ['South Africa','Egypt','Kenya','Nigeria','Algeria']



African_subcontinent_count[countries].plot(kind='bar',cmap='Reds',linewidth=1,edgecolor='k')

plt.gcf().set_size_inches(10,8)

plt.title('Country wise Female respondents from Africa ', fontsize = 15)

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.xlabel('Years',fontsize=15)

plt.ylabel('Number of Respondents',fontsize=15)

plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left")

plt.show()
df = pd.DataFrame(data = [len(df_2017),len(df_2018),len(df_2019)],

                          columns = ['Number of responses'], index = ['2017','2018','2019'])

df.index.names = ['Year of Survey']



df.plot(kind='bar',color='r',legend=False,linewidth=1,edgecolor='k')

plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')

plt.ylabel('Number of Respondents',fontsize=15)

plt.show()