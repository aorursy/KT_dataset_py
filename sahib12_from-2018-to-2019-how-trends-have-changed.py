# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graphing capabilities

from beautifultext import BeautifulText as bt # utility script

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import seaborn as sns# for data viz.

import plotly.express as px



from plotly.subplots import make_subplots# for subplots using plotly



import plotly.graph_objects as go





pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
pd.read_csv('/kaggle/input/kaggle-survey-2018/SurveySchema.csv').head(20)
g1=bt(font_family='Comic Sans MS',color='Dark Black',font_size=19)

g1.printbeautiful('Reading Files')
multiple_choice_responses = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv",low_memory=False)# this warning shows when pandas finds 

# difficult to guess datatype for each column in large dataset

other_text_responses = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv",low_memory=False)

questions_only = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv",low_memory=False)

survey_schema = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv",low_memory=False)
multiple_2018=pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv',low_memory=False)
multiple_2018.head(3)
g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)



g1.printbeautiful('MULTIPLE CHOICE QUESTION OVERVIEW')
multiple_choice_responses.head(3)
g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('MUTIPLE CHOICE RESPONSES MISSING VALUES')
total = multiple_choice_responses.isnull().sum().sort_values(ascending=False)

percent_1 = multiple_choice_responses.isnull().sum()/multiple_choice_responses.isnull().count()*100

percent_1 = (round(percent_1, 1)).sort_values(ascending=False)

missing_multiple_choice_responses = pd.concat([total, percent_1], axis=1, keys=["Total", "%"], sort=False)



g1=bt(font_family='Comic Sans MS',color='bLUE',font_size=19)

g1.printbeautiful('PERCENTAGE OF MISSING VALUES FROM DATA OF 2019 ')

missing_multiple_choice_responses.head(10)
age_groups=multiple_choice_responses.groupby('Q1').count().Q2

age_groups.drop(age_groups.tail(1).index,inplace=True)





age_groups_2018=multiple_2018.groupby('Q2').count().Q1

age_groups_2018.drop(age_groups_2018.tail(1).index,inplace=True)



fig=make_subplots(rows=1,cols=2,

                  specs=[[{"type": "bar"},{"type": "bar"}]] # always make list of list for making subplots

                 ,subplot_titles=('2019','2018'))

fig.add_trace(go.Bar(x=age_groups.index,y=age_groups.values,name='2019'),row=1, col=1)



fig.add_trace(go.Bar(x=age_groups_2018.index,y=age_groups_2018.values,name='2018'),row=1,col=2)



fig.update_layout(title_text='AGE GROUPS OF KAGGLERS ')
age_25_29_dict={}

age_25_29=multiple_choice_responses.loc[multiple_choice_responses.Q1=='25-29']

age_25_29_dict['Twitter']=age_25_29.Q12_Part_1.value_counts().sum()

age_25_29_dict['Hacker News']=age_25_29.Q12_Part_2.value_counts().sum()

age_25_29_dict['Reddit']=age_25_29.Q12_Part_3.value_counts().sum()

age_25_29_dict['Kaggle']=age_25_29.Q12_Part_4.value_counts().sum()

age_25_29_dict['Course Forums']=age_25_29.Q12_Part_5.value_counts().sum()

age_25_29_dict['YouTube']=age_25_29.Q12_Part_6.value_counts().sum()

age_25_29_dict['Podcasts']=age_25_29.Q12_Part_7.value_counts().sum()

age_25_29_dict['Blogs']=age_25_29.Q12_Part_8.value_counts().sum()

age_25_29_dict['Journal Publications']=age_25_29.Q12_Part_9.value_counts().sum()

age_25_29_dict['Slack']=age_25_29.Q12_Part_10.value_counts().sum()

age_25_29_dict['Other']=age_25_29.Q12_Part_11.value_counts().sum()



age_25_29_dict=pd.DataFrame(list(age_25_29_dict.items()),columns=['Learning Resources','Numbers'])



percentage_25_29=[]

for i in range(len(age_25_29_dict)):

    percentage_25_29.append(round(age_25_29_dict.at[i,'Numbers']/age_25_29_dict['Numbers'].sum(),2)*100)



age_25_29_dict['Percentage']=percentage_25_29



age_25_29_dict=age_25_29_dict.sort_values(['Percentage'],ascending=False)

fig=px.bar(age_25_29_dict,y='Learning Resources',x='Percentage', height=500,width=1000,orientation='h',color="Percentage",title='AGE GROUP 25-29')

fig.show()

#######################################################

age_22_24_dict={}

age_22_24=multiple_choice_responses.loc[multiple_choice_responses.Q1=='22-24']

age_22_24_dict['Twitter']=age_22_24.Q12_Part_1.value_counts().sum()

age_22_24_dict['Hacker News']=age_22_24.Q12_Part_2.value_counts().sum()

age_22_24_dict['Reddit']=age_22_24.Q12_Part_3.value_counts().sum()

age_22_24_dict['Kaggle']=age_22_24.Q12_Part_4.value_counts().sum()

age_22_24_dict['Course Forums']=age_22_24.Q12_Part_5.value_counts().sum()

age_22_24_dict['YouTube']=age_22_24.Q12_Part_6.value_counts().sum()

age_22_24_dict['Podcasts']=age_22_24.Q12_Part_7.value_counts().sum()

age_22_24_dict['Blogs']=age_22_24.Q12_Part_8.value_counts().sum()

age_22_24_dict['Journal Publications']=age_22_24.Q12_Part_9.value_counts().sum()

age_22_24_dict['Slack']=age_22_24.Q12_Part_10.value_counts().sum()

age_22_24_dict['Other']=age_22_24.Q12_Part_11.value_counts().sum()



age_22_24_dict=pd.DataFrame(list(age_22_24_dict.items()),columns=['Learning Resources','Numbers'])



percentage_22_24=[]

for i in range(len(age_22_24_dict)):

    percentage_22_24.append(round(age_22_24_dict.at[i,'Numbers']/age_22_24_dict['Numbers'].sum(),2)*100)



age_22_24_dict['Percentage']=percentage_22_24



age_22_24_dict=age_22_24_dict.sort_values(['Percentage'],ascending=False)

fig=px.bar(age_22_24_dict,y='Learning Resources',x='Percentage', height=400,width=1000,color='Percentage',title='AGE GROUP 22-24',orientation='h')

fig.show()
gender_dist_2019=multiple_choice_responses.Q2.iloc[1:].value_counts()

gender_dist_2018=multiple_2018.Q1.iloc[1:].value_counts()





fig=make_subplots(rows=1,cols=2,

                  specs=[[{"type": "pie"},{"type": "pie"}]] # always make list of list for making subplots

                 ,subplot_titles=('2019','2018'))

fig.add_trace(go.Pie(labels=gender_dist_2019.index[:2],values=gender_dist_2019.values[:2],hole=0.2,name='2019',pull=[0,0.3]),row=1,col=1)

fig.add_trace(go.Pie(labels=gender_dist_2018.index[:2], values=gender_dist_2018.values[:2],name='2018', hole=.2,pull=[0,0.3]),row=1,col=2)



fig.update_layout(title_text='AGE GROUPS OF KAGGLERS ')
key1 = "University Courses (resulting in a university degree)"

df=multiple_choice_responses.copy()

# df1 = df[df['Q13_Part_10'] == key1]

# df2 = df[df['Q13_Part_10'] != key1]



nations = ["United States of America", "Canada", "United Kingdom of Great Britain and Northern Ireland", "Brazil", "Russia", "Germany", "Spain", "France",  "India", "Japan", "China", "Other"]

nation_map = {"United States of America" : "USA", "United Kingdom of Great Britain and Northern Ireland" : "UK"}

plt.figure(figsize=(12,12))



vals = []

for j in range(len(nations)):

    country = nations[j]

    country_df = df[df['Q3'] == country]

    ddf1 = country_df[country_df['Q13_Part_10'] == key1]

    ddf2 = country_df[country_df['Q13_Part_10'] != key1]

    plt.subplot(4, 4, j+1)

    

    if j < 4:

        colors = ["#ff8ce0",'#89e8a2']

    elif j < 8:

        colors = ["#60cfe6","#827ec4" ]

    else:

        colors = ["#ff8ce0","#89e8a2"]

    

    vals.append(len(ddf1) / (len(ddf1) + len(ddf2)))    

    plt.pie([len(ddf1), len(ddf2)],

            labels=["With Degree", "No Degree"],

            autopct="%1.0f%%", 

            colors=colors,

            wedgeprops={"linewidth":5,"edgecolor":"white"})

    if country in nation_map:

        country = nation_map[country]

    plt.title(r"$\bf{" + country + "}$")
country=multiple_choice_responses.Q3.value_counts()





fig = go.Figure(go.Treemap(

    labels = country.index,

    parents=['World']*len(country),

    values = country

))



fig.update_layout(title = 'Country of Survey Participants')

fig.show()



## credits https://www.kaggle.com/subinium/the-hitchhiker-s-guide-to-the-kaggle thanks for this wonderful plot type
lang_recom_2019=multiple_choice_responses.Q19[1:].value_counts()



lang_recom_2018=multiple_2018.Q18[1:].value_counts()



print(lang_recom_2019)
lang_recom_2019=multiple_choice_responses.Q19[1:].value_counts()



lang_recom_2018=multiple_2018.Q18[1:].value_counts()





fig=go.Figure(data=[

    go.Bar(x=lang_recom_2019.index,y=lang_recom_2019.values,name='2019',marker_color='rgb(55, 83, 109)'),

    go.Bar(x=lang_recom_2018.index,y=lang_recom_2018.values,name='2018',marker_color='rgb(26, 118, 255)')

])

fig.update_layout(barmode='group')

fig.show()
multiple_choice_responses.head(2)
india_df=multiple_choice_responses.loc[multiple_choice_responses['Q3']=='India']



usa_df=multiple_choice_responses.loc[multiple_choice_responses['Q3']=='United States of America']



age_groups_india=india_df.groupby('Q1').count().Q2



age_groups_usa=usa_df.groupby('Q1').count().Q2



fig=make_subplots(rows=1,cols=2,

                  specs=[[{"type": "bar"},{"type": "bar"}]] # always make list of list for making subplots

                 ,subplot_titles=('INDIA',"USA"))





fig.add_trace(go.Bar(x=age_groups_india.index,y=age_groups_india.values,name='INDIA'),row=1, col=1)



fig.add_trace(go.Bar(x=age_groups_usa.index,y=age_groups_usa.values,name='USA'),row=1,col=2)



fig.update_layout(title_text='AGE GROUPS OF KAGGLERS ')
fig=make_subplots(rows=1,cols=2,

                 specs=[[{"type": "pie"},{"type": "pie"}]]

                 ,subplot_titles=('INDIA',"USA"),

            )



gender_india=india_df.Q2.value_counts()



gender_usa=usa_df.Q2.value_counts()



fig.add_trace(go.Pie(labels=gender_india[:2].index,values=gender_india.values[:2],name='INDIA',pull=[0.4,0]),row=1,col=1)

fig.add_trace(go.Pie(labels=gender_usa[:2].index,values=gender_usa.values[:2],name='USA',pull=[0.4,0],),row=1,col=2)

fig.update_layout(height=500, showlegend=True)

india_formal_edu=india_df.Q4.value_counts().sort_values(ascending=False)# to make groups of eduaction with their frequencies



usa_formal_edu=usa_df.Q4.value_counts().sort_values(ascending=False)# to make groups of eduaction with their frequencies



fig=make_subplots(rows=1,cols=2,

                 specs=[[{'type':'bar'},{'type':'bar'}]],

                 subplot_titles=('INDIA',"USA"))

fig.add_trace(go.Bar(x=india_formal_edu.index[:3],y=india_formal_edu.values[:3],name='INDIA'),row=1,col=1)

fig.add_trace(go.Bar(x=usa_formal_edu.index[:3],y=usa_formal_edu.values[:3],name="USA"),row=1,col=2)
usa_comm=usa_df.Q6.value_counts()

india_comm=india_df.Q6.value_counts()

fig=go.Figure(data=[

    go.Bar(x=usa_comm.index,y=usa_comm.values,name='USA',marker_color='rgb(55, 83, 109)'),

    go.Bar(x=india_comm.index,y=india_comm.values,name='INDIA',marker_color='rgb(26, 118, 255)')

])

fig.update_layout(barmode='group')

fig.show()
usa_data_media={}



india_data_media={}



usa_data_media['Twitter']=usa_df.Q12_Part_1.value_counts().sum()

usa_data_media['Hacker News']=usa_df.Q12_Part_2.value_counts().sum()

usa_data_media['Reddit']=usa_df.Q12_Part_3.value_counts().sum()

usa_data_media['Kaggle']=usa_df.Q12_Part_4.value_counts().sum()

usa_data_media['Course Forums']=usa_df.Q12_Part_5.value_counts().sum()

usa_data_media['YouTube']=usa_df.Q12_Part_6.value_counts().sum()

usa_data_media['Podcasts']=usa_df.Q12_Part_7.value_counts().sum()

usa_data_media['Blogs']=usa_df.Q12_Part_8.value_counts().sum()

usa_data_media['Journal Publications']=usa_df.Q12_Part_9.value_counts().sum()

usa_data_media['Slack']=usa_df.Q12_Part_10.value_counts().sum()

usa_data_media['Other']=usa_df.Q12_Part_11.value_counts().sum()



usa_media_df=pd.DataFrame(list(usa_data_media.items()),columns=['Sources','Numbers'])



india_data_media['Twitter']=india_df.Q12_Part_1.value_counts().sum()

india_data_media['Hacker News']=india_df.Q12_Part_2.value_counts().sum()

india_data_media['Reddit']=india_df.Q12_Part_3.value_counts().sum()

india_data_media['Kaggle']=india_df.Q12_Part_4.value_counts().sum()

india_data_media['Course Forums']=india_df.Q12_Part_5.value_counts().sum()

india_data_media['YouTube']=india_df.Q12_Part_6.value_counts().sum()

india_data_media['Podcasts']=india_df.Q12_Part_7.value_counts().sum()

india_data_media['Blogs']=india_df.Q12_Part_8.value_counts().sum()

india_data_media['Journal Publications']=india_df.Q12_Part_9.value_counts().sum()

india_data_media['Slack']=india_df.Q12_Part_10.value_counts().sum()

india_data_media['Other']=india_df.Q12_Part_11.value_counts().sum()



india_media_df=pd.DataFrame(list(india_data_media.items()),columns=['Sources','Numbers'])# dataframe for Indian Data science community about how

#they update themselves about Data Science



fig=go.Figure(data=[

    go.Bar(x=india_media_df['Sources'],y=india_media_df['Numbers'],name='INDIA'),

    go.Bar(x=usa_media_df['Sources'],y=usa_media_df['Numbers'],name="USA")

])



# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
usa_comm_tools=usa_df.Q14.value_counts()

india_comm_tools=india_df.Q14.value_counts()

fig=go.Figure(data=[

    go.Bar(x=usa_comm_tools.index[:2],y=usa_comm_tools.values[:2],name='USA'),

    go.Bar(x=india_comm_tools.index[:2],y=india_comm_tools.values[:2],name='INDIA')

])

fig.update_layout(barmode='group')

fig.show()
usa_lang={}

india_lang={}



usa_lang['Python']=usa_df.Q18_Part_1.value_counts().sum()

usa_lang['R']=usa_df.Q18_Part_2.value_counts().sum()

usa_lang['SQL']=usa_df.Q18_Part_3.value_counts().sum()

usa_lang['C']=usa_df.Q18_Part_4.value_counts().sum()

usa_lang['C++']=usa_df.Q18_Part_5.value_counts().sum()

usa_lang['Java']=usa_df.Q18_Part_6.value_counts().sum()

usa_lang['Javascript']=usa_df.Q18_Part_7.value_counts().sum()

usa_lang['TypeScript']=usa_df.Q18_Part_8.value_counts().sum()

usa_lang['Bash']=usa_df.Q18_Part_9.value_counts().sum()

usa_lang['MATLAB']=usa_df.Q18_Part_10.value_counts().sum()

usa_lang['None']=usa_df.Q18_Part_11.value_counts().sum()

usa_lang['Other']=usa_df.Q18_Part_12.value_counts().sum()





usa_lang_df=pd.DataFrame(list(usa_lang.items()),columns=['Languages','Numbers'])





india_lang['Python']=india_df.Q18_Part_1.value_counts().sum()

india_lang['R']=india_df.Q18_Part_2.value_counts().sum()

india_lang['SQL']=india_df.Q18_Part_3.value_counts().sum()

india_lang['C']=india_df.Q18_Part_4.value_counts().sum()

india_lang['C++']=india_df.Q18_Part_5.value_counts().sum()

india_lang['Java']=india_df.Q18_Part_6.value_counts().sum()

india_lang['Javascript']=india_df.Q18_Part_7.value_counts().sum()

india_lang['TypeScript']=india_df.Q18_Part_8.value_counts().sum()

india_lang['Bash']=india_df.Q18_Part_9.value_counts().sum()

india_lang['MATLAB']=india_df.Q18_Part_10.value_counts().sum()

india_lang['None']=india_df.Q18_Part_11.value_counts().sum()

india_lang['Other']=india_df.Q18_Part_12.value_counts().sum()



india_lang_df=pd.DataFrame(list(india_lang.items()),columns=['Languages','Numbers'])



fig=make_subplots(rows=1,cols=2,

                 specs=[[{"type": "pie"},{"type": "pie"}]]

                 ,subplot_titles=('INDIA',"USA"),

            )



fig.add_trace(go.Pie(labels=india_lang_df['Languages'],values=india_lang_df['Numbers'],name='INDIA'),row=1,col=1)



fig.add_trace(go.Pie(labels=usa_lang_df['Languages'],values=usa_lang_df['Numbers'],name='USA'),row=1,col=2)



fig.update_layout(height=800, showlegend=True)



fig.update_traces(hole=.4)# to create donut like pie chart



fig.show()
india_lang_combined_dict={}



usa_lang_combined_dict={}



india_lang_combined=india_df.loc[:,['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4'

               ,'Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8',

               'Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']]



india_lang_combined_dict['Python and R']=india_lang_combined.loc[(india_lang_combined['Q18_Part_1']=='Python') & 

                         (india_lang_combined['Q18_Part_2']=='R') ].shape[0]



india_lang_combined_dict['Python and SQL']=india_lang_combined.loc[(india_lang_combined['Q18_Part_1']=='Python') & 

                         (india_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



india_lang_combined_dict['R and SQL']=india_lang_combined.loc[(india_lang_combined['Q18_Part_2']=='R') & 

                         (india_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



india_lang_combined_dict['Python ,R and SQL']=india_lang_combined.loc[(india_lang_combined['Q18_Part_1']=='Python') & (india_lang_combined['Q18_Part_2']=='R') &

                         (india_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



india_lang_combined_dict=pd.DataFrame(list(india_lang_combined_dict.items()),columns=['Languages',"Numbers"])

#######################################################################################################3

usa_lang_combined=usa_df.loc[:,['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4'

               ,'Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8',

               'Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']]



usa_lang_combined_dict['Python and R']=usa_lang_combined.loc[(usa_lang_combined['Q18_Part_1']=='Python') & 

                         (usa_lang_combined['Q18_Part_2']=='R') ].shape[0]



usa_lang_combined_dict['Python and SQL']=usa_lang_combined.loc[(usa_lang_combined['Q18_Part_1']=='Python') & 

                         (usa_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



usa_lang_combined_dict['R and SQL']=usa_lang_combined.loc[(usa_lang_combined['Q18_Part_2']=='R') & 

                         (usa_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



usa_lang_combined_dict['Python ,R and SQL']=usa_lang_combined.loc[(usa_lang_combined['Q18_Part_1']=='Python') & (usa_lang_combined['Q18_Part_2']=='R') &

                         (usa_lang_combined['Q18_Part_3']=='SQL') ].shape[0]



usa_lang_combined_dict=pd.DataFrame(list(usa_lang_combined_dict.items()),columns=['Languages',"Numbers"])
fig=make_subplots(1,2,specs=[[{'type':'bar'},{'type':'bar'}]], subplot_titles=('INDIA',"USA"),)



fig.add_trace(go.Bar(x=india_lang_combined_dict['Languages'],y=india_lang_combined_dict['Numbers'],name='INDIA'),row=1,col=1)



fig.add_trace(go.Bar(x=usa_lang_combined_dict['Languages'],y=usa_lang_combined_dict['Numbers'],name='USA'),row=1,col=2)
usa_hardware={}

india_hardware={}





usa_hardware['CPU']=usa_df.Q21_Part_1.value_counts().sum()

usa_hardware['GPU']=usa_df.Q21_Part_2.value_counts().sum()

usa_hardware['TPU']=usa_df.Q21_Part_3.value_counts().sum()

usa_hardware['other']=usa_df.Q21_Part_5.value_counts().sum()

usa_hardware=pd.DataFrame(list(usa_hardware.items()),columns=['Hardware',"Numbers"])



india_hardware['CPU']=india_df.Q21_Part_1.value_counts().sum()

india_hardware['GPU']=india_df.Q21_Part_2.value_counts().sum()

india_hardware['TPU']=india_df.Q21_Part_3.value_counts().sum()

india_hardware['other']=india_df.Q21_Part_5.value_counts().sum()

india_hardware=pd.DataFrame(list(india_hardware.items()),columns=['Hardware',"Numbers"])



fig=make_subplots(rows=1,cols=2,

                 specs=[[{"type": "bar"},{"type": "bar"}]]

                 ,subplot_titles=('INDIA',"USA"),

            )

fig.add_trace(go.Bar(y=india_hardware['Hardware'],x=india_hardware['Numbers'],name='INDIA', orientation='h'),row=1,col=1)

fig.add_trace(go.Bar(y=usa_hardware['Hardware'],x=usa_hardware['Numbers'],name="USA", orientation='h'),row=1,col=2)
country_list={}

countries=multiple_choice_responses.loc[1:].Q3.unique()

countries=list(countries)

for i in countries[:]:

    f=multiple_choice_responses.loc[multiple_choice_responses['Q3']==i]

    country_list[i]=f.Q2.value_counts()[:2]

#     print(f.Q2.unique())

country_list=pd.DataFrame(list(country_list.items()),columns=['Name','Male'])

# country_list