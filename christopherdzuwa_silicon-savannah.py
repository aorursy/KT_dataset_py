import numpy as np

import pandas as pd

import seaborn as sns

sns.set()

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



#load datasets from 2017 through 2019



mc2019 = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")

other_text_responses2019 = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv")

questions_only2019 = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv")

survey_schema2019 = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")



mc2018 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")



mc2017 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='latin-1', low_memory=False)



#ai hostspots in africa

ai_hotspots = ['Kenya','Egypt','South Africa','Nigeria']



#filter aihotspots from 2017 kaggle ds and ml dataset

df_afr2017 = mc2017[mc2017['Country'].isin(ai_hotspots)]



#filter aihotspots from 2018 kaggle ds and ml dataset

df_afr2018 = mc2018[mc2018['Q3'].isin(ai_hotspots)]



#filter aihotspots from 2019 kaggle ds and ml dataset

df_afr2019 = mc2019[mc2019['Q3'].isin(ai_hotspots)]



#prepare stage settings

pd.set_option('mode.chained_assignment', None)

colors = ["Tomato",

         "SlateBlue",

         "Fuchsia",

         "MediumSeaGreen",

          "Teal",

          "Violet",

         ]



#helper functions   

def round2(value):

    return round(value, 2)



def calculate_percentage(data, question):

    df = (data[question]

                 .value_counts(normalize=True)

                 .rename('percentage')

                 .mul(100)

                 .sort_index())

    return df



def plot_graph(data,

               question_code,

               title,

               x_axis_title,

               y_axis_title,

               order,

               use_multi_colors=True):

    

    df = calculate_percentage(data,question_code)

    c = colors if use_multi_colors else 'rgb(55, 83, 109)'

    trace1 = go.Bar(

                    x = df.index,

                    y = round2(df[df.index]),

                    #orientation='h',

                    marker = dict(color=c,

                                 line=dict(color='black',width=1)),

                    text = df.index)

    data = [trace1]

    layout = go.Layout(barmode = "group",title=title,width=800, height=500,

                       xaxis=dict(type='category',categoryorder='array',categoryarray=order,title=y_axis_title),

                       yaxis= dict(title=x_axis_title))

                       

    fig = go.Figure(data = data, layout = layout)

    iplot(fig)

    

    

def plot_graph2(data, 

                question_code, 

                xaxis_title=None, 

                yaxis_title=None, 

                data_label=None,

                perc = None,

                rotate_xlabels=False):

    

    df = calculate_percentage(data, question_code)

    data = {data_label:df.index, 'percentage': df if perc is None else perc}

    

    # plot data

    plt.figure(figsize=(12,7))

    ax = sns.barplot(x=data_label, y='percentage', data=data)

    

    #set chart labels

    _ = ax.set(xlabel= xaxis_title, ylabel= yaxis_title)

    

    #default rotaton to 45 degrees

    if rotate_xlabels:

        _ = plt.setp(ax.get_xticklabels(), rotation=45)



survey_years=["2017", "2018", "2019"]

nigeria_stat = [df_afr2017['Country']

                               .value_counts()['Nigeria'], df_afr2018['Q3']

                               .value_counts()['Nigeria'],  df_afr2019['Q3']

                               .value_counts()['Nigeria']]



south_africa_stat = [df_afr2017['Country']

                               .value_counts()['South Africa'], df_afr2018['Q3']

                               .value_counts()['South Africa'],  df_afr2019['Q3']

                               .value_counts()['South Africa']]



egypt_stat = [df_afr2017['Country']

                               .value_counts()['Egypt'], df_afr2018['Q3']

                               .value_counts()['Egypt'],  df_afr2019['Q3']

                               .value_counts()['Egypt']]



kenya_stat = [df_afr2017['Country']

                               .value_counts()['Kenya'], df_afr2018['Q3']

                               .value_counts()['Kenya'],  df_afr2019['Q3']

                               .value_counts()['Kenya']]



#chart style

layout = go.Layout(

    title=go.layout.Title(

        text="# of Respondents from Africa's 'BIG 4'" ,

        xref='paper',

        x=0

    ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text='Year',

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='Count',

        )

    )

)



fig = go.Figure(data=[

    go.Bar(name='Nigeria', x=survey_years, y=nigeria_stat),

    go.Bar(name='Egypt', x=survey_years, y=egypt_stat),

    go.Bar(name='South Africa', x=survey_years, y= south_africa_stat), 

    go.Bar(name='Kenya', x=survey_years, y= kenya_stat), 

], layout=layout)



# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
# computed by wolfram alpha

population = {'Nigeria' : 191e6, 

              'Kenya' : 49.7e6, 

              'Egypt' : 97.6e6, 

              'South Africa' : 56.7e6} 



#compute relative ratios

nigeria_relative = np.array(nigeria_stat) / population['Nigeria']

egypt_relative = np.array(egypt_stat) / population['Egypt']

south_africa_relative = np.array(south_africa_stat) / population['South Africa']

kenya_relative = np.array(kenya_stat) / population['Kenya']



#chart style

layout = go.Layout(

    title=go.layout.Title(

        text="Relative ratios for Africa's 'BIG 4'" ,

        xref='paper',

        x=0

    ),

    xaxis=go.layout.XAxis(

        title=go.layout.xaxis.Title(

            text='Year',

        )

    ),

    yaxis=go.layout.YAxis(

        title=go.layout.yaxis.Title(

            text='Relative ratio',

        )

    )

)





fig = go.Figure(data=[

    go.Bar(name='Nigeria', x=survey_years, y=nigeria_relative),

    go.Bar(name='Egypt', x=survey_years, y= egypt_relative),

    go.Bar(name='South Africa', x=survey_years, y=south_africa_relative),

    go.Bar(name='Kenya', x=survey_years, y= kenya_relative ),   

], layout=layout)



# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
world_startups = mc2019[mc2019['Q6'].isin(['0-49 employees'])]

afr_startups = df_afr2019[df_afr2019['Q6'].isin(['0-49 employees'])]



# kenyan startups

kenya = df_afr2019[df_afr2019['Q3'].isin(['Kenya'])]

kenya_startups = kenya[kenya['Q6'].isin(['0-49 employees'])]



# Nigerian Startups

nigeria = df_afr2019[df_afr2019['Q3'].isin(['Nigeria'])]

nigeria_startups = nigeria[nigeria['Q6'].isin(['0-49 employees'])]



# South African startups

rsa = df_afr2019[df_afr2019['Q3'].isin(['South Africa'])]

rsa_startups = rsa[rsa['Q6'].isin(['0-49 employees'])]



#Egyptian startups

egypt = df_afr2019[df_afr2019['Q3'].isin(['Egypt'])]

egypt_startups = egypt[egypt['Q6'].isin(['0-49 employees'])]



#substitute longer words in ml uses

kenya_startups['Q4'] =  kenya_startups['Q4'].replace({"Some college/university study without earning a bachelor’s degree":"Some College"})

uses =   {'No (we do not use ML methods)':"Don't use",

          "We are exploring ML methods (and may one day put a model into production)":"Exploration",

          "We have well established ML methods (i.e., models in production for more than 2 years)":"Models in prod > 2years",

          "We recently started using ML methods (i.e., models in production for less than 2 years)" : "Models in prod < 2 years",

          "We use ML methods for generating insights (but do not put working models into production)" : "Generate Insights"}



kenya_startups['Q8'] =  kenya_startups['Q8'].replace(uses)
round2((afr_startups.shape[0] / world_startups.shape[0]) * 100)
kp =kenya_startups.shape[0] / afr_startups.shape[0]

ep = egypt_startups.shape[0] /  afr_startups.shape[0]

sap = rsa_startups.shape[0] /  afr_startups.shape[0]

ngp = nigeria_startups.shape[0] / afr_startups.shape[0]



#compute total ratio

total = ep + ngp + sap + kp

total
# process data

perc = np.array([kp, ep, sap,  ngp]) * 100



data = {'countries':ai_hotspots, 'percentage':perc}

plt.figure(figsize=(12,7))

ax = sns.barplot(x='countries', y='percentage', data=data)



#set chart labels

_ = ax.set(xlabel='Country', ylabel='% of Respondents')
#ratios in relation to country's population

kr = (kp * 100) / kenya.shape[0]

ngr = (ngp * 100) / nigeria.shape[0]

rsr = (sap * 100) / rsa.shape[0]

er = (ep * 100) / egypt.shape[0]



# organize data

ratio = [kr, er,rsr, ngr]

data = {'countries':ai_hotspots, 'ratio':ratio}



# plot data

plt.figure(figsize=(12,7))

ax = sns.barplot(x='countries', y='ratio', data=data)



#set chart labels

_ = ax.set(xlabel='Country', ylabel='Relative ratio')
plot_graph2(kenya, 'Q1',  xaxis_title='Age(Years)',  yaxis_title='% of Respondents',  data_label='age')
df = kenya_startups["Q2"].value_counts()

labels = ["Male","Female", ]

values = [df["Male"], 

          df["Female"],

         ]



#chart style

layout = go.Layout(

    title=go.layout.Title(

        text="Women vs Men in Kenyan startups" ,

        xref='paper',

        x=0

    )

)



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0)], layout=layout)

fig.show()

plot_graph2(kenya_startups, 'Q4', xaxis_title='Level of education',  yaxis_title='% of Respondents',  data_label='education')
from wordcloud import WordCloud

job_titles = kenya_startups['Q5'].dropna()



  

# plot the WordCloud image                        

plt.figure(figsize = (20, 7), facecolor = None) 



wordcloud = WordCloud( background_color='white',

                        width=600, height=400).generate(" ".join(job_titles))

plt.imshow(wordcloud) 

plt.axis("off") # .set_title("Job Titles in Kenya's Startups, 2019",fontsize=20)

plt.tight_layout(pad = 0) 

years_order = ['I have never written code',

                '< 1 years',

                '1-2 years',

                '3-5 years',

                '5-10 years',]



plot_graph(kenya_startups,'Q15',

               'Years writting code','% of Respondents','Experience in Years',  years_order)

    
plot_graph2(kenya_startups, 'Q8',  xaxis_title='ML Usage',  yaxis_title='% of Respondents',  data_label='ml_use', rotate_xlabels=True)
money_spent_order = ['$0 (USD)',

                '$1-$99',

                '$100-$999',

                '$1000-$9,999',

                '$10,000-$99,999',]



plot_graph(kenya_startups,'Q11',

               'Money spent on ML','% of Respondents','Money spent', money_spent_order)

    
colleagues_order = ['0',

                '1-2',

                '3-4',

                '5-9',

                '10-14',]



plot_graph(kenya_startups,'Q7',

               'Data Science Colleagues','% of Respondents','Colleagues', colleagues_order)
comp_order = ['$0-999',

                '1,000-1,999',

                '2,000-2,999',

                '3,000-3,999',

                '4,000-4,999',

                '5,000-7,499',

                '7,500-9,999',

                 '10,000-14,999',

                '15,000-19,999',

                '25,000-29,999',

                '50,000-59,999',

                '60,000-69,999',]



plot_graph(kenya_startups,'Q10',

               'Yearly compensation','% of Respondents','Compensation in USD', comp_order, use_multi_colors=False)

    
comp_vs_education = (kenya_startups.groupby(['Q4'])['Q10']

                     .value_counts(normalize=True)

                     .rename('percentage')

                     .mul(100)

                     .reset_index())



df = calculate_percentage(kenya_startups,"Q10")



com1 = comp_vs_education[comp_vs_education['Q4'].isin(["Bachelor’s degree"])]

trace1 = go.Bar(

                x = com1['Q10'],

                y = round2(com1['percentage']),

                name = "Bachelor's degree",

                marker = dict(

                             line=dict(color='black',width=1)),

                text = com1['Q4'])



com2 = comp_vs_education[comp_vs_education['Q4'].isin(["Master’s degree"])]

trace2 = go.Bar(

                x = com2['Q10'],

                y = round2(com2['percentage']),

                name = "Master's degree",

                marker = dict(

                             line=dict(color='black',width=1)),

                text = com2['Q4'])



com3 = comp_vs_education[comp_vs_education['Q4'].isin(["Some College"])]

trace3 = go.Bar(

                x = com3['Q10'],

                y = com3['percentage'],

                name = "Some College",

                marker = dict(

                             line=dict(color='black',width=1)),

                text = com3['Q4'])



data = [trace1, trace2, trace3]



#set chart layout

layout = go.Layout(barmode = "stack",title="Compensation vs level of education",width=800, height=500,

                   xaxis=dict(type='category',categoryorder='array',categoryarray=comp_order,title="Yearly compensation"),

                   yaxis= dict(title="% of Respondents"))



#prepare and plot 

fig = go.Figure(data = data, layout = layout)

iplot(fig)

    