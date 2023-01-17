# Suppress warnings 

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)



from IPython.display import HTML
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objects as go

from plotly import tools

init_notebook_mode(connected=True)





%matplotlib inline

pd.set_option("display.max_rows",500)

pd.set_option("display.max_columns",200)
def compute_percentage(df,col):

    return df[col].value_counts(normalize=True) * 100



def bi_variant_chart(col1,col2,x_title,y_title, mcr_brazil):

    index = mcr_brazil[col1].dropna().unique()

    vals = mcr_brazil[col2].unique()

    layout = go.Layout()

    trace = []

    for j,y_axis in enumerate(vals):

        trace.append(go.Bar(x = mcr_brazil[mcr_brazil[col2] == y_axis][col1].value_counts().index,

                            y = mcr_brazil[mcr_brazil[col2] == y_axis][col1].sort_values().value_counts().values,

                opacity = 0.6, name = vals[j]))

    fig = go.Figure(data = trace, layout = layout)

    fig.update_layout(

        title = x_title,

        yaxis = dict(title = y_title),

        legend = dict( bgcolor = 'rgba(255, 255, 255, 0)', bordercolor = 'rgba(255, 255, 255, 0)'),

        bargap = 0.15, bargroupgap = 0.1,legend_orientation="h")

    fig.show()

    

def bar_graph(col,type_of_graph, mcr_brazil):

    data_frame = compute_percentage(mcr_brazil,col)

    layout = go.Layout()

    

    if type_of_graph == 'bar':

        data = [go.Bar(

                x = data_frame.values,

                y = data_frame.index,

                opacity = 0.6,

                orientation='h',

               marker=dict(color=data_frame.values,colorscale='portland') 



            )]

    elif type_of_graph == 'pie':

        data = [go.Pie(

            labels = data_frame.index,

            values = data_frame.values,

            textfont = dict(size = 20)

        )]

    fig = go.Figure(data = data, layout = layout)

    py.iplot(fig)
mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')

qs = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
mcr_brazil = mcr[mcr['Q3']=='Brazil']
print("There are ",mcr_brazil.shape[0], "respondents ",mcr_brazil.shape[0]/mcr.shape[0],"%")
mcr_brazil = mcr_brazil.reset_index().drop('index', axis=1)

mcr_brazil.shape
mcr_brazil_male = mcr_brazil[mcr_brazil.Q2=='Male'].reset_index().drop('index', axis=1)

mcr_brazil_male.shape
mcr_brazil_female = mcr_brazil[mcr_brazil.Q2=='Female'].reset_index().drop('index', axis=1)

mcr_brazil_female.shape
mcr_brazil.head()
mcr_brazil.Q1.value_counts()
mcr_brazil.Q2.value_counts()
mcr_brazil.Q4.value_counts()
mcr_brazil.Q5.value_counts()
mcr_brazil.Q6.value_counts()
mcr_brazil.Q7.value_counts()
mcr_brazil.Q8.value_counts()
mcr_brazil.Q11.value_counts()
mcr_brazil.Q14.value_counts()
mcr_brazil.Q15.value_counts()
mcr_brazil.Q19.value_counts()
mcr_brazil.Q22.value_counts()
mcr_brazil.Q23.value_counts()
bar_graph("Q2","bar",mcr_brazil)
bi_variant_chart("Q6","Q1","Company size VS age group","Count",mcr_brazil)
bi_variant_chart("Q6","Q5","Company size VS Designation","Count",mcr_brazil)
bi_variant_chart("Q6","Q10","Company size VS Salary","",mcr_brazil)
bi_variant_chart("Q6","Q1","Company size VS age group","Count",mcr_brazil_female)
bi_variant_chart("Q6","Q4","Company size VS highest level of formal education","Count",mcr_brazil_female)
bi_variant_chart("Q6","Q5","Company size VS Designation","Count", mcr_brazil_female)
bi_variant_chart("Q6","Q10","Company size VS Salary","", mcr_brazil_female)
HTML('<iframe width="1280" height="720" src="https://www.youtube.com/embed/zBqPg80l7xA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')