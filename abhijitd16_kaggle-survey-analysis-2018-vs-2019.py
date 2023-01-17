# import the necessary libraries

import numpy as np 

import pandas as pd 

import os



# visualization

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import missingno as msno

import networkx as nx

import plotly.graph_objects as go

from IPython.core.display import display, HTML

from matplotlib.gridspec import GridSpec



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')
# Importing the 2017,2018 and 2019 survey dataset



#Importing the 2019 Dataset

mcq19 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

schema19 = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')





#Importing the 2018 Dataset

mcq18 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')

schema18 = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv')

mcq18.rename(columns=mcq18.iloc[0], inplace=True)

mcq18.drop(0, axis=0, inplace=True)

mcq19.rename(columns=mcq19.iloc[0], inplace=True)

mcq19.drop(0, axis=0, inplace=True)

schema18.rename(columns=schema18.iloc[0], inplace=True)

schema18.drop(0, axis=0, inplace=True)

schema19.rename(columns=schema19.iloc[0], inplace=True)

schema19.drop(0, axis=0, inplace=True)
def extract_19_data(qstring):

    lst = []

    for i in mcq19.columns:

        if i[:len(qstring)]==qstring:

            lst.append(i)

    for i in schema19.columns:

        if i[:len(qstring)]==qstring:

            num = schema19[i].iloc[0]

    df = mcq19[lst]

    df_choice = pd.Series(df.columns).apply(lambda x: ''.join(x.split('-')[2:]))

    df_colnames = {b:a for a, b in zip(df_choice.values, df.columns)}

    df.rename(columns=df_colnames, inplace=True)  

    return df, df_choice, num
def extract_18_data(qstring):

    lst = []

    for i in mcq18.columns:

        if i[:len(qstring)]==qstring:

            lst.append(i)

    for i in schema18.columns:

        if i[:len(qstring)]==qstring:

            numResponses = schema18[i].iloc[0]

       

    df = mcq18[lst]

    df_choice = pd.Series(df.columns).apply(lambda x: ''.join(x.split('-')[2:]))

    df_colnames = {b:a for a, b in zip(df_choice.values, df.columns)}

    df.rename(columns=df_colnames, inplace=True)

    return df, df_choice, numResponses
def chartPlotter(chartData, totalResponses, chartTitle):

    counts = {}

    for i in chartData.columns[:-1]:

        counts.update(chartData[i].value_counts())

    counts = pd.Series(counts).sort_values(ascending=False)

    counts = counts.apply(lambda x: x/int(numResponses))

    fig = go.Figure([go.Bar(x=counts.index, y=counts)])

    fig.update_layout(title=chartTitle)

    fig.show()
def chartPlotterDouble(chartData1, totalResponses1, chartTitle1, chartData2, totalResponses2, chartTitle2):

    counts1 = {}

    counts2 = {}

    for i in chartData1.columns[:-1]:

        counts1.update(chartData1[i].value_counts())

    counts1 = pd.Series(counts1).sort_values(ascending=False)

    counts1 = counts1.apply(lambda x: x*100/int(totalResponses1)) # Calculate %

    

    for i in chartData2.columns[:-1]:

        counts2.update(chartData2[i].value_counts())

    counts2 = pd.Series(counts2).sort_values(ascending=False)

    counts2 = counts2.apply(lambda x: x*100/int(totalResponses2)) # Calculate %

    plt.figure(2, figsize=(20,6) )

    grid = GridSpec(1,2)

    sns.set( style="whitegrid" )

    plt.figure(2, figsize=(20,6) )

    grid = GridSpec(1,2)

    sns.set( style="whitegrid" )

    

    plt.subplot( grid[0,0], title= chartTitle1 )

    chart2018 = sns.barplot( x=counts1.index, y=counts1, palette="GnBu_d" )



    plt.subplot( grid[0,1], title= chartTitle2 )

    chart2019 = sns.barplot( x=counts2.index, y=counts2, palette="GnBu_d" )

    chart2018.set_xticklabels( counts1.index, rotation="90" )

    chart2019.set_xticklabels( counts2.index, rotation="90" )

    chart2018.set_ylabel("% Responses")

    chart2019.set_ylabel("% Responses")

    plt.show()

    
chart_data19, data_choices19, numResponses19 = extract_19_data('What programming languages do you use on a regular basis?')

chart_data18, data_choices18, numResponses18 = extract_18_data("What programming languages do you use on a regular basis?")

chartPlotterDouble(chart_data18, numResponses18, "Programming Languages 2018",  chart_data19, numResponses19, "Programming Languages 2019")
chart_data19, data_choices19, numResponses19 = extract_19_data("What programming language would you recommend an aspiring data scientist to learn first?")

chart_data18, data_choices18, numResponses18 = extract_18_data("What programming language would you recommend an aspiring data scientist to learn first?")

chartPlotterDouble(chart_data18, numResponses18, "Recommended Programming Languages 2018",  chart_data19, numResponses19, "Recommended Programming Languages 2019")
chart_data19, data_choices19, numResponses19 = extract_19_data("Which of the following machine learning frameworks do you use on a regular basis?")

chart_data18, data_choices18, numResponses18 = extract_18_data("Of the choices that you selected in the previous question, which ML library have you used the most?")

chartPlotterDouble(chart_data18, numResponses18, "ML framework 2018",  chart_data19, numResponses19, "ML framework 2019")
chart_data19, data_choices19, numResponses19 = extract_19_data("Which specific cloud computing products do you use on a regular basis?")

chart_data18, data_choices18, numResponses18 = extract_18_data("Which of the following cloud computing products have you used at work or school in the last 5 years")

chartPlotterDouble(chart_data18, numResponses18, "Cloud Computing 2018",  chart_data19, numResponses19, "Cloud Computing 2019")
chart_data19, data_choices19, numResponses19 = extract_19_data("Which specific big data / analytics products do you use on a regular basis?")

chart_data18, data_choices18, numResponses18 = extract_18_data("Which of the following big data and analytics products have you used at work or school in the last 5 years?")

chartPlotterDouble(chart_data18, numResponses18, "Big Data Analytics 2018",  chart_data19, numResponses19, "Big Data Analytics 2019")
chart_data19, data_choices19, numResponses19 = extract_19_data("What data visualization libraries or tools do you use on a regular basis?")

chart_data18, data_choices18, numResponses18 = extract_18_data("What data visualization libraries or tools have you used in the past 5 years?")

chartPlotterDouble(chart_data18, numResponses18, "Data Visualization 2018",  chart_data19, numResponses19, "Data Visualization 2019")
chartdata19, data_choices19, numResponses19 = extract_19_data("Which specific big data / analytics products do you use on a regular basis?")

companysize = pd.DataFrame(mcq19["What is the size of the company where you are employed?"])

chartdata19.drop([' Text'], axis=1, inplace=True)

chartdata19=chartdata19.fillna(0).apply(lambda x : x!=0)

chartdata19['companysize'] = companysize



table = pd.pivot_table(chartdata19, values= [' Google BigQuery', ' AWS Redshift', ' Teradata', ' Databricks', 

                                             ' AWS Elastic MapReduce', ' Microsoft Analysis Services', 

                                             ' Google Cloud Dataflow', ' AWS Athena', ' AWS Kinesis',

                                             ' Google Cloud Pub/Sub'], index=['companysize'], aggfunc=np.sum)

table