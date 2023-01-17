import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
QUERY = """
        SELECT language
        FROM `bigquery-public-data.github_repos.languages`
        """
print ("QUERY SIZE:   ")
print (str(round((bq_assistant.estimate_query_size(QUERY)),2))+str(" GB"))
QUERY = """
        SELECT count(language) as COUNT
        FROM `bigquery-public-data.github_repos.languages`
        """
df = bq_assistant.query_to_pandas_safe(QUERY)
print (df)
QUERY = """
        SELECT language
        FROM `bigquery-public-data.github_repos.languages`
        limit 50000
        """
df = bq_assistant.query_to_pandas_safe(QUERY)
## lets find out which file is the most language file is the most found in github repositories?
Names=[]
for x in df.language:
    Names.extend(x)
Count_Names={}
Average_Size={}
for x in Names:
    if x["name"] not in Count_Names:
        Count_Names[x["name"]]=0
        Average_Size[x["name"]]=0
    Count_Names[x["name"]]+=1
    Average_Size[x["name"]]+=x["bytes"]

for x in Count_Names.keys():
    Average_Size[x]=Average_Size[x]/Count_Names[x]
def Create_WordCloud(Frequency):
    wordcloud = WordCloud(background_color='black',
                              random_state=42).generate_from_frequencies(Frequency)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):
    list_of_tuples=list_of_tuples[:items_to_show]
    data = [go.Bar(
            x=[val[0] for val in list_of_tuples],
            y=[val[1] for val in list_of_tuples]
    )]
    layout = go.Layout(
    title=title,xaxis=dict(
        autotick=False,
        tickangle=290 ),)
    fig = go.Figure(data=data, layout=layout)
    #py.offline.iplot(data,layout=layout)
    
    py.offline.iplot(fig)
Create_WordCloud(Count_Names)
sorted_names = sorted(Count_Names.items(), key=operator.itemgetter(1), reverse=True)
Create_Bar_plotly(sorted_names,30,"Most found programming languages source files in github")
Common_languages=["C","Python", "Java", "C++", "JavaScript","C#"]
Common=[]
for x in Common_languages:
    Common.append((x, Count_Names[x]))
Create_Bar_plotly(Common,40,"Comparision of Commonly known languages")
Create_WordCloud(Average_Size)
Common_languages=["C","Python", "Java", "C++", "JavaScript","C#"]
Common=[]
for x in Common_languages:
    Common.append((x, Average_Size[x]))

sorted_average = sorted(Average_Size.items(), key=operator.itemgetter(1), reverse=True)
Create_Bar_plotly(sorted_average,35,"Programming languages Average source file size")
sorted_average = sorted(Average_Size.items(), key=operator.itemgetter(1))
Create_Bar_plotly(sorted_average,40)
Create_Bar_plotly(Common,40)
#I can only fetch 50000 entries of the total of 3359866 entries, more than that it
# gives error Please suggest on how to fix it.