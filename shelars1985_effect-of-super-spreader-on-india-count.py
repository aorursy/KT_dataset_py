import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.options.mode.chained_assignment = None

import datetime





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import string

import nltk

from nltk.corpus import stopwords



covid_19_df = pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")



def lower_text(text):

    return text.lower()



covid_19_df['notes'] = covid_19_df['notes'].astype(str)

covid_19_df['notes'] = covid_19_df.apply(lambda x: lower_text(x['notes']),axis=1)





index_india = covid_19_df['notes'].value_counts().index

values_india = covid_19_df['notes'].value_counts().values



data = {'index_india':index_india ,'values_india':values_india } 

df = pd.DataFrame(data) 

df = df.dropna()

df [df['values_india'] > 10]
def text_process(mess):

    # Check characters to see if they are in punctuation

    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.

    nopunc = ''.join(nopunc)

    # Now just remove any stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
from nltk import FreqDist

df['index_india'] = df['index_india'].apply(text_process)

covid_19_df['notes'] = covid_19_df['notes'].apply(text_process)
list_all_word = []

delhi_occur = 0

travel_occur = 0

contact_occur = 0

awaited_occur = 0

for list_word, occur in zip(df['index_india'] , df['values_india']) :

    for word_list in list_word :

        if word_list == 'delhi':

           #print(word_list)

           #print(occur)  

           delhi_occur = occur + delhi_occur

        elif word_list == 'travelled':

            travel_occur = occur + travel_occur

        elif word_list == 'contact':

            contact_occur = occur + contact_occur

        elif word_list == 'awaited':

            awaited_occur = occur + awaited_occur        

        list_all_word.append(word_list)

#print(delhi_occur)

#print(travel_occur)

#print(contact_occur)

#print(awaited_occur)
list_all_word = []

for list_word in covid_19_df['notes'] :

    for word_list in list_word :        

        list_all_word.append(word_list)
reason = []

reason_occr =[]

freq = FreqDist(list_all_word)

for x in freq :

   reason.append(x)

   reason_occr.append(freq[x])



data = {'reason':reason ,'reason_occr':reason_occr } 

df_reason = pd.DataFrame(data)

df_reason = df_reason[df_reason['reason_occr'] > 50]

df_reason
fig = px.bar(df_reason.sort_values('reason_occr', ascending=False).sort_values('reason_occr', ascending=True), 

             x="reason_occr", y="reason", 

             title='Frequency of words in notes of Corona patients', 

             text='reason_occr', 

             orientation='h', 

             width=800, height=700, range_x = [0, max(df_reason['reason_occr'])])

fig.update_traces(marker_color='#46cdfb', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
#covid_19_df['detected_state'].value_counts()

covid_19_df['diagnosed_date'] = pd.to_datetime(covid_19_df['diagnosed_date'],format='%d/%m/%Y')



count_ss = 0

covid_19_df['no'] = 1

covid_19_df['Delhi_SS'] = 0

for list_word in covid_19_df['notes'] :

   covid_19_df['Delhi_SS'][count_ss] = 0

   for  word_list in list_word :

        if word_list == 'delhi' :

          covid_19_df['Delhi_SS'][count_ss] = 1

   count_ss = count_ss + 1
covid_19_df = covid_19_df[covid_19_df['diagnosed_date'] > '2020-02-27']

covid_19_df_ss = covid_19_df[covid_19_df['Delhi_SS'] == 1]
def gen_xaxis(title):

    """

    Creates the X Axis layout and title

    """

    xaxis = dict(

            title=title,

            titlefont=dict(

                color='#AAAAAA'

            ),

            showgrid=False,

            color='#AAAAAA',

            )

    return xaxis





def gen_yaxis(title):

    """

    Creates the Y Axis layout and title

    """

    yaxis=dict(

            title=title,

            titlefont=dict(

                color='#AAAAAA'

            ),

            showgrid=False,

            color='#AAAAAA',

            )

    return yaxis





def gen_annotations(annot):

    """

    Generates annotations to insert in the chart

    """

    if annot is None:

        return []

    

    annotations = []

    # Adding labels

    for d in annot:

        annotations.append(dict(xref='paper', x=d['x'], y=d['y'],

                           xanchor='left', yanchor='bottom',

                           text= d['text'],

                           font=dict(size=13,

                           color=d['color']),

                           showarrow=False))

    return annotations
annotations = [{'x': "2020-03-28", 'y': 500, 'text': 'Overall Count','color': 'gray'},

              {'x': "2020-03-28", 'y': 200, 'text': 'Contribution of superspread','color': 'mediumaquamarine'}]



title_text = ['<b>Patient count per day with Contribution of Super Spreader </b>', 'Date', 'Count per day']
fig = go.Figure(data=[

    go.Bar(name='Overall', x=covid_19_df['diagnosed_date'].value_counts().index, y=covid_19_df['diagnosed_date'].value_counts().values,marker={'color': 'gray'}),

    go.Bar(name='Delhi Incident', x=covid_19_df_ss['diagnosed_date'].value_counts().index, y=covid_19_df_ss['diagnosed_date'].value_counts().values,marker={'color': 'mediumaquamarine'})

])

# Change the bar mode

#fig.update_layout(barmode='group',showlegend=False,plot_bgcolor='rgb(240, 240, 240)',annotations = gen_annotations(annotations))

fig.update_layout(barmode='group',

                  showlegend=True,plot_bgcolor='rgb(240, 240, 240)',

                  title='<b>Overall count of patient per day with Contribution from Delhi Incident</b>',

                  xaxis=gen_xaxis('Date'),yaxis=gen_yaxis('Count per day'))

fig.show()