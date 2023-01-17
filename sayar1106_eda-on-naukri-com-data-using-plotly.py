import numpy as np

import pandas as pd

import plotly

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from nltk.tokenize import RegexpTokenizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        PATH = os.path.join(dirname, filename)



df = pd.read_csv(PATH)
df.head(10)
df.tail(10)
print("There are {} rows and {} columns in the dataset.".format(df.shape[0], df.shape[1]))
df.isnull().sum()
job_exp = df["Job Experience Required"].value_counts().nlargest(n=10)

job_exp_all = df["Job Experience Required"].value_counts()

fig = make_subplots(1,2, 

                    subplot_titles = ["Top 10 experience ranges", 

                                      "All experience ranges"])

fig.append_trace(go.Bar(y=job_exp.index,

                          x=job_exp, 

                          orientation='h',

                          marker=dict(color=job_exp.values, coloraxis="coloraxis", showscale=False),

                          texttemplate = "%{value:,s}",

                          textposition = "inside",

                          name="Top 10 experience ranges",

                          showlegend=False),

                

                 row=1,

                 col=1)

fig.update_traces(opacity=0.7)

fig.update_layout(coloraxis=dict(colorscale='tealrose'))

fig.append_trace(go.Scatter(x=job_exp_all.index,

                          y=job_exp_all, 

                          line=dict(color="#008B8B",

                                    width=2),

                          showlegend=False),

                 row=1,

                 col=2)

fig.update_layout(showlegend=False)

fig.show()
role = df['Role Category'].value_counts().nlargest(n=10)

fig = px.pie(role, 

       values = role.values, 

       names = role.index, 

       title="Top 10 Role Categories", 

       color=role.values,

       color_discrete_sequence=px.colors.qualitative.Prism)

fig.update_traces(opacity=0.7,

                  marker_line_color='rgb(8,48,107)',

                  marker_line_width=1.5)

fig.update_layout(title_x=0.5)

fig.show()
location = df['Location'].value_counts().nlargest(n=10)

fig = px.bar(y=location.values,

       x=location.index,

       orientation='v',

       color=location.index,

       text=location.values,

       color_discrete_sequence= px.colors.qualitative.Bold)



fig.update_traces(texttemplate='%{text:.2s}', 

                  textposition='outside', 

                  marker_line_color='rgb(8,48,107)', 

                  marker_line_width=1.5, 

                  opacity=0.7)



fig.update_layout(width=800, 

                  showlegend=False, 

                  xaxis_title="City",

                  yaxis_title="Count",

                  title="Top 10 cities by job count")

fig.show()
title = df['Job Title'].value_counts().nlargest(n=10)

fig = make_subplots(2, 1,

                    subplot_titles=["Top 10 Job Titles", 

                                    "Top Job Titles in Mumbai and Bengaluru"])

fig.append_trace(go.Bar(y=title.index,

                        x=title.values,

                        orientation='h',

                        marker=dict(color=title.values, coloraxis="coloraxis"),

                        texttemplate = "%{value:,s}",

                        textposition = "inside",

                        showlegend=False),

                  row=1,

                  col=1)

fig.update_layout(coloraxis_showscale=False)

fig.update_layout(height=800, 

                  width=800,

                  yaxis=dict(autorange="reversed"),

                  coloraxis=dict(colorscale='geyser'),

                  coloraxis_colorbar=dict(yanchor="top", y=1, x=0)

)



fig.append_trace(go.Bar(y=df[df['Location'].isin(['Mumbai'])]['Job Title'].value_counts().nlargest(n=10).index,

                        x=df[df['Location'].isin(['Mumbai'])]['Job Title'].value_counts().nlargest(n=10).values,

                        marker_color='#008080',

                        orientation='h',

                        showlegend=True,

                        name="Mumbai"),

                row=2,

                col=1)



fig.append_trace(go.Bar(y=df[df['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).index,

                        x=df[df['Location'].isin(['Bengaluru'])]['Job Title'].value_counts().nlargest(n=10).values,

                        marker_color='#00CED1',

                        orientation='h',

                        showlegend=True,

                        name="Bengaluru"),

                row=2,

                col=1,

                )

fig.update_layout(legend=dict(x=1,

                              y=0.3))

fig.show()
functional_words = df['Functional Area'].dropna().to_list()

tokenizer = RegexpTokenizer(r'\w+')

tokenized_list = [tokenizer.tokenize(i) for i in functional_words]

tokenized_list = [w for l in tokenized_list for w in l]



tokenized_list = [w.lower() for w in tokenized_list]

string = " ".join(w for w in tokenized_list)

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', 

                min_font_size = 10).generate(string) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
skills = df['Key Skills'].to_list()

skills = [str(s) for s in skills]

skills = [s.strip().lower()  for i in skills for s in i.split("|")]

string = " ".join(w for w in skills)

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', 

                min_font_size = 10).generate(string) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 