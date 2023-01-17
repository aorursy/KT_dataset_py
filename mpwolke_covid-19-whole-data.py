# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/non-pharmaceutical-data/COVID_WHOLE_DATA.csv")

df.head()
fig = plt.figure(figsize=[10,7])

sns.countplot(df['militarycareer'], color=sns.xkcd_rgb['greenish cyan'])

plt.title('Military career')

plt.show()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df.isna().sum()
# filling missing values with NA

df[['BTI_2020', 'smokers', 'gdp', 'av_temp', 'doctor_per_capita', 'nurse_per_capita', 'social_support', 'healthy_life_exp']] = df[['BTI_2020', 'smokers', 'gdp', 'av_temp', 'doctor_per_capita', 'nurse_per_capita', 'social_support', 'healthy_life_exp']].fillna('NA')
fig = px.bar(df[['elected', 'Confirmed']].sort_values('Confirmed', ascending=False), 

             y="Confirmed", x="elected", color='elected', 

             log_y=True, template='ggplot2', title='Confirmed Cases vs Elected')

fig.show()
fig = px.bar(df, 

             x='border_day', y='lockdown_date', color_discrete_sequence=['#D63230'],

             title='Covid-19 Confirmed and Lockdown date', text='Confirmed')

fig.show()
fig = px.bar(df,

             y='militarycareer',

             x='elected',

             orientation='h',

             color='population',

             title='Military Career & Elected',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.scatter(df, x="male", y="Fatalities", color="militarycareer")

fig.show()
fig = px.bar(df,

             y='school_day',

             x='Confirmed',

             orientation='h',

             color='lockdown_day',

             title='Covid-19 Confirmed & School Day',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
fig = px.scatter(df, x="elected", y="age", color="militarycareer", marginal_y="rug", marginal_x="histogram")

fig
fig = px.scatter(df, x="Fatalities", y="tenure_months", color="av_temp", title='Covid-19 Fatalities Average temperature tenure months', marginal_y="violin",

           marginal_x="box", trendline="ols")

fig.show()
df["age"] = df["Confirmed"]/100

fig = px.scatter(df, x="Confirmed", y="militarycareer", color="male", error_x="age", error_y="age")

fig.show()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )



plot_correlation_map(df) 
df.hist()
fig = px.line(df, x="av_hum", y="tenure_months", color_discrete_sequence=['green'], 

              title="Average humidity & Tenure Months")

fig.show()
fig = px.pie(df, values=df['Fatalities'], names=df['militarycareer'],

             title='Covid-19 Fatalities in Military Career',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.pie(df,

             values="av_hum",

             names="gdp",

             template="presentation",

             labels = {'av_hum' : 'militarycareer', 'Confirmed' : 'Fatalities'},

             color_discrete_sequence=['#4169E1', '#DC143C', '#006400'],

             width=800,

             height=450,

             hole=0.6)

fig.update_traces(rotation=180, pull=0.05, textinfo="percent+label")

py.offline.iplot(fig)
fig = px.pie(df, values='age', 

             names='male',

             hole=.5,

             color_discrete_sequence=px.colors.sequential.RdBu, 

             template="presentation")

fig.update_layout(title="Covid-19 Whole Data")

fig.show()
hist = df[['elected','lockdown_date']]

bins = range(hist.elected.min(), hist.elected.max()+10, 5)

ax = hist.pivot(columns='lockdown_date').elected.plot(kind = 'hist', stacked=True, alpha=0.5, figsize = (10,5), bins=bins, grid=False)

ax.set_xticks(bins)

ax.grid('on', which='major', axis='x')
bboxtoanchor=(1.1, 1.05)

#seaborn.set(rc={'axes.facecolor':'03fc28', 'figure.facecolor':'03fc28'})

df.plot.area(y=['elected','male','militarycareer', 'tenure_months', 'Confirmed', 'Fatalities'],alpha=0.4,figsize=(12, 6));
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.country)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Dark2', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()