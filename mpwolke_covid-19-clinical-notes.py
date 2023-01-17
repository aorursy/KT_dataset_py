# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/covid-ctscans/metadata.csv', encoding='ISO-8859-2')

df.head()
import missingno as msno



p=msno.bar(df)
from colorama import Fore, Style



def count(string: str, color=Fore.RED):

    """

    Saves some work 😅

    """

    print(color+string+Style.RESET_ALL)
def statistics(dataframe, column):

    count(f"The Average value in {column} is: {dataframe[column].mean():.2f}", Fore.RED)

    count(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)

    count(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)

    count(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25)}", Fore.GREEN)

    count(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50)}", Fore.CYAN)

    count(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75)}", Fore.MAGENTA)
# Print Offset Column Statistics

statistics(df, 'offset')
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['offset'], color='blue')

plt.title(f"Offset Distribution [\u03BC : {df['offset'].mean():.2f} conditions | \u03C3 : {df['offset'].std():.2f} conditions]")

plt.xlabel("Offset")

plt.ylabel("Count")

plt.show()
# Print Age Column Statistics

statistics(df, 'age')
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(df['age'], color='red')

plt.title(f"age Distribution [\u03BC : {df['age'].mean():.2f} old | \u03C3 : {df['age'].std():.2f} old]")

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(df['lymphocyte_count'], color='green')

plt.title(f"Lymphocyte Count [\u03BC : {df['lymphocyte_count'].mean():.2f} increase | \u03C3 : {df['lymphocyte_count'].std():.2f} increase]")

plt.xlabel("Lymphocyte Ccount")

plt.ylabel("Count")

plt.show()
# Print  Column Statistics

statistics(df, 'lymphocyte_count')
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('pO2_saturation').size()/df['needed_supplemental_O2'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('age').size()/df['went_icu'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
plt.style.use("ggplot")

plt.figure(figsize=(18, 9))

sns.boxplot(df['offset'], df['age'])

plt.title("Age & Offset")

plt.xlabel("Offset")

plt.ylabel("Age")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(8, 4))

sns.set(style='ticks')

scatter_df = df[["age", "pO2_saturation", "leukocyte_count", "neutrophil_count"]]

sns.pairplot(scatter_df)

plt.show()
plt.figure(figsize=(8,4))

sns.countplot(x= 'sex', data = df, palette="cool",edgecolor="black")

plt.title('Sex Distribution')

plt.show()
sns.countplot(y=df.modality ,data=df)

plt.xlabel("count")

plt.ylabel("Exam Modality")

plt.show()
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['sex', 'intubated'], as_index=False).offset.sum()



fig = px.bar(plot_data, x='sex', y='offset', color='intubated', title='Intubated Patients by Sex')

fig.show()
from plotly.subplots import make_subplots





fig= make_subplots(rows= 2,cols=2, 

                    specs=[[{'secondary_y': True},{'secondary_y': True}],[{'secondary_y': True},{'secondary_y': True}]],

                    subplot_titles=("pO2_saturation","leukocyte_count","temperature","lymphocyte_count")

                   )

fig.add_trace(go.Bar(x=df['age'],y=df['pO2_saturation'],

                    marker=dict(color=df['pO2_saturation'],coloraxis='coloraxis')),1,1)



fig.add_trace(go.Bar(x=df['age'],y=df['leukocyte_count'],

                    marker=dict(color=df['leukocyte_count'],coloraxis='coloraxis1')),1,2)



fig.add_trace(go.Bar(x=df['age'],y=df['temperature'],

                    marker=dict(color=df['temperature'],coloraxis='coloraxis2')),2,1)



fig.add_trace(go.Bar(x=df['age'],y=df['lymphocyte_count'],

                    marker=dict(color=df['lymphocyte_count'],coloraxis='coloraxis3')),2,2)
ax = sns.countplot(x = 'location',data=df,order=['Cho Ray Hospital, Ho Chi Minh City, Vietnam', 'Changhua Christian Hospital, Changhua City, Ta...', 'Wuhan Jinyintan Hospital, Wuhan, Hubei Provinc..', 'Mount Sinai Hospital, Toronto, Ontario, Canada'])

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.2, p.get_height()))

plt.xticks(rotation=45)        
ax = sns.countplot(x = 'location',data=df,order=['Melbourne, Australia','Adelaide, Australia', 'Hungary', 'Calgary,Canada'])

for p in ax.patches:

        ax.annotate(format(p.get_height()), (p.get_x()+0.2, p.get_height()))

plt.xticks(rotation=45)        
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.clinical_notes)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='GnBu', background_color="white").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.other_notes)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='GnBu', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()