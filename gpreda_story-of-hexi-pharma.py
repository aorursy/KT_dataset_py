import os

import re

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

import warnings 

warnings.simplefilter("ignore")
data_df = pd.read_csv("/kaggle/input/public-tenders-romania-20072016/contracts.csv", low_memory=False)
pattern = re.compile(r'^HEXI')

company_list = list(data_df.Winner.unique())

matches = [x for x in company_list if pattern.match(x)]

print(f"Matching Winner names: {matches}")
hexi_df = data_df.loc[data_df.Winner.isin(matches)]

print(f"Hexi Pharma appears for {hexi_df.shape[0]} times in the data.")
def plot_count(feature, title, df, size=1, rotation=False, order=True):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    if order:

        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:31], palette='Set3')

    else:

        g = sns.countplot(df[feature],  palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if rotation:

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 0.2,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("Winner", "aparition of Hexi Pharma (with different names)", hexi_df, size=2, rotation=True)
plot_count("Type", "- (Award) Type", hexi_df, size=2)
plot_count("Contract_Type", "contracts - grouped by Contract type", hexi_df, size=1)
plot_count("Procedure_Type", "Procedure type", hexi_df, size=2, rotation=True)
plot_count("Contracting_Authority", "Contracting Authority (top 30)", hexi_df, size=4, rotation=True)
plot_count("Contracting_Authority_Type", "Contracting Authority Type (first 30)", hexi_df, size=4, rotation=True)
plot_count("Contracting_Authority_Activity_Type", "Contracting Authority Activity Type", hexi_df, size=3, rotation=True)


hexi_df["Award_Announcement_Date"] = hexi_df["Award_Announcement_Date"].apply(lambda x: datetime.strptime(x[0:10], '%Y-%m-%d'))

hexi_df["Award_Announcement_Year"] = hexi_df["Award_Announcement_Date"].dt.year

hexi_df["Award_Announcement_Month"] = hexi_df["Award_Announcement_Date"].dt.month
plot_count("Award_Announcement_Year", " contracts grouped by Year", hexi_df, size=2, order=False)
plot_count("Award_Announcement_Month", " contracts grouped by Month", hexi_df, size=2, order=False)
plot_count("Currency", " contracts grouped by Currency", hexi_df, size=1)
plot_count("Offers_Number", " contracts grouped by Offers Number (first 30)", hexi_df, size=4)
plot_count("Financing_Type", " contracts grouped by Financing Type (first 30)", hexi_df, size=2)
plot_count("Financing_Method", " contracts grouped by Financing Method (first 30)", hexi_df, size=4, rotation=True)
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(hexi_df["Contract_Title"], "Prevalent words in the contract")
hexi_df[["Contracting_Authority", "Award_Announcement_Year", "Value"]].sort_values(by=["Value"], ascending=False).head(10)
hexi_df.loc[(hexi_df.Contracting_Authority=="SPITAL CLINIC JUDETEAN DE URGENTA ORADEA")&(hexi_df.Award_Announcement_Year==2015)][["Contract_Title", "Procedure_Type", "Value", "Award_Announcement_Month"]]
hexi_df[["Contract_Title"]].value_counts()[:10]
agg_data = hexi_df.groupby(["Winner", "Winner_Country", "Award_Announcement_Year", "Contracting_Authority", "Contracting_Authority_Activity_Type"])["Value_RON"].agg(["sum", "count"])
agg_data_df = pd.DataFrame(agg_data).reset_index()

agg_data_df.columns = ["Winner", "Winner Country", "Year", "Contracting Authority","Activity Type", "Total", "Count"]
agg_data_df.sort_values(by=["Total"], ascending=False).head(10)
agg_data_df.sort_values(by=["Count"], ascending=False).head(10)
import plotly.express as px

fig = px.scatter(agg_data_df, x='Total', y='Count', color='Contracting Authority', size='Total', size_max = 50,

                hover_name='Contracting Authority', animation_frame='Year',

                 title="Winner contract numbers vs. total amount grouped by Activity Type, per year",

                 animation_group='Activity Type', range_x=[-2000, 30000000], range_y=[-1, 16],

                width=800, height=800)

fig.update_layout(legend=dict(orientation="h",  yanchor="bottom", y=-1, xanchor="right", x=1, font=dict(family="Courier", size=12,color="black")))

fig.show()
agg2_data = hexi_df.groupby(["Award_Announcement_Year", "Contracting_Authority_Activity_Type"])["Value_RON"].agg(["sum", "count"])
agg2_data_df = pd.DataFrame(agg2_data).reset_index()

agg2_data_df.columns = ["Year", "Activity Type", "Total", "Count"]

agg2_data_df.head()
agg2_data_df.sort_values(by=["Total"], ascending=False).head(10)
agg2_data_df.sort_values(by=["Count"], ascending=False).head(10)
agg_data_df.loc[agg_data_df.Year==2010].sort_values(by=["Count"], ascending=False).head(10)
agg_data_df.loc[agg_data_df.Year==2010].sort_values(by=["Total"], ascending=False).head(10)
import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
def plot_time_variation(df, y='Total', hue='Activity Type', size=1, title=""):

    

    data = []

    groups = df[hue].unique()

    

    for group in groups:

        df_ = df[(df[hue]==group)] 

        traceS = go.Bar(

            x = df_['Year'],y = df_[y],

            name=group,

            marker=dict(

                        line=dict(

                            color='black',

                            width=0.75),

                        opacity=0.7,

                    ),

            text=df_[hue],

        )

        data.append(traceS)

    layout = dict(title = title,

              xaxis = dict(title = 'Year', showticklabels=True), 

              yaxis = dict(title = f'{y}'),

              hovermode = 'closest',

              barmode='stack'

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='total-count')
plot_time_variation(agg2_data_df, 'Total', 'Activity Type', 4, "Total amount of contracts (2007-2016)")
plot_time_variation(agg2_data_df, 'Count', 'Activity Type', 4, "Number of contracts (2007-2016)")
heatmap = agg2_data_df.pivot("Year", "Activity Type", 'Total')

fig, ax = plt.subplots(nrows=1,figsize=(14,8))

sns.heatmap(heatmap, linewidths=.5)

plt.title("Winner Contracts Total (RON) grouped by Activity Type per Year")

plt.xticks(rotation=90, size=8)



plt.show()
heatmap = agg2_data_df.pivot("Year", "Activity Type", 'Count')

fig, ax = plt.subplots(nrows=1,figsize=(14,8))

sns.heatmap(heatmap, linewidths=.5)

plt.title("Winner Contracts Number grouped by Activity Type per Year")

plt.xticks(rotation=90, size=8)



plt.show()
filtered_agg_data_df = agg2_data_df.loc[agg2_data_df.Total>1]

filtered_agg_data_df.shape
import plotly.express as px

fig = px.scatter(filtered_agg_data_df, x='Total', y='Count', color='Activity Type', size='Total', size_max = 50,

                hover_name='Activity Type', log_x=True, animation_frame='Year',

                 title="Winner contract numbers vs. total amount grouped by Activity Type, per year",

                 animation_group='Activity Type', range_x=[1, 80000000], range_y=[-1, 100],

                width=800, height=800)

fig.show()