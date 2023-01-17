# Install paperai project

!pip install git+https://github.com/neuml/paperai
from paperai.highlights import Highlights

from paperai.tokenizer import Tokenizer



from nltk.corpus import stopwords



from wordcloud import WordCloud



import matplotlib.pyplot as plt

import numpy as np

import plotly.graph_objects as go

import pycountry



STOP_WORDS = set(stopwords.words("english")) 



# Tokenizes text and removes stopwords

def tokenize(text, case_sensitive=False):

    # Get list of accepted tokens

    tokens = [token for token in Tokenizer.tokenize(text) if token not in Highlights.STOP_WORDS]

    

    if case_sensitive:

        # Filter original tokens to preserve token casing

        return [token for token in text.split() if token.lower() in tokens]



    return tokens



# Country data

countries = [c.name for c in pycountry.countries]

countries = countries + ["USA"]



# Lookup country name for alpha code. If already an alpha code, return value

def countryname(x):

    country = pycountry.countries.get(alpha_3=x)

    return country.name if country else x

    

# Resolve alpha code for country name

def countrycode(x):

    return pycountry.countries.get(name=x).alpha_3



# Tokenize and filter only country names

def countrynames(x):

    return [countryname(country) for country in countries if country.lower() in x.lower()]



# Word Cloud colors

def wcolors(word=None, font_size=None, position=None,  orientation=None, font_path=None, random_state=None):

    colors = ["#7e57c2", "#03a9f4", "#011ffd", "#ff9800", "#ff2079"]

    return np.random.choice(colors)



# Word Cloud visualization

def wordcloud(df, title, recent):

    # Set random seed to have reproducible results

    np.random.seed(64)

    

    wc = WordCloud(

        background_color="white" if recent else "black",

        max_words=200,

        max_font_size=40,

        scale=5,

        random_state=0

    ).generate_from_frequencies(df)



    wc.recolor(color_func=wcolors)

    

    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')



    if title:

        fig.suptitle(title, fontsize=14)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wc),

    plt.show()



# Map visualization

def mapplot(df, title, bartitle, color1, color2):

    fig = go.Figure(data=go.Choropleth(

        locations = df["Code"],

        z = df["Count"],

        text = df["Country"],

        colorscale = [(0, color1), (1, color2)],

        showscale = False,

        marker_line_color="darkgray",

        marker_line_width=0.5,

        colorbar_title = bartitle,

    ))



    fig.update_layout(

        title={

            'text': title,

            'y':0.9,

            'x':0.5,

            'xanchor': 'center',

            'yanchor': 'top'},

        geo=dict(

            showframe=False,

            showcoastlines=False,

            projection_type='equirectangular'

        )

    )

    

    fig.show(config={"displayModeBar": False, "scrollZoom": False})

# Build a word cloud for Top 25 cited articles

def citecloud(recent):

    # Connect to database

    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")



    # Citations

    citations = pd.read_sql_query("select text from sections where article in " + 

                                "(select a.id from articles a join citations c on a.title = c.title " + 

                                 "where tags is %s null %s order by mentions desc limit 25)" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 

                                                                                                                        "'World Health Organization'" if not recent else ""), db)

    freqs = pd.Series(np.concatenate([tokenize(x) for x in citations.Text])).value_counts()

    wordcloud(freqs, "Most Frequent Words In Highly Cited %s Papers" % ("COVID-19" if recent else "Historical"), recent)
# Show top countries for Top 25 cited articles

def citemap(recent):

    # Connect to database

    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")



    sections = pd.read_sql_query("select text from sections where article in (select id from articles a join citations c on a.title = c.title " + 

                                 "where tags is %s null %s order by mentions desc limit 25)" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 

                                                                                                                         "'World Health Organization'" if not recent else ""), db)

    

    # Filter tokens to only country names. Build dataframe of Country, Count, Code

    mentions = pd.Series(np.concatenate([countrynames(x) for x in sections.Text])).value_counts()

    mentions = mentions.rename_axis("Country").reset_index(name="Count")

    mentions["Code"] = [countrycode(x) for x in mentions["Country"]]



    mapplot(mentions, "Highly Cited %s Papers - Country Mentioned" % ("COVID-19" if recent else "Historical"), "Articles by Country", 

            "#fffde7" if recent else "#ffcdd2", "#f57f17" if recent else "#b71c1c")
import datetime

import os

import sqlite3



import pandas as pd



from IPython.core.display import display, HTML



# Workaround for mdv terminal width issue

os.environ["COLUMNS"] = "80"



from paperai.query import Query



def design(df):

    # Study Design

    return "%s" % Query.design(df["Design"]) + ("<br/><br/>" + Query.text(df["Sample"]) if df["Sample"] else "")



def citations(recent):

    # Connect to database

    db = sqlite3.connect("../input/cord-19-analysis-with-sentence-embeddings/cord19q/articles.sqlite")



    # Citations

    citations = pd.read_sql_query("select published, authors, publication, a.title, reference, mentions as Cited from articles a join citations c on a.title = c.title " + 

                                  "where tags is %s null %s order by mentions desc limit 25" % ("not" if recent else "", "and published <= '2020-01-01' and a.title != " + 

                                                                                                                         "'World Health Organization'" if not recent else ""), db)

    citations["Published"] = citations["Published"].apply(Query.date)

    citations["Authors"] = citations["Authors"].apply(Query.authors)

    citations["Title"] = "<a href='" + citations["Reference"] + "'>" + citations["Title"] + "</a>"



    citations.style.bar(subset=["Cited"], color='#d65f5f')

    citations.style.hide_index()



    # Remove unnecessary columns

    citations = citations.drop("Reference", 1)



    # Set index to be 1-based

    citations.index = np.arange(1, len(citations) + 1)



    ## Show table as HTML

    display(HTML(citations.to_html(escape=False)))

citecloud(True)
citemap(True)
citations(True)