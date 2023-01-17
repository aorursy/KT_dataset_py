# Install cord19q project
!pip install git+https://github.com/neuml/cord19q

# Install scispacy model
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_md-0.2.5.tar.gz
import os
import shutil

from paperetl.cord19.execute import Execute as Etl

# Copy study design models locally
os.mkdir("cord19q")
shutil.copy("../input/cord19-study-design/attribute", "cord19q")
shutil.copy("../input/cord19-study-design/design", "cord19q")

# Build SQLite database for metadata.csv and json full text files
Etl.run("../input/CORD-19-research-challenge", "cord19q", "cord19q", "../input/cord-19-article-entry-dates/entry-dates.csv", False)
import shutil

from paperai.index import Index

# Copy vectors locally for predictable performance
shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")

# Build the embeddings index
Index.run("cord19q", "/tmp/cord19-300d.magnitude")
import os

# Workaround for mdv terminal width issue
os.environ["COLUMNS"] = "80"

from paperai.highlights import Highlights
from paperai.tokenizer import Tokenizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pycountry

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
def wordcloud(df, title = None):
    # Set random seed to have reproducible results
    np.random.seed(64)
    
    wc = WordCloud(
        background_color="white",
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

# Dataframe plot
def plot(df, title, kind="bar", color="bbddf5"):
    # Remove top and right border
    ax = plt.axes()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set axis color
    ax.spines['left'].set_color("#bdbdbd")
    ax.spines['bottom'].set_color("#bdbdbd")

    df.plot(ax=ax, title=title, kind=kind, color=color);

# Pie plot
def pie(labels, sizes, title):
    patches, texts = plt.pie(sizes, colors=["#4caf50", "#ff9800", "#03a9f4", "#011ffd", "#ff2079", "#7e57c2", "#fdd835"], startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.title(title)
    plt.show()
    
# Map visualization
def mapplot(df, title, bartitle):
    fig = go.Figure(data=go.Choropleth(
        locations = df["Code"],
        z = df["Count"],
        text = df["Country"],
        colorscale = [(0,"#fffde7"), (1,"#f57f17")],
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
import pandas as pd
import sqlite3

# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Articles
pd.set_option("max_colwidth", 125)
articles = pd.read_sql_query("select * from articles where tags is not null LIMIT 5", db)
articles
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Sections
pd.set_option("max_colwidth", 125)
sections = pd.read_sql_query("select * from sections where tags is not null LIMIT 5", db)
sections
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

# Select data
articles = pd.read_sql_query("select title from articles where tags is not null and title is not null", db)

# Build word frequencies on filtered tokens
freqs = pd.Series(np.concatenate([tokenize(x) for x in articles.Title])).value_counts()
wordcloud(freqs, "Most frequent words in article titles tagged as COVID-19")
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

sections = pd.read_sql_query("select text from sections where tags is not null", db)

# Filter tokens to only country names. Build dataframe of Country, Count, Code
mentions = pd.Series(np.concatenate([countrynames(x) for x in sections.Text])).value_counts()
mentions = mentions.rename_axis("Country").reset_index(name="Count")
mentions["Code"] = [countrycode(x) for x in mentions["Country"]]

# Set max to 1000 to allow shading for multiple countries
mentions["Count"] = mentions["Count"].clip(upper=2500)

mapplot(mentions, "Tagged Articles by Country Mentioned", "Articles by Country")
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select source from articles where tags is not null", db)

freqs = articles.Source.value_counts().sort_values(ascending=True)
plot(freqs, "Tagged Articles by Source", "barh", "#1976d2")
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select case when (Publication = '' OR Publication IS NULL) THEN '[None]' ELSE Publication END AS Publication from articles where tags is not null", db)

freqs = articles.Publication.value_counts().sort_values(ascending=True)[-20:]

plot(freqs, "Tagged Articles by Publication", "barh", "#7e57c2")
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query("select strftime('%Y-%m', published) as Published from articles where tags is not null and published >= '2020-01-01' order by published", db)

freqs = articles.Published.value_counts().sort_index()
plot(freqs, "Tagged Articles by Publication Month", "bar", "#ff9800")
# Connect to database
db = sqlite3.connect("cord19q/articles.sqlite")

articles = pd.read_sql_query('select count(*) as count, case when design=1 then "systematic review" when design in (2, 3) then "control trial" ' + 
                             'when design in (4, 5) then "prospective studies" when design=6 then "retrospective studies" ' +
                             'when design in (7, 8) then "case series" else "modeling" end as design from articles ' +
                             'where tags is not null and design > 0 group by design', db)

articles = articles.groupby(["design"]).sum().reset_index()

# Plot a pie chart of study types
pie(articles["design"], articles["count"], "Tagged Articles by Study Design")

from paperai.embeddings import Embeddings

embeddings = Embeddings()
embeddings.load("cord19q")

vectors = embeddings.vectors

pd.DataFrame(embeddings.vectors.most_similar("covid-19", topn=10), columns=["key", "value"])

vectors.similarity("coronavirus", ["sars", "influenza", "ebola", "phone"])
sentence1 = "Range of incubation periods for the disease in humans"
sentence2 = "The incubation period of 2019-nCoV is generally 3-7 days but no longer than 14 days, and the virus is infective during the incubation period"

embeddings.similarity(Tokenizer.tokenize(sentence1), [Tokenizer.tokenize(sentence2)])
sentence1 = "Range of incubation periods for the disease in humans"
sentence2 = "The medical profession is short on facemasks during this period, more are needed"

embeddings.similarity(Tokenizer.tokenize(sentence1), [Tokenizer.tokenize(sentence2)])
from paperai.query import Query

# Execute a test query
Query.run("antiviral covid-19 success treatment", 5, "cord19q")
%%capture --no-display

from paperai.report.execute import Execute as Report
from IPython.display import display, Markdown

query = """
name: query

antiviral covid-19 success treatment:
    query: antiviral covid-19 success treatment
    columns:
        - name: Date
        - name: Study
        - name: Study Type
        - name: Sample Size
        - name: Study Population
        - name: Matches
        - name: Entry
"""

# Execute report query
Report.run(query, 10, "md", "cord19q")

# Render report
display(Markdown(filename="query.md"))