# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import re

# trying my hands on plotly

from plotly import tools

import plotly.graph_objs as go

# TextBlob for quick text analysis

from textblob import TextBlob

# wordcloud for, well like the name suggests

from wordcloud import WordCloud, STOPWORDS

# matplotlib to visualize the word clouds

import matplotlib.pyplot as plt

# To generate word clouds using a mask object. I might not use this

from PIL import Image

# Wordcloud usually generates in various colors. this library might help to convert to a gray color

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the data

employee_reviews=pd.read_csv("../input/employee_reviews.csv", index_col=[0])

# Clean up the column headers

employee_reviews.columns = employee_reviews.columns.str.replace("-","_")



replace_none_as_nan = lambda t: float('nan') if t=="none" else t

employee_reviews.overall_ratings = employee_reviews.overall_ratings.map(replace_none_as_nan)

employee_reviews.work_balance_stars = employee_reviews.work_balance_stars.map(replace_none_as_nan)

employee_reviews.culture_values_stars = employee_reviews.culture_values_stars.map(replace_none_as_nan)

employee_reviews.carrer_opportunities_stars = employee_reviews.carrer_opportunities_stars.map(replace_none_as_nan)

employee_reviews.comp_benefit_stars = employee_reviews.comp_benefit_stars.map(replace_none_as_nan)

employee_reviews.senior_mangemnet_stars = employee_reviews.senior_mangemnet_stars.map(replace_none_as_nan)



# A quick view of the data to get the lay of the land, also because Excel has spoiled me that way

employee_reviews.head(5)
# Get the list of reviews per company

employee_companies = employee_reviews.company.value_counts()

# Prepare a Bar graph of the number of records

data = [go.Bar(

    x=employee_companies.index.str.title().tolist(),

    y=employee_companies.values

)]

layout = go.Layout(title="# Reviews by Organization",yaxis=dict(title="# Reviews"), xaxis=dict(title="Organizations"))

figure = go.Figure(data=data, layout=layout)

iplot(figure)
# Generate polarity and subjectivity number for just the summary column

employee_reviews["summary_polarity"] = employee_reviews.summary.apply(lambda t: TextBlob(str(t)).sentiment.polarity)

employee_reviews["summary_subjectivity"] = employee_reviews.summary.apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)
# A short function to generate histograms for the polarity and subjectivity for each company. I could repeat the statements over and over, but that would make the graph generation appear too large

def generate_histogram(for_column,company_name,opacity=0.5):

    '''

    returns a plotly Histogram object with the parameters specified

    

    for_column: Specify the columns with which the histogram must be generated. In this case, it would be either "summary_polarity" or "summary_subjectivity"

    

    company_name: Specify the company name in lower case. In this case, it would be one of the following: amazon, apple, facebook, google, microsoft, or netflix

    

    opacity: the opacity of each hisogram visualization. By default it will be 0.5 or 50% opaque

    '''

    return go.Histogram(

        x = employee_reviews[employee_reviews.company==company_name][for_column],

        opacity=opacity,

        xbins=dict(start=-1.0,end=1.1,size=0.2),

        name=company_name.title()

    )
# Generate a Histogram if Polarity for each company.



amazon_polarity=generate_histogram("summary_polarity","amazon")

apple_polarity=generate_histogram("summary_polarity","apple")

google_polarity=generate_histogram("summary_polarity","google")

facebook_polarity=generate_histogram("summary_polarity","facebook")

microsoft_polarity=generate_histogram("summary_polarity","microsoft")

netflix_polarity=generate_histogram("summary_polarity","netflix")

data=[

    amazon_polarity,

    apple_polarity,

    google_polarity,

    facebook_polarity,

    microsoft_polarity,

    netflix_polarity

]

layout = go.Layout(barmode="overlay", title="Summary Polarity by Company",xaxis=dict(title="Polarity"),yaxis=dict(title="# of Reviews"))

figure = go.Figure(data=data,layout=layout)

iplot(figure)
# Generate Subjectivity Histogram for each company

amazon_subjectivity=generate_histogram("summary_subjectivity","amazon")

apple_subjectivity=generate_histogram("summary_subjectivity","apple")

google_subjectivity=generate_histogram("summary_subjectivity","google")

facebook_subjectivity=generate_histogram("summary_subjectivity","facebook")

microsoft_subjectivity=generate_histogram("summary_subjectivity","microsoft")

netflix_subjectivity=generate_histogram("summary_subjectivity","netflix")

data=[

    amazon_subjectivity,

    apple_subjectivity,

    google_subjectivity,

    facebook_subjectivity,

    microsoft_subjectivity,

    netflix_subjectivity

]

layout = go.Layout(barmode="overlay", title="Summary Subjectivity by Company",xaxis=dict(title="Subjectivity"),yaxis=dict(title="# of Reviews"))

figure = go.Figure(data=data,layout=layout)

iplot(figure)
# Slightly larger function to generate wordclouds for the summary, pros, and cons.



# configure stop words aka words that dont need to be considered for word clouds

stopwords=set(STOPWORDS)

stopwords.add("let")

stopwords.add("to")

stopwords.add("from")

stopwords.add("a")

stopwords.add("an")

stopwords.add("the")

stopwords.add("of")



# The function itself

def generate_wordcloud(reviews, generate_by_frequency=False, addl_stopwords=[],):

    '''

    return a word cloud object that can be used in a matplotlib.pyplot's imshow function

    

    reviews: specify the entire employee_object for which the word cloud needs to be generated. Filtering must be done manually

    

    generate_by_frequency: default False. Configure whether the word cloud must be generated using a text string (WordCloud.generate_by_text) or if the word cloud must be generated using frequencies (WordCloud.generate_by_frequencies). See wordcloud documentation for further reference

    

    addl_stopwords: array of additonal words that must be added to the list of stop words

    '''

    

#   Add the additional stopwords to the stopword list

    for t in addl_stopwords:

        stopwords.add(str(t))



#   Combine all the reviews in the review series so that it becomes one really large text that can be passed to the wordcloud's generate function

    def format_reviews(review):

        processed_reviews = " ".join(str(t) for t in review)

        processed_reviews = processed_reviews.replace("\,+"," ").replace("\.+"," ").replace("\*+"," ").replace("\n+", " ")

        return processed_reviews

    

#   Function to generate the word cloud words in gray color instead of various colors

    def grey_color_func(word, font_size, position, orientation, random_state=None,**kwargs):

        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

    

#   Initialize the WordCloud object

    wc = WordCloud(background_color="black", max_words=100, stopwords=stopwords, max_font_size=40, random_state=42,width=600,height=200,margin=1)

    if generate_by_frequency:

        w_counts = TextBlob(format_reviews(reviews)).word_counts

        for t in stopwords:

            if t in w_counts:

                del w_counts[t]

        wc.generate_from_frequencies(w_counts)

    else:

        wc.generate_from_text(format_reviews(reviews))

    wc.recolor(color_func=grey_color_func, random_state=3)

    return wc
fig, ax = plt.subplots(nrows=6,ncols=3,figsize=(36,36))



# For each company name, generate a wordcloud of the company's summary, pros, and cons (in that order) and display the visualizations

for i in np.arange(employee_reviews.company.nunique()):

    company_name=employee_reviews.company.unique()[i]

    summary=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].summary, addl_stopwords=[company_name], generate_by_frequency=True)

    pros=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].pros, addl_stopwords=[company_name], generate_by_frequency=True)

    cons=generate_wordcloud(employee_reviews[employee_reviews.company==company_name].cons, addl_stopwords=[company_name], generate_by_frequency=True)

    ax[i,0].set_title("{0} Summary".format(company_name.title()),fontsize=36)

    ax[i,1].set_title("{0} Pros".format(company_name.title()),fontsize=36)

    ax[i,2].set_title("{0} Cons".format(company_name.title()),fontsize=36)

    ax[i,0].imshow(summary, interpolation='bilinear')

    ax[i,1].imshow(pros, interpolation='bilinear')

    ax[i,2].imshow(cons, interpolation='bilinear')

    ax[i,0].axis("off")

    ax[i,1].axis("off")

    ax[i,2].axis("off")
# A short function to generate bar graphs for ratings. That being said, this is also the largest code block in this notebook! (Yikes!)

def generate_bargraphs(for_rating):

    temp = pd.concat([

        employee_reviews[(employee_reviews["company"]==t)][for_rating].value_counts(normalize=True).rename(t.title()) for t in employee_reviews.company.unique().tolist()

    ],axis=1,sort=True);

    return [

        go.Bar(

            x=temp.T[t].index.tolist(),

            y=temp.T[t].values.tolist(),

            hovertemplate="%{y:.2%}",

            name=t,

#             orientation="h",

        )

        for t in temp.index];



rating_colorpalette=["#aff895","#89d471","#64b04e","#3f8e2b","#106d00"]



# Generate graphs for each rating

overall_rating_per_company = generate_bargraphs("overall_ratings")

work_balance_rating_per_company = generate_bargraphs('work_balance_stars')

culture_values_rating_per_company = generate_bargraphs('culture_values_stars')

carrer_opportunities_rating_per_company = generate_bargraphs('carrer_opportunities_stars')

comp_benefit_rating_per_company = generate_bargraphs('comp_benefit_stars')

senior_mangemnet_rating_per_company = generate_bargraphs('senior_mangemnet_stars')



# Generate Subplots

figure = tools.make_subplots(rows=2,cols=5,shared_yaxes=True,specs=[[{'colspan':5},{},{},{},{}],[{},{},{},{},{}]],

subplot_titles=("Overall Rating per Company","","","","","Work<br>Balance", "Culture<br>Values","Career<br>Opportunities","Compensation<br>Benefits","Senior<br>Management"))



# Stack the graphs so it forms a 100% stacked bar graph

figure.layout.barmode="stack"



# A decent height for the large plot

figure.layout.height=800



# Hide the y axis since the numbers are visible on hover

figure.layout.yaxis.visible=False

figure.layout.yaxis2.visible=False



# Set the palette for the ratings. 

# Figuring out how to add in the palette for a stacked graph took me around 350 failed tries. 

# Was trying so much with the Bar.marker.colorscales because I was using colorscales to set Pandas plot for the earlier visualizations

# Turns out that it was easier using the figure.layout.colorway and I still figure out how the figure.layout.colorscale fits into all this. In time.

figure.layout.colorway=rating_colorpalette



# Hide the legend because this subplot ends up showing a legend graph for each subplot

figure.layout.showlegend=False



# These subplots can only handle a single graph per subplot area. If the data is an array of plots, then iteration over each element is necessary

for t in overall_rating_per_company:

    figure.add_trace(t,1,1)

for t in work_balance_rating_per_company:

    figure.add_trace(t,2,1)

for t in culture_values_rating_per_company:

    figure.add_trace(t,2,2)

for t in carrer_opportunities_rating_per_company:

    figure.add_trace(t,2,3)

for t in comp_benefit_rating_per_company:

    figure.add_trace(t,2,4)

for t in senior_mangemnet_rating_per_company:

    figure.add_trace(t,2,5)



# Reaching the plot finally

iplot(figure);