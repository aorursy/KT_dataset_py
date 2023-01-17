# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import time

from pandasql import sqldf 

import plotly.express as px 

import datetime

import plotly.graph_objects as go



from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt


### used to iteratively make word clouds for each rating level

def cloudMaker(text): 

    wordcloud = WordCloud(

        width = 500,

        height = 500,

        background_color = 'white',

        stopwords = STOPWORDS).generate(str(text))

    fig = plt.figure(

        figsize = (10, 5),

        facecolor = 'k',

        edgecolor = 'k')

    plt.imshow(wordcloud, interpolation = 'bilinear')

    plt.axis('off')

    plt.tight_layout(pad=0)

    plt.show()



# for calling cloud maker function for each rating level 

def textValues(split_text): 

    for rating in split_text: 

        cloudMaker(rating.one_liner.values)



# for iteratively making clouds 

def cloudMakers(array): 

    for clouds in array: 

        textValues(clouds)

        

# for splitting ratings into individual rating levels 

def star_splitter(cdf): 

    

    ratings = [1,2,3,4,5]

    df = cdf 

    index = 0

    split_stars = []

    

    for rating in ratings: 

        stars = ratings[index]

        query = "SELECT one_liner from %s WHERE stars = %d" %(df, stars)

        temp_df = sqldf(query)

        split_stars.append(temp_df)

        index += 1

    return split_stars 



# for making all the graphs 

def graphsMaker(review_file): 

    cleaned_reviews = pd.read_csv(review_file) 

    cleaned_reviews.dtypes

    rating_query = sqldf("SELECT month, COUNT(stars) AS num_reviews, AVG(stars) AS avg_rating from cleaned_reviews GROUP BY month")

    rating_query.head()

    rating_table = go.Figure(data=[go.Table(header=dict(values=['Month', 'Total Reviews', 'Average Rating']),

                     cells=dict(values=[rating_query['month'], rating_query['num_reviews'],rating_query['avg_rating']]))

                         ])

    rating_table.show()

    

    rating_query2 = sqldf("SELECT month, stars, COUNT(stars) AS num_reviews from cleaned_reviews GROUP BY month, stars")

    rating_query2.head()





    ratings_ot = px.line(rating_query2, x="month", y="num_reviews", title='Ratings Over Time', color='stars')

    ratings_ot.add_trace(go.Scatter(x=rating_query.month, y=rating_query.num_reviews,

                        mode='lines',

                        name='Total Number of Ratings', line=dict(color='orange', width=1.5, dash='dash')))

    ratings_ot.show()



    avg_ratings = px.line(rating_query, x="month", y="avg_rating", title='Average Rating Over Time')

    avg_ratings.show()



    one_stars = sqldf("SELECT month, COUNT(stars) AS one_count from cleaned_reviews WHERE stars='1' GROUP BY month")

    two_stars = sqldf("SELECT month, COUNT(stars) AS two_count from cleaned_reviews WHERE stars='2' GROUP BY month")

    three_stars = sqldf("SELECT month, COUNT(stars) AS three_count  from cleaned_reviews WHERE stars='3' GROUP BY month")

    four_stars = sqldf("SELECT month, COUNT(stars) AS four_count from cleaned_reviews WHERE stars='4' GROUP BY month")

    five_stars = sqldf("SELECT month, COUNT(stars) AS five_count  from cleaned_reviews WHERE stars='5' GROUP BY month")



    join1 = sqldf("SELECT  A.month, A.one_count, B.two_count from one_stars AS A LEFT JOIN two_stars AS B on A.month = B.month ")

    join2 = sqldf("SELECT A.*,  B.three_count from join1 AS A LEFT JOIN three_stars AS B on A.month = B.month ")

    join3 = sqldf("SELECT A.*,  B.four_count from join2 AS A LEFT JOIN four_stars AS B on A.month = B.month ")

    join4 = sqldf("SELECT A.*,  B.five_count from join3 AS A LEFT JOIN five_stars AS B on A.month = B.month ")



    columns_to_sum = ['one_count', 'two_count', 'three_count', 'four_count', 'five_count']

    join4["sum_count"] = join4[columns_to_sum].sum(axis=1)



    join4['one_percent'] = round(join4.one_count / join4.sum_count,2)

    join4['two_percent'] = round(join4.two_count / join4.sum_count,2)

    join4['three_percent'] = round(join4.three_count / join4.sum_count,2)

    join4['four_percent'] = round(join4.four_count / join4.sum_count,2)

    join4['five_percent'] = round(join4.five_count / join4.sum_count,2)



    rating_percent = sqldf("SELECT month, sum_count, one_percent, two_percent, three_percent, four_percent, five_percent from join4")

    rating_percent.fillna(0, inplace=True)



    ratings_dist = cleaned_reviews.stars



    fig = px.histogram(ratings_dist, x="stars", title="Ratings Distribution")

    fig.show()
doordash_df = pd.read_csv('../input/chow-sentiment/0.csv')

postmates_df = pd.read_csv('../input/chow-sentiment/1.csv')

ubereats_df = pd.read_csv('../input/chow-sentiment/full_reviews.csv')

chowbus_df = pd.read_csv("../input/chow-sentiment/full_reviews.csv")
doordash_df['company'] = 'doordash'

postmates_df['company'] = 'postmates'

ubereats_df['company'] = 'ubereats'

chowbus_df['company'] = 'chowbus'




chowbus_avg = sqldf("SELECT month, AVG(stars) as avg_rating, company from chowbus_df GROUP BY month")

doordash_avg = sqldf("SELECT month, AVG(stars) as avg_rating, company from doordash_df GROUP BY month")

postmates_avg = sqldf("SELECT month, AVG(stars) as avg_rating, stars, company from postmates_df GROUP BY month")

ubereats_avg = sqldf("SELECT month, AVG(stars) as avg_rating, stars, company from ubereats_df GROUP BY month")
fig = px.line(chowbus_avg, x="month", y="avg_rating", title='Chowbus Ratings Over Time')

fig.add_trace(go.Scatter(x=doordash_avg.month, y=doordash_avg.avg_rating,

                    mode='lines',

                    name='doordash', line=dict(color='red', width=1.5)))

fig.add_trace(go.Scatter(x=postmates_avg.month, y=postmates_avg.avg_rating,

                    mode='lines',

                    name='postmates', line=dict(color='orange', width=1.5)))

fig.add_trace(go.Scatter(x=ubereats_avg.month, y=ubereats_avg.avg_rating,

                    mode='lines',

                    name='ubereats', line=dict(color='black', width=1.5)))

fig.show()
graphsMaker('../input/chow-sentiment/full_reviews.csv')
graphsMaker('../input/chow-sentiment/0.csv')
graphsMaker('../input/chow-sentiment/1.csv')
graphsMaker('../input/chow-sentiment/2.csv')
doordash_split = star_splitter('doordash_df')

postmates_split = star_splitter('postmates_df')

ubereats_split = star_splitter('ubereats_df')

chowbus_split = star_splitter('chowbus_df')
textValues(chowbus_split)
textValues(doordash_split)
textValues(postmates_split)
textValues(ubereats_split)