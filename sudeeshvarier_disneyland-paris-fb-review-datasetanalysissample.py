# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import plotly.plotly as py1
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools

import cufflinks as cf
cf.go_offline()

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/facebook_reviews_disneylandParis_format.csv", engine='python')
df.head()
#I notice that in this data set most of the information is not in english
# This means we can have a look at how many different languages were used to write reviews.
Langgroup = df.groupby(['review_lang']).size()
#Langgroup.index
Group_of_Languages = pd.DataFrame({'labels': Langgroup.index,
                                   'values': Langgroup.values })

Group_of_Languages.iplot(kind='pie',
                         labels='labels',
                         values='values', 
                         title='People Grouped by Languages')
#The maximum number of reviewers are local and belong to France. 
#Note: all the users have not given reviews. But everyone has given a proper star rating. 
count_of_columns = df.count().sort_values(ascending=False)
count_of_columns
#df.count().sort_values(ascending=False).apply(lambda x: x*100)

#Hence we will look at what percent of users have actually given the review
percent_reviewers = count_of_columns['review']/count_of_columns['user_id']*100
percent_reviewers
#Only 5% of the people have actually have expressed their responses. Not enough response to validate the effectiveness of reviews
#Most of the users have preferred to just give a star rating than write a review.
reviewstars = df.groupby(['stars']).size()
#Langgroup.index
star_rating_category = pd.DataFrame({'labels': reviewstars.index,
                                   'values': reviewstars.values })

star_rating_category.iplot(kind='pie',
                         labels='labels',
                         values='values', 
                         title='Grouped by star ratings')
import matplotlib.pyplot as plt
%matplotlib inline

f, g = plt.subplots(figsize=(12, 9))
df['stars'].value_counts().plot.bar(color="green")
g.set_xticklabels(g.get_xticklabels(), rotation=0)
plt.title("Categorization by rating")
plt.show(g)
# The difference between the 4 star rating and 5 star rating is almost 1:3. This is very good. Most of the customers think Disneyland Paris is very good. 
f, g = plt.subplots(figsize=(12, 9))
df['day_of_week'].value_counts().plot.bar(color="blue")
g.set_xticklabels(g.get_xticklabels(), rotation=90)
plt.title("Categorization day of the week when the review was submitted")
plt.show(g)
#The maximum number of reviews were submitted on a Sunday
plt.hist(df['hour_of_day'], bins=12, histtype='bar', orientation='vertical', label='Time of day when review is submitted');
#Most of the reviews are recieved during the evening hours. 
#Number of people who have written reviews apart from giving a star rating.
df['review_nb_words'].fillna('0', inplace=True)
df['review_nb_words'].nunique()

###########################################################################################################################################################################################################################
##############################################################################-------Summary-----#########################################################################################################################
###########################################################################################################################################################################################################################
#To summarize, a little less than 6% of 30000 people who visited Disneyland have submitted their written reviews. 76.2% of them gave 5 star reviews. 52% of the visting population are locals, in this case french. 
#Most of the remaining 48% are europeans. Most of the people prefer to arrive during weekends and since most of the reviews seems to have been done on weekends. 
#70 to 80% of the reviews were given during evening hours which makes sense as the customers who enjoyed their time at Disneyland should have given the reviews. 
###########################################################################################################################################################################################################################
