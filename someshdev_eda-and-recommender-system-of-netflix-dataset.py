# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting of figures 

from matplotlib.dates import DateFormatter #for date fromatting on plots 

from datetime import datetime



import plotly.graph_objects as go

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#first we read in the dataset

netflix=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

print(netflix.shape)

netflix.head()
#Let's first look at the proprotion of movies vs tv shows 

labels=list(netflix['type'].unique())

values=pd.DataFrame(netflix['type'].value_counts())



#plot the pie chart 

fig1, ax1 = plt.subplots()

explode = (0, 0.1)

ax1.pie(values['type'],explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Proportion of Movies to Tv-shows on Netflix')

plt.show()
#want to see where majority of movie and tv shows are shown 



#First we pre process the column and make a list for each

for i in range (netflix.shape[0]):

    netflix['country'][i]=[x.strip() for x in str(netflix['country'][i]).split(",")]

    

#next create a dictionary of countries and count of countries 

country_data=netflix['country']

count_country={}

for i in range (len(country_data)):

    for j in country_data[i]:

                if j in count_country:

                    count_country[j]+=1

                else:

                    count_country[j]=1

    

  



#now we plot the top 10 countries 

count_country_final=pd.DataFrame(list(count_country.items()),columns = ['country','count']) 

count_country_final=count_country_final.sort_values(by='count',ascending=False)

#remove nan

indexNames = count_country_final[ count_country_final['country'] == 'nan' ].index

 

# Delete these row indexes from dataFrame

count_country_final.drop(indexNames , inplace=True)





#for plotting horizontal barchart

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

country = count_country_final['country'][:10]

value = count_country_final['count'][:10]

ax.barh(country,value)



plt.title('Top 10 countries having the most number of tv-shows and movies combined')

plt.gca().invert_yaxis()

plt.show()



def country_show_over_time(country):



    #first we filter out the country 

    index=[country in netflix['country'][i] for i in range(netflix.shape[0])]

    target=netflix.iloc[index]





    # now we need to find the cumulative count of shows over time. First we sort by date

    target['date_added']=pd.to_datetime(target['date_added'])

    target=target.sort_values(by='date_added')



    #create a column for cumulative numbers 

    target['cumsum']=1

    target['cumsum']=target['cumsum'].cumsum()

    

    #and finally plot the reuslts out 

    

    

    fig,ax=plt.subplots()

    

    ax.plot(target['date_added'],target['cumsum'])

    

    myFmt = DateFormatter("%m-%y")

    ax.xaxis.set_major_formatter(myFmt) 

    

    plt.ylabel('Number of shows')

    plt.xlabel('Time')

    plt.title('Shows Over time in '+country)

    

    

    

    

    plt.show()

#now we plot India's number of shows and movies over time 

country_show_over_time("India")
#import the necessary modules 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
#create the vectorizer and form the tfidf matrix 

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(netflix['description'])
#built the cosine similarity function to get the recommendations out  

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 

results = {}

for idx, row in netflix.iterrows():

   similar_indices = cosine_similarities[idx].argsort()[:-100:-1] 

   similar_items = [(cosine_similarities[idx][i], netflix['show_id'][i]) for i in similar_indices] 

   results[row['show_id']] = similar_items[1:]
#now we create a function which will pull out the necessary recommendations as computed from the above model 

#def item(title):  

 # return netflix.loc[netflix['title'] == title]['title'].tolist()[0].split(' - ')[0] 



#to recommend the top num movies or tv shows 

def recommend(title, num):

    print("Recommending " + str(num) + " products similar to " + title + "...")   

    print("-------")

    item_id=int(netflix.loc[netflix['title'] == title]['show_id'])

    recs = results[item_id][:num]   

    for rec in recs: 

       print("Recommended: " + str(netflix.loc[netflix['show_id']==rec[1]]['title']) + " (score:" +      str(rec[0]) + ")")

recommend('Transformers Prime',5)