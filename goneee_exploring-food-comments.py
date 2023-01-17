# import the necessary libraries

import pandas as pd # data processing
import matplotlib.pyplot as plt # plotting 
import seaborn as sns # visualizations
# storing or settings to access the API in variables
api_base_url = 'https://www.vegguide.org' 
region_id = 2 # we can retrieve reviews from different regions
resource_path='/entry/{0}/reviews'.format(region_id) 
api_comments_url = api_base_url+resource_path #actual url that will be used
import requests # importing the requests library to perform http requests the api
# perform a http GET request to the URL specified earlier with custom HTTP headers
req = requests.get(api_comments_url, headers={
    'User-Agent':'SampleApi/0.01',
    'Accept':'application/json'})
#extract the data from the http response
data = req.json()
data # just viewing the data
# How many rows/records
print('We have {0} rows/records in the retrieved dataset'.format(len(data))) 
# What does one row/record look like?
data[0] 
# What keys/entries/columns are available in json row?
data[0].keys()
data[0]['body'] # a close look at the body key
data[0]['body']['text/vnd.vegguide.org-wikitext']
# This cell has an error, only kept for discussion purposes


#data_rows=[]
#for index in range(0,len(data)):
#    row = data[index]
#    data_rows.append({
#        'comment':row['body']['text/vnd.vegguide.org-wikitext'],
#        'date':row['last_modified_datetime'],
#        'user_veg_level_num':row['user']['veg_level'],
#        'user_veg_level_desc':row['user']['veg_level_description'],
#        'user_name':row['user']['name'],
#        'rating':row['rating']
#    })
#data_rows
# This cell has an error, only kept for discussion purposes


#data_rows=[]
#for index in range(0,len(data)):
#    row = data[index]
#    print(index)
#    print(row.keys())
#    data_rows.append({
#        'comment':row['body']['text/vnd.vegguide.org-wikitext'],
#        'date':row['last_modified_datetime'],
#        'user_veg_level_num':row['user']['veg_level'],
#        'user_veg_level_desc':row['user']['veg_level_description'],
#        'user_name':row['user']['name'],
#        'rating':row['rating']
#   })
#data_rows
#extracting the data as flat records to be added to a list `data_rows`
data_rows=[] # create a list to store all records
for index in range(0,len(data)): # iterate for each row in dataset
    row = data[index] #temporary variable to store row
    data_rows.append({ # extracting data from json document and creating dictionary and appending
        'comment':row['body']['text/vnd.vegguide.org-wikitext'] if 'body' in row else '',
        'date':row['last_modified_datetime'] if 'last_modified_datetime' in row else None,
        'user_veg_level_num':row['user']['veg_level'],
        'user_veg_level_desc':row['user']['veg_level_description'],
        'user_name':row['user']['name'],
        'rating':row['rating']
    })
data_rows #previewing results
data2 = pd.DataFrame(data_rows) # transform results as dataframe
data2.head() #preview dataframe
#utiliy method to draw a wordcloud from list/series
#modified
def wordcloud_draw(data, color = 'black'):
    """
       Draws a wordcloud
       
       params:
           data : list/series - set of sentences to include
        
       requires ```wordcloud``` package
    """
    from wordcloud import WordCloud, STOPWORDS
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
#visualize what everyone is saying
wordcloud_draw(data2['comment'])
# import library to assist with sentiment analysis
# Currently using https://github.com/cjhutto/vaderSentiment
# NB. There are implications and factors to consider, we will discuss
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# create a SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
# let us extract ONE (1) senetence
test_sentence = data2['comment'][0]
# analyze that sentence
sentiment_result = analyzer.polarity_scores(test_sentence)
# it returns a python dictionary of values
sentiment_result
# we could analyze each row in our data set using the apply method
data2['comment'].apply(analyzer.polarity_scores)
# let us define a function to analyze one sentence and return the compound value
def get_how_positive(sentence):
    return analyzer.polarity_scores(sentence)['compound']
# testing the application of the method
data2['comment'].apply(get_how_positive)
# creating a new column in our data set to store the sentiment value
data2['sentiment'] = data2['comment'].apply(get_how_positive)
# previewing updates
data2.head(10)
# Deterimining the correlation between the sentiment values and existing ratings
print("Correlation")
data2[['rating','sentiment']].corr()
# Visualizing the correlation on a heatmap
sns.heatmap(data2[['rating','sentiment']].corr())
