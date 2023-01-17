# Let's ensure that we install some packages on our Kaggle VM

!pip install tweepy nltk spacy python-louvain python-igraph
# Installs spacy portuguese model

!python -m spacy download pt_core_news_sm
import numpy as np

import pandas as pd

import networkx as nx

import community as community_louvain

from igraph import *



from IPython.display import Image

import matplotlib.pyplot as plt



import os, tweepy, sys, unicodedata, re, csv, spacy, ast

from time import sleep, time, strptime, strftime

from datetime import date, datetime, timedelta

from unidecode import unidecode



from nltk.corpus import stopwords
# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pt_core_news_sm
# Loads SpaCy's model

nlp = pt_core_news_sm.load(disable=['tagger', 'ner', 'textcat'])

#nlp = English().from_disk("/model", disable=["ner"])
# Getting secrets from Kaggle

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()



CONSUMER_KEY = user_secrets.get_secret("CONSUMER_KEY")

CONSUMER_SECRET = user_secrets.get_secret("CONSUMER_SECRET")

ACCESS_TOKEN = user_secrets.get_secret("ACCESS_TOKEN")

ACCESS_TOKEN_SECRET = user_secrets.get_secret("ACESS_TOKEN_SECRET")
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# And we may call the REST API only to check if our credentials are ok!

api = tweepy.API(auth)

Image(api.me()._json['profile_image_url_https'])
class CustomStreamListener(tweepy.StreamListener):

    def __init__(self, time_limit=25):

        # For now, we won't want this class to be running forever

        

        self.start_time = time()

        

        self.limit = time_limit

        

        super(CustomStreamListener, self).__init__()



    def on_status(self, status): # First, we'll deal with the incoming statuses

        

        # A little note to limit the streaming time!

        if (time() - self.start_time) >= self.limit:

            return False



        # We'll check if retweet

        is_retweet =  hasattr(status, 'retweeted_status')



        # Then, we'll check if extended tweet

        if hasattr(status, 'extended_tweet'):

            

            text = status.extended_tweet['full_text']

            

        else:  # if neither of them

            

            text = status.text

           



        # BUT let's check if quote tweet

        is_quote_tweet = hasattr(status, 'quoted_status')

        quoted_text = ''



        if is_quote_tweet:

            # And if it is, let's check if quote tweet has been truncated

            

            if hasattr(status.quoted_status, 'extended_tweet'):

                

                quoted_text = status.quoted_status.extended_tweet['full_text']

                

            else:

                

                quoted_text = status.quoted_status.text

                

        # Then, let's define what we're going to get

        id_str = status.id_str

        created_at = status.created_at

        user_screen_name = status.user.screen_name

        user_location = status.user.location

        followers_count = status.user.followers_count

        retweet_count = status.retweet_count

        status_source = status.source

        mentions = status.entities['user_mentions']

        

        print(str(id_str), 

              str(created_at), 

              str(user_screen_name), 

              str(user_location), 

              str(followers_count), 

              str(retweet_count), 

              str(status_source), 

              str(text),

             '\n -> User mentions: {}'.format(str(mentions)))
SUBJECTS = ['Lady Gaga', 'Katy Perry', 'Dua Lipa', 'Beyonce', 

            'Taylor Swift', 'Ariana Grande', 'Miley Cyrus', 'Selena Gomez']
# Then, we create the streaming_api object

streaming_api = tweepy.streaming.Stream(auth, CustomStreamListener(), 

                                        timeout=25, tweet_mode='extended')



# Here, we also define the portuguese language (hence the pt comes from)

streaming_api.filter(track=SUBJECTS, languages=['pt'], is_async=False) 
class CustomStreamListener(tweepy.StreamListener):

    def __init__(self, time_limit=25):

        

        # For now, we won't want this class to be running forever

        self.start_time = time()

        

        self.limit = time_limit

        

        super(CustomStreamListener, self).__init__()



    def on_status(self, status): # First, we'll deal with the incoming statuses

        

        # A little note to limit the streaming time!

        if (time() - self.start_time) >= self.limit:

            return False # Return False kills the stream



        # We'll check if retweet

        is_retweet =  hasattr(status, 'retweeted_status')



        # Then, we'll check if extended tweet

        if hasattr(status, 'extended_tweet'):

            

            text = status.extended_tweet['full_text']

            

        else:  # if neither of them

            

            text = status.text

           



        # BUT let's check if quote tweet

        is_quote_tweet = hasattr(status, 'quoted_status')

        quoted_text = ''



        if is_quote_tweet:

            

            # And if it is, let's check if quote tweet has been truncated            

            if hasattr(status.quoted_status, 'extended_tweet'):

                

                quoted_text = status.quoted_status.extended_tweet['full_text']

                

            else:

                

                quoted_text = status.quoted_status.text

                

        # Then, let's define what we're going to get

        id_str = status.id_str

        created_at = status.created_at

        user_screen_name = status.user.screen_name

        user_location = status.user.location

        followers_count = status.user.followers_count

        retweet_count = status.retweet_count

        status_source = status.source

        mentions = status.entities['user_mentions']



        # It's useful to clean some undesired characters 

        # From https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python/49146722#49146722

        emoji_pattern = re.compile("["

         u"\U0001F600-\U0001F64F"  # emoticons

         u"\U0001F300-\U0001F5FF"  # symbols & pictographs

         u"\U0001F680-\U0001F6FF"  # transport & map symbols

         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

         u"\U00002702-\U000027B0"

         u"\U000024C2-\U0001F251"

         "]+", flags=re.UNICODE)



        text = emoji_pattern.sub(r'', text)

        quoted_text = emoji_pattern.sub(r'', quoted_text)

      

        try: # save to csv 

            

            # We can convert, optionally, the created_at timezone to 

            # our own (this is mine, at Brazil)

            conv_created_at = created_at - timedelta(hours=3)



            # this block saves effectively to csv, appending to the 

            # file with the respective date

            with open('{}.csv'

                      .format(conv_created_at

                              .date()

                              .strftime('%Y%m%d')),

                      mode='a', 

                      newline='', 

                      encoding='utf-8') as csvfile:



                csvwriter = csv.writer(csvfile, delimiter=',')



                csvwriter.writerow([id_str, 

                                    created_at, 

                                    conv_created_at, 

                                    text,

                                    quoted_text, 

                                    is_retweet, 

                                    user_screen_name, 

                                    user_location, 

                                    followers_count,

                                    retweet_count,

                                    status_source,

                                    mentions])

            

        except Exception as e:

                print("Error saving to csv: ", e)

                return

            

    #####                #####

    ##### ERROR HANDLING #####

    #####                #####

    

    def on_error(self, status_code):  # Here, we'll deal with general errors. 

        # We'll append the error to a log.txt file

        

        with open('log.txt', 'a') as f:

            

            f.write('Encountered error with status code: ' + 

                    str(status_code) + ',' + str(datetime.now()) + '\n')

        

        return True # Returning True doesn't kill the stream

    

    

    def on_limit(self, status_code): # Here, we'll deal with the most 

                                            # feared error: Rate Limits

        

        with open('log.txt', 'a') as f:

            

            f.write('Rate Limit Exceeded... ' + str(status_code) + ',' + str(datetime.now()) + '\n')

        

        return True  # Fun stuff: when you return True here, Tweepy takes care by

                        # exponentially increasing the time between each call

                        # when you Rate Limit, as per Twitter docs here: 

                        # https://developer.twitter.com/en/docs/basics/rate-limiting



            

    def on_disconnect(self, notice): 

        # Sometimes, twitter may disconnect 

        # you, for some reason.        

        # You may read more about it here 

        # https://developer.twitter.com/en/docs/tutorials/consuming-streaming-data#disconnections

        

        with open('log.txt', 'a') as f:  

            

            f.write('Disconnected: ' + str(notice) + ',' + str(datetime.now()) + '\n')

        

        sleep(2)

        

        return False # Obviously, the stream is killed



    

    def on_timeout(self): # And other times, you may only find a timeout.

        

        with open('log.txt', 'a') as f:

            

            f.write('Timeout... ' + ',' + str(datetime.now()) 

                    + '\n') # We log the incident...

        

        sleep(60) # We wait...

        

        return True # We try again (don't kill the stream)
# Ensure we're on this folder

os.chdir('/kaggle/working')
# Then, we create the streaming_api object

streaming_api = tweepy.streaming.Stream(auth, CustomStreamListener(), 

                                        timeout=25, tweet_mode='extended')



# Here, we also define the portuguese language (hence the pt comes from)

streaming_api.filter(track=SUBJECTS, languages=['pt'], is_async=False) 
cols = ['id_str',

        'created_at',

        'conv_created_at',

        'text',

        'quoted_text',

        'is_retweet',

        'user_screen_name',

        'user_location',

        'followers_count',

        'retweet_count',

        'status_source',

        'mentions']



# To automatically read our data from the current day, we'd use this. 

# But we're going to use a previously saved file today.

# df = pd.read_csv((datetime.today() - timedelta(hours=3)).strftime('%Y%m%d') + '.csv',

#           names = cols, encoding='utf-8')

# df.head()
# This is the previously saved file

df = pd.read_csv('/kaggle/input/sample_data.csv', 

                 names=['id_str', 'created_at','conv_created_at', 'text',

                        'quoted_text', 'is_retweet', 'user_screen_name', 

                        'user_location', 'followers_count', 'retweet_count', 

                        'status_source', 'mentions'])
# Taking a sample

df = df.iloc[:500,:]
print('Dataframe shape: ' + str(df.shape))
df.head()
# This is an example of Portuguese stopwords

sr = stopwords.words('portuguese')

print(sr)

print(len(sr))
# Remove RT, links, special characters, 

# hashtags, mentions (these are on entities object)



# Makes everything lowercase

df['text'] = df['text'].str.lower()

df['quoted_text'] = df['quoted_text'].str.lower()



# Removes the word RT

df['replaced_text'] = df['text'].str.replace(r'\s*rt\s', '',

                                             case=True, 

                                             regex=True)



# Replacing @s and #s terms - Or you can use this regex 

# as a way to find and store these values

# Or you can use the Twitter's Entities object: 

# https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/entities-object

df['replaced_text'] = df['replaced_text'].str.replace(r'\s([@#][\w_-]+)', '',

                                                      case=False,

                                                      regex=True)



# Replaces URLs in any position (start, middle or end)- Or you can use this regex as a way to find and store these values

df['replaced_text'] = df['replaced_text'].str.replace(r'http\S+\s|\swww.\S+\s|http\S+|www.\S+|\shttp\S+|\swww.\S+', '', 

                                                      case=False)



# Removes special characters from words. Ex. amanhã -> amanha

df['replaced_text'] = df['replaced_text'].apply(lambda text: unidecode(text))

#df['replaced_quoted_text'] = df['replaced_quoted_text'].apply(lambda text: unidecode(text))





# Removes any remaining special characters

df['replaced_text'] = df['replaced_text'].str.replace(r'[^0-9a-zA-Z ]+', '', 

                                                      case=False, 

                                                      regex=True)



# Same as before for the the quoted text

#df['replaced_quoted_text'] = df['replaced_quoted_text'].str.replace(r'(^|[ ])rt', '', case=False)

#df['replaced_quoted_text'] = df['quoted_text'].str.replace(r'\s([@#][\w_-]+)', '', case=False)

#df['replaced_quoted_text'] = df['replaced_quoted_text'].str.replace(r'http\S+\s|\swww.\S+\s|http\S+|www.\S+|\shttp\S+|\swww.\S+', '', case=False)

#df['replaced_quoted_text'] = df['replaced_quoted_text'].str.replace(r'[^0-9a-zA-Z ]+', '', case=False)
df[['text', 'replaced_text']].head()
# See https://www.kaggle.com/caractacus/thematic-text-analysis-using-spacy-networkx

tokens = []

parsed_doc = [] 

col_to_parse = 'replaced_text'



for doc in nlp.pipe(df[col_to_parse].astype('unicode').values, batch_size=50,

                        n_threads=3):

    if doc.is_parsed:

        parsed_doc.append(doc)

        tokens.append([n.text for n in doc if n.text not in sr])

    else:

        # We want to make sure that the lists of parsed results have the

        # same number of entries of the original Dataframe, 

        # so add some blanks in case the parse fails

        parsed_doc.append(None)

        tokens.append(None)





df['parsed_doc'] = parsed_doc

df['tokenized'] = tokens



# Ensure that we won't have any whitespace on tokens

df['tokenized'] = df['tokenized'].apply(lambda text: [word for word in text if word != ' '])
df.head()
# We could can all the words that are related with the

# code snippet below.

df['edges'] = df['tokenized'].apply(lambda col: tuple(((x,y) for x in col for y in col if x != y)))



df['edges'].head()
# Adapted from https://www.kaggle.com/caractacus/thematic-text-analysis-using-spacy-networkx

G = nx.Graph() # undirected

n = 0



for row in df.iterrows():

    

    for tup in row[1]['edges']:

        

        G.add_edge(tup[0], tup[1])

        

        n += 1        
print(G.number_of_nodes(), "nodes, and", G.number_of_edges(), "edges created.")
# https://stackoverflow.com/questions/40941264/how-to-draw-a-small-graph-with-community-structure-in-networkx

partition = community_louvain.best_partition(G)
# Assigning a position to the nodes

pos=nx.spring_layout(G)
# To check edges:

# G.edges()
# Check partition values

partition.values()
fig = plt.figure(figsize=(15,15))



# Draws a graph and defines color by nodes partitions

nx.draw_networkx_nodes(G, pos, node_size=20, cmap=plt.cm.RdYlBu, node_color=list(partition.values()))

nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='lightgray', width=0.1)



plt.show()
top_degree_sequence = sorted([(d, n)for n, d in G.degree()], reverse=True)

bottom_degree_sequence = sorted([(d, n)for n, d in G.degree()], reverse=False)
pd.DataFrame(top_degree_sequence, columns=['Occurences', 'Terms']).head()
pd.DataFrame(bottom_degree_sequence, columns=['Occurences', 'Terms']).head()
# Let's list mentioned users

df['mentions'] = df['mentions'].apply(lambda row: ast.literal_eval(row))

df['mentions'] = df['mentions'].apply(lambda row: [item['screen_name'] for item in row if item != None])

df['mentions'].head()
# Let's get every mention and explode it on rows

tst = df[['user_screen_name', 'mentions']].explode('mentions')



print(tst.shape)
tst = tst.dropna() # Drops if any NaN row



print(tst.shape)
# Assign the number 1 for every row -- to count and assign weights

tst['count'] = 1



# We group every mention, and use the above column to count how many mentions a user

# has mentioned another

tst = tst.groupby(by=['user_screen_name', 'mentions'], as_index=False).sum().sort_values(by=['count'], ascending=False)



# Then, we rename this column as weight

tst.rename(columns={"count": "weight"}, inplace=True)
# Here, we define a directed Graph with TupleList. Obviously, it expects tuples as input

# Then, we convert the dataframe with the itertuples method. You may check a bit more on 

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html

# Besides that, the TupleList method expects the edges and a weight column

    # which we defined earlier. So, when we define weights=True, it will use

    # this field

g = Graph.TupleList(tst.itertuples(index=False), directed=True, weights=True)
out = plot(g, bbox = (600, 600), vertex_size=3, edge_width=1, edge_arrow_size=0.3)

out.save('nodes_no_communities.png')

out
# We'll create an empty list called hubs

hubs = []



# This returns a list of degrees for each vertex

in_degrees = g.degree(mode=IN, loops=True)



# For each vertex, it will analyze if it has 8 or more in degrees

# If so, it will add its name as label, if not, it will add an

# empty string

for index, item in enumerate(in_degrees):

    if item >= 8:

        hubs.append(g.vs[index]["name"])

    else:

        hubs.append('')



# Finally, we set the labels. This is because the plot method

# Automatically search for a "label" attribute to plot

g.vs["label"] = hubs
# https://stackoverflow.com/questions/9471906/what-are-the-differences-between-community-detection-algorithms-in-igraph

# https://stackoverflow.com/questions/37855553/python-igraph-community-cluster-colors

i = g.community_infomap(edge_weights=None, vertex_weights=tst['weight'], trials=10)



pal = drawing.colors.ClusterColoringPalette(len(i))

g.vs['color'] = pal.get_many(i.membership)



out_comunnities = plot(g, bbox = (600, 600), vertex_size=5, edge_width=1, edge_arrow_size=0.2)

out_comunnities.save('nodes_communities.png')

out_comunnities