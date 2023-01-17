# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read in 'User' dataset containing information about users

users = pd.read_csv('../input/users.csv')

#Read in 'tweets' dataset containing actual tweets

tweets = pd.read_csv('../input/tweets.csv')
#First five rows of users dataset

users.head()
#create a simplified users dataset

users = users.iloc[:,[0,3,7,8,10,12]]

users.head()
#Let's look at the first 5 rows of the tweets dataset

tweets.head()
#create a simplified tweets df

tweets = tweets.iloc[:,[0,1,2,3,7,8,10]]

tweets.head()
#Plot the relationship between the number of a user's "friends" and "followers"

sns.scatterplot(x=users['friends_count'], y=users['followers_count'])

plt.xlabel('Number of Friends')

plt.ylabel('Number of Followers')

plt.show()
#subset and copy users df to include only users with 'friend_count' > 0

clean_users = users[users['friends_count'] > 0].copy()

#create new column containing the ratio for follower to friends

clean_users['f_f_ratio'] = clean_users['followers_count'] / clean_users['friends_count']

clean_users['f_f_ratio'].describe(percentiles = [0.5,0.6,0.7,0.8,0.9])
#subset users data based on the f_f_ratio, using a threshold of 2 followers to friends

popular = clean_users[clean_users['f_f_ratio'] > 2].copy()

unpopular = clean_users[clean_users['f_f_ratio'] < 2].copy()

per_popular = popular.shape[0] / clean_users.shape[0] *100

print(per_popular)
popular.sort_values(['f_f_ratio'],ascending=False).head(10)
#Plot 'friends' vs. 'followers' with points colored by language

sns.scatterplot(x='friends_count',y='followers_count',hue='lang',data=clean_users)

plt.xlim(-10000,40000)

plt.show()
print('Fraction of popular accounts that are Russian language users')

print(popular[popular['lang']=='ru'].shape[0]/popular.shape[0])
#subset popular df by english language users

en_pop = popular.loc[popular['lang'] == 'en','screen_name'].str.lower().values

print(en_pop)

    
#convert popular users' screen names to 'user_key' format

pop_peeps = popular['screen_name'].str.lower().values

#convert unpopular users' screen names to 'user_key' format

unpop_peeps = unpopular['screen_name'].str.lower().values

#no_friends 'user_keys'

no_friends = users.loc[users['friends_count'] == 0,'screen_name'].str.lower().values

#all 'user_keys'

all_users = users['screen_name'].str.lower().values
#count total number of tweets in tweets df for popular vs unpopular vs all users

name_list = ['popular','unpopular','all']

list_list = [pop_peeps,unpop_peeps,all_users]

count_list = []

for el in list_list:

    count = 0

    for peep in el:

        pop = tweets[tweets['user_key'] == peep]

        count += pop.shape[0]

    count_list.append(count)

for i in range(0,3):

    print('percent of tweets from {} people: {}'.format(name_list[i],count_list[i]/tweets.shape[0]*100))

        
#calculate the percent of tweet influence (among those users included in the popular and unpopular dataframes)

#accounted for by 'popular' vs 'unpopular' users





df_list = [popular,unpopular]

name_list = ['popular','unpopular']

df_count = []

#iterate through screen_name col

for df in df_list:

    count = 0

    for peep in df['screen_name'].values:

        #count number of followers for a given user in users df

        followers = int(users.loc[users['screen_name']==peep,'followers_count'])

        #convert screen_name to user_key

        uk = peep.lower()

        #count tweets from a given user in tweets df

        pop = tweets[tweets['user_key'] == uk]

        tweet_count = pop.shape[0]

        #scale tweets by number of followers

        mag = followers * tweet_count

        count += mag

    df_count.append(count)    

#print(df_count)

for i in range(0,2):

    print('Percent of tweet influence attributable to {} people: {}'.format(name_list[i],(df_count[i]/sum(df_count)*100)))



#convert created_at column to datetime objects and plot a histogram of creation years

users['created_at'] = pd.to_datetime(users['created_at'])

years = users['created_at'].dt.year

years.plot(kind='hist')

plt.xlabel('Account Created (Year)')

plt.show()
#Convert created_at to datetime and plot histograms of creation years for popular and unpopular users

popular['created_at'] = pd.to_datetime(popular['created_at'])

pop_years = popular['created_at'].dt.year

unpopular['created_at'] = pd.to_datetime(unpopular['created_at'])

unpop_years = unpopular['created_at'].dt.year

plt.hist([pop_years,unpop_years], alpha=0.5,label = ['Popular','Unpopular'])

plt.xlabel('Account Created (Year)')

plt.ylabel('Frequency')

plt.legend(loc='upper left')

plt.show()
#Plot histogram of creation date by month in the year 2014

pop_year = popular[popular['created_at'].dt.year == 2014]

plt.hist(pop_year['created_at'].dt.month,bins=12)

plt.xlabel('Month (1-12)')

plt.ylabel('Frequency')

plt.show()
#Plot histograms of creation dates by month over the years 2013-2016 for unpopular accounts

months_list = []

yr_list = [2013,2014,2015,2016]

for el in yr_list:

    df = unpopular[unpopular['created_at'].dt.year == el]

    months = df['created_at'].dt.month

    months_list.append(months)



plt.hist(months_list,alpha = 0.5, label = ['2013','2014','2015','2016'])

plt.xlabel('Month (1-12)')

plt.ylabel('Frequency')

plt.legend(loc='upper left')         

plt.show()
#create a dictionary to replace 'user_key' values in tweets df with 'screen_name' values

users['lower'] = users['screen_name'].str.lower()

s_name = users.screen_name.values

lower = users.lower.values

conversion = dict(zip(lower,s_name))
#replace 'user_keys' with 'screen_names' in tweets dataframe

tweets['user_key'].replace(conversion,inplace=True)

tweets.head()
#create a new True/False column indicating whether the tweet is a retweet

tweets['is_retweet'] = tweets['text'].str.contains('RT @')

tweets['is_retweet'].mean()
#create a new row in tweets containing the screen_name of the user who is being retweeted if applicable

tweets['RT_source'] = tweets['text'].str.extract(r'@(\S+):')

tweets['RT_source'].fillna('None',inplace=True)

                      

tweets['RT_source'].head()
#create a list of the unique screen_names

user_list = tweets['user_key'].unique()



#Define a function to test whether a value is part of the user_list (for apply function below)

def test_in(el):

    return (el in user_list)



#create a new column in tweets indicating whether the source of a retweeted tweet was in our list of Russian screen_names

tweets['RT_from_user_list'] = tweets['RT_source'].apply(test_in)

#find fraction of all tweets that were retweeting tweets from a Russian troll

tweets['RT_from_user_list'].mean()
#Create a new column in tweets indicating T/F of whether a RT was originally composed by the same user

tweets['RT_self'] = tweets['user_key'] == tweets['RT_source']

tweets['RT_self'].mean()
tweets.loc[tweets['RT_from_user_list'] == True, 'RT_source'].value_counts().head(10)

top_retweeted = tweets.loc[tweets['RT_from_user_list'] == True, 'RT_source'].value_counts().head(10).index

print(top_retweeted)
top_retweeted = tweets.loc[tweets['RT_from_user_list'] == True, 'RT_source'].value_counts().head(20).index

frac_RT = []

for user in top_retweeted:

    is_rt = round(tweets.loc[tweets['user_key'] == user,'is_retweet'].mean(),2)

    frac_RT.append(is_rt)

    per_rt = is_rt *100

    print('{}% of {}\'s tweets were retweets'.format(per_rt,user))

print(frac_RT)    
tweets.loc[tweets['RT_from_user_list'] == False,'RT_source'].value_counts().head(10)
tweets['user_key'].value_counts().head(10)
top_tweeter = tweets['user_key'].value_counts().head(20).index

top_frac_RT = []

for user in top_tweeter:

    is_rt = round(tweets.loc[tweets['user_key'] == user,'is_retweet'].mean(),2)

    top_frac_RT.append(is_rt)

    per_rt = is_rt *100

    print('{}% of {}\'s tweets were retweets'.format(per_rt,user))

print(top_frac_RT) 
#Create a simplified df containing only 'user_key','created_str','RT_source' and dropna rows

simple = tweets.iloc[:,[1,3,8]].dropna().copy()

#convert 'created_str' to datetime object

simple['created_str'] = pd.to_datetime(simple['created_str'])

#collect hour attribute in a new column

simple['hour'] = simple['created_str'].dt.hour.astype(float)

#print(simple.head())





#Function that accepts a list of users and the 'user_key' or 'source' column names as input

#Generates a count of a users tweets for each hour of the day

#Returns a dictionary mapping a 'user_key' or a 'source' to a list containing the hourly tweet count

def hour_hist(user_list,column,RT_col = False):

    time_dict = {}

    for user in user_list:

        #print(user)

        if user in simple[column].tolist():

            #print(user)

            time_list = simple.loc[simple[column]==user,'hour'].tolist()

            #print(time_list)

            count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            for el in time_list:

                int_el = int(el)

                count_list[int_el] += 1

            #print(count_list)    

            sum_count = sum(count_list)

            #print(sum_count)

            new_list = [el/sum_count for el in count_list]

            if RT_col == True:

                rt_frac = round(tweets.loc[tweets['user_key'] == user,'is_retweet'].mean(),2)

                new_list.append(rt_frac)

                time_dict[user] = new_list

                data_frame = pd.DataFrame.from_dict(time_dict,orient='index')

                data_frame.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,'RT']

            else:

                time_dict[user] = new_list

                data_frame = pd.DataFrame.from_dict(time_dict,orient='index')

                #print(data_frame.head())

                data_frame.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

    return data_frame 

#Tweet pattern all tweets

all_users = tweets['user_key'].unique().tolist()

#print(all_users)

#all_user_dict = hour_hist(all_users, 'user_key')

all_heat = hour_hist(all_users,'user_key',RT_col=True)

   

sns.clustermap(all_heat,metric='minkowski',col_cluster=False,robust=True)


#Tweet pattern for users with >90% original tweets

originals = all_heat.loc[all_heat['RT'] < 0.1].index

#Create heatmap dataframe

original_heat = hour_hist(originals,'user_key')

#display clustermap   

sns.clustermap(original_heat,metric='minkowski',col_cluster=False,robust=True)
#create a list of the top 20 tweeters in the tweets dataset

top_twenty = simple['user_key'].value_counts().head(20).index

#create heatmap dataframe.

twenty_time_heat = hour_hist(top_twenty, 'user_key',RT_col=True)

#Plot a clustered heatmap to show tweeting patterns over a 24 hour period for the top 20 tweeters

sns.clustermap(twenty_time_heat,metric='minkowski',col_cluster=False,robust=True)

#Tweet pattern of top 20 most retweeted users

retweet_top_twenty = tweets.loc[tweets['RT_from_user_list'] == True, 'RT_source'].value_counts().head(20).index

#Create heatmap dataframe

RT_heat = hour_hist(retweet_top_twenty,'user_key',RT_col=True)

#Plot clustered heatmap

sns.clustermap(RT_heat,metric='minkowski',col_cluster=False,robust=True)
#Pattern of tweets that are retweeting the most commonly retweeted users OUTSIDE of Russian user set

out_rt_top_twenty = tweets.loc[tweets['RT_from_user_list'] == False, 'RT_source'].value_counts().head(20).index



#Create heatmap dataframe, RT column is omitted because it obscures tweet pattern data

out_rt_heat = hour_hist(out_rt_top_twenty,'RT_source')

#Show clustermap

sns.clustermap(out_rt_heat,metric='minkowski',col_cluster=False,robust=True)
#What is the tweet time distribution for users pretending to be news outlets?



#List of words commonly associated with news outlets

news_word_list = ['today','post','online','new','voice']

joined_news = '|'.join(news_word_list)

#lower case all users

lower_users = tweets['user_key'].str.lower()

#create array of unique 'news' users

news_users = tweets.loc[lower_users.str.contains(joined_news),'user_key'].unique()

#Create heatmap dataframe 

news_heat = hour_hist(news_users,'user_key',RT_col=True) 

#display clustermap                       

sns.clustermap(news_heat,metric='minkowski',col_cluster=False,robust=True)

#Copy tweets dataframe, removing rows that have na values for text

new_tweets = tweets.loc[tweets['text'].notna()].copy() 

#Remove the RT @xyz: part of a tweet by splitting text at : that is NOT preceded by https

split = new_tweets['text'].str.split(r'(?<!https):',n=1,expand=True)

#Take the column containing the RT-free text, fill Na values with the left column for those tweets that were not retweets

split[1].fillna(split[0],inplace=True)

#Add this new 'clean_text' column to the new_tweets data frame

new_tweets['clean_text'] = split[1]

#Clean up the 'clean_text' column further, by removing hashtags, mentions, #, @, and other various symbols

new_tweets['clean_text'] = (new_tweets['clean_text']

                                .str.replace('7yrs','')

                                #remove hashtags

                                .str.replace(r'(?<=#)[A-Za-z0-9]+','')

                                #remove mentions

                                .str.replace(r'(?<=@)[A-Za-z0-9]+','')

                                #.str.lstrip('#')

                                .str.strip('[# ]+')

                                .str.strip('[@ ]+')

                                .str.strip(')')

                                .str.strip('(')

                                .str.strip('lol')

                                .str.strip('"')

                                .str.strip(u'\u2026')

                                .str.strip('#')

                                .str.lstrip(':')

                                .str.strip(' ')

                            

                           )

new_tweets.head()
#Create a list of strings containing the 200 most retweeted 'core' tweets

string_list = new_tweets['clean_text'].value_counts(dropna=True).head(200).index



def time_diff(df):

    #This function takes a dataframe as input, which should be structured like the new_tweets df

    #creates a subsetted df with a selection of the columns, converts tweet time to datetime, subtracts

    #datetime object from the original tweet time, and returns a dataframe with a new column 'time_diff'

    #with the time difference from the original tweet presented in hours. This is a helper function for time_df_list (below)

    

    s_df = df.iloc[:,[1,3,4,7,8,9,10,11]].copy()

    s_df['created_str'] = pd.to_datetime(s_df['created_str'])

    s_df.sort_values(by='created_str',ascending=True,inplace=True)

    #initialize a list of time differences; the first element is the first tweet minus itself, so 0

    time_diff_list = [0]

    #subtract the time of each tweet from the time of the first tweet

    for i in range (1, len(s_df['created_str'])):

        time_diff = s_df.iloc[i,1] - s_df.iloc[0,1]

        #represent time differences in hours 

        time_diff_list.append(time_diff.total_seconds()/3600)

    #add a column to the subsetted data frame with the time difference values

    s_df['time_diff'] = time_diff_list

    #sort time_diff column

    s_df.sort_values(by='time_diff',ascending=True, inplace=True)

    return s_df

    

def time_df_list(tweet_list,chain_plots=False):

    #initialize a list to contain dataframes for each tweet

    df_list = []

    #iterate over 'core' tweet list

    for el in tweet_list:

        #subset dataframe to include only tweets with a given 'clean_text'

        df = new_tweets[new_tweets['clean_text'] == el]

        if chain_plots == False:

            #size variable counts the number of times this text has been retweeted

            size = df.shape[0]

            #If the text is not an empty string, and it has been retweeted more than 10 times, create a small

            #dataframe with select columns, convert the 'created_str' column to datetimes, and sort from first to last

            if (el != '') & (size > 10):

                s_df = time_diff(df)

                #append sorted data frame to list of dataframes

                df_list.append(s_df)

        else:

            #this definition of size selects for those retweet patterns where there is a chain of single retweets

            size = df.shape[0] / len(df['user_key'].unique())

            if (el != '') & (size < 2):

                s_df = time_diff(df)

                #append sorted data frame to list of dataframes

                df_list.append(s_df)

    return df_list



text_df_list = time_df_list(string_list[:20])



#plot catplots one at a time (this was the cleanest way I found to do this, otherwise there were issues with facets in the figure)

for i in range(0,len(text_df_list)-1):

    p = sns.catplot(data=text_df_list[i],x='time_diff',y='user_key',col='clean_text',aspect=2)

    p.set_titles('{col_name}')

    p.set_xlabels('Time From First Tweet (Hours)')

    

#create dataframe list for chain retweets

chain_df_list = time_df_list(string_list,chain_plots=True)

#plot chains

for i in range(0,10):

    p = sns.catplot(data=chain_df_list[i],x='time_diff',y='user_key',col='clean_text',hue='RT_source',aspect=2)

    p.set_titles('{col_name}')

    p.set_xlabels('Time From First Tweet (Hours)')

#Create a list of the top 40 users retweeting other Russian users

in_rt_top_forty = tweets.loc[tweets['RT_from_user_list'] == True, 'user_key'].value_counts().head(40).index



#This effectively creates a heatmap of a user's typical position in the retweet chain

user_position_dict = {}

for user in in_rt_top_forty:

    #print(user)

    position_dict = {}

    #iterate through chain tweet dataframes

    for i in range(0,len(chain_df_list)):

        now_df = chain_df_list[i].copy()

        #reset index to be able to access index as integers

        now_df.reset_index(inplace=True)

        #test if user is in the 'user_key' column of people who retweeted a given tweet

        if user in now_df['user_key'].tolist():

            #capture the position of that user in the sequence of retweets as a list (sometimes a person tweets twice)

            positions = now_df.index[now_df['user_key']==user].tolist()

            #Pick the first occasion that the person tweeted (if multiple tweets)

            first_position = positions[0]

            #Count the number of times that person tweets at a given position in the sequence of retweets

            if first_position in position_dict:

                position_dict[first_position] += 1

            else:

                position_dict[first_position] = 1

        

        else:

            pass

    #map the position_dict to the user_key

    user_position_dict[user] = position_dict

#create a dataframe from the dictionary        

sequence = pd.DataFrame.from_dict(user_position_dict,orient='index')   

#sequence
#get columns names

cols = sequence.columns

#sort columns numerically and take only the first 12

cols = sorted(cols)[:12]

#sort columns and create a new dataframe n_sequence

n_sequence = sequence[cols].copy()

#replace na values with 0

n_sequence.fillna(0,inplace=True)

#Display clustermap

sns.clustermap(n_sequence,metric='minkowski',col_cluster=False)
edge_list = []

#iterate over the first 10 users in the top_retweeted list

for user in top_retweeted[:10]:

    #initialize a dataframe to consolidate 'edges' - connections between a source and a retweeter

    edge_df = pd.DataFrame()

    #collect all retweets from a given source, and present the 'RT_source' column

    edge_df['source'] = tweets.loc[(tweets['is_retweet']==True) & (tweets['RT_source']==user),'RT_source']

    #collect all retweets from a given source, and present the 'user_key' column

    edge_df['user'] = tweets.loc[(tweets['is_retweet']==True) & (tweets['RT_source']==user),'user_key']

    #create an array of the top twenty users who retweeted the source 

    top_retweeters = edge_df['user'].value_counts().head(20).index

    #Create a list of (source,retweeter) tuples

    top_list = list(zip([user]*len(top_retweeters),top_retweeters))

    for el in top_list:

        #add each edge for each top_retweeted source to the edge_list

        edge_list.append(el)

import networkx as nx

#Create a directional graph object

B = nx.DiGraph()

#Populate the graph with connections in edge_list

B.add_edges_from(edge_list)

#calculate in_degree_centrality for each person in the graph

#this is the number of edges that end at a given user divided by the total number of users

#in this case it reflects how many people a given person retweets

in_deg_dict = nx.in_degree_centrality(B)

#create the 'central' dataframe from the in_deg_dict dictionary

central = pd.DataFrame.from_dict(in_deg_dict,orient='index')

#sort dataframe by in-degree values

central = central.sort_values(by=0,ascending=False)

#assign top 20 retweeters to the top_retweeter variable

top_retweeter = central.head(20).index



print(top_retweeter)
#This code was modified from code found here: https://bokeh.pydata.org/en/latest/docs/user_guide/graph.html

from bokeh.io import show, output_notebook

from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, BoxZoomTool, ResetTool

from bokeh.models.graphs import from_networkx





#top 20 most retweeted users

top_retweeted = tweets.loc[tweets['RT_from_user_list'] == True, 'RT_source'].value_counts().head(20).index

#list of 10 colors for the top 10 most retweeted users

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

#Create a dict mapping a user to a color

top_rt_map = dict(zip(top_retweeted,colors))



#SET EDGE COLORS

#initialize the edge_attr dictionary

edge_attr = {}

#iterate over edges list for the graph

for start_node,end_node, _ in B.edges(data=True):

    #define color based on the start node (retweeted user) in the user:color dictionary above

    color = top_rt_map[start_node]

    #map the edge to the color selected above

    edge_attr[(start_node, end_node)] = color

#create a new attribute of the edges in B called edge color, using the edge_attr dictionary  

nx.set_edge_attributes(B, edge_attr, "edge_color")



#SET NODE COLOR AND SIZE

#initialize dicts to map a user's node to a color and size

color_map = {}

size_map = {}

#iterate through nodes in the B graph (this is a tuple ('user_key',{}), so we need to access the first element)

for node in B.nodes(data=True):

    #Test if user is in the top 10 retweeted set

    if node[0] in top_rt_map:

        #if yes, assign the color using the dictionary above, and the size = 15

        color_map[node[0]] = top_rt_map[node[0]]

        size_map[node[0]] = 15

    #Test if the user is one of the top 20 retweeters in the top_retweeter list (above)

    elif node[0] in top_retweeter:

        #if so color the user's node black and set size at 8

        color_map[node[0]] = 'black'

        size_map[node[0]] = 8

    #if not either of these things, set users color grey and size 4

    else:

        color_map[node[0]] = 'grey'

        size_map[node[0]] = 4



#create and set node attibutes 'color' and 'size' using the dictionaries created above       

nx.set_node_attributes(B, color_map, 'color')

nx.set_node_attributes(B, size_map,'size')



#Show with Bokeh

#Initialize a plot object with the given size and title

plot = Plot(plot_width=500, plot_height=500,

            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

plot.title.text = "Linkage Between Top 10 retweeted users and their Top 20 retweeters"

#Initialize this cool node_hover_tool, which allows users to inspect node identity

node_hover_tool = HoverTool(tooltips=[("index", "@index")])

#add the hover_tool, zoomtool, and resettool to the plot

plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())



#create a graph renderer using the data in B, applying a circular layout

graph_renderer = from_networkx(B, nx.circular_layout, scale=1, center=(0, 0))

#render the nodes using circles, with the size and color defined according to the node attributes set above

graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color',fill_alpha=1,line_color='color')

#render the edges with the edge_color defined in the edge attributes

graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', line_alpha=0.8, line_width=1)

#add the rendered data to the plot

plot.renderers.append(graph_renderer)

#display plot in notebook setting

output_notebook()

show(plot)
#Create array of top 50 retweeters

top_fifty_rt = tweets.loc[tweets['is_retweet'] == True, 'user_key'].value_counts().head(50).index



#Create an edge list mapping each of the top 50 retweeters to their top source, and collect their number of

#retweetes in the size_df_dict

in_edge_list = []

source_list = []

size_df_dict = {}

for user in top_fifty_rt:

    #create data frame mapping sources to retweeters 

    edge_df = pd.DataFrame()

    edge_df['source'] = tweets.loc[(tweets['is_retweet']==True) & (tweets['user_key']==user),'RT_source']

    edge_df['user'] = tweets.loc[(tweets['is_retweet']==True) & (tweets['user_key']==user),'user_key']

    #count number of retweets for a given retweeter

    rt_count = edge_df.shape[0]

    #map user_key to number of tweets

    size_df_dict[user]=rt_count

    #find the most common source for a given retweeter

    top_source = edge_df['source'].value_counts().head(1).index[0]

    source_list.append(top_source)

    in_edge_list.append((top_source,user))  
#The purpose of this block of code is to make it possible to visualize tweet number in the graph below

#This is a way of binning the number of tweets into discrete bins in the range (fewest_retweets to most_retweets)

most_tweets = max(size_df_dict.values())

fewest_tweets = min(size_df_dict.values())

#Create a list of numbers linearly spaced from fewest to most tweeets, with a length = to the number of values

tweet_range = np.linspace(fewest_tweets,most_tweets,len(size_df_dict.values()))

#Iterate over the users in the size_df_dict

for user in size_df_dict:

    #iterate over each bin of the tweet_range list

    for i in range(len(tweet_range)-1):

        tweet_bin_lo = tweet_range[i]

        tweet_bin_hi = tweet_range[i+1]

        #if the number of tweets falls within a given bin, remap the user to the bin number in the size_df_dict

        if (size_df_dict[user] >= tweet_bin_lo) & (size_df_dict[user]<=tweet_bin_hi):

            size_df_dict[user] = i      
#Create a directional graph object

G = nx.DiGraph()

#Populate graph using in_edge_list data

G.add_edges_from(in_edge_list)

#set size of nodes based on the size_df_dict created above

nx.set_node_attributes(G,size_df_dict,'df_size')
#create a dictionary mapping the unique top sources to colors

import bokeh.palettes as bp

unique_source = set(source_list)

unique_list = list(unique_source)

#create a list of colors from the viridis palette with a length equal to the number of unique sources

colors = bp.viridis(len(unique_source))

source_color = [(unique_list[i],colors[i]) for i in range(len(unique_source))]

#map each source to a color

source_map = dict(source_color)   
#COLOR EDGES BASED ON SOURCE COLOR

edge_attr = {}

for start_node,end_node, _ in G.edges(data=True):

    color = source_map[start_node]

    edge_attr[(start_node, end_node)] = color

nx.set_edge_attributes(G, edge_attr, "edge_color")



#COLOR NODES IF IN SOURCE:COLOR Dictionary, else grey

color_map = {}

for node in G.nodes(data=True):

    #print(node[1])

    if node[0] in source_map:

        color_map[node[0]] = source_map[node[0]]

    else:

        color_map[node[0]] = 'grey'



#SCALE NODES in proportion to a given user's number of retweets (if in the top retweeter list), else size=4

size_map = {}

for node in G.nodes(data=True):

    if node[0] in size_df_dict:

        size_map[node[0]] = size_df_dict[node[0]]

    else:

        size_map[node[0]] = 4

        

#set node color and size attributes using the dictionaries created above        

nx.set_node_attributes(G, color_map, 'color')

nx.set_node_attributes(G, size_map,'size')



#Create a large bokeh plot object and title

plot = Plot(plot_width=800, plot_height=800,

            x_range=Range1d(-2, 2), y_range=Range1d(-2, 2))

plot.title.text = "Linkage Between Top 50 retweeters and their Respective Top Sources"

#initialize and add plot tools

node_hover_tool = HoverTool(tooltips=[("index", "@index")])

plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

#initialize renderer using data from G, render nodes and edges using the attributes set above

graph_renderer = from_networkx(G, nx.circular_layout, scale=1.8, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(size='size', fill_color='color',fill_alpha=0.7,line_color='color')

graph_renderer.edge_renderer.glyph = MultiLine(line_color='edge_color', line_alpha=0.8, line_width=2)

plot.renderers.append(graph_renderer)



output_notebook()

show(plot)
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize



#create a set of words to ignore in the text

stop = set(stopwords.words('english'))

#add some additional tweet specific characters and words

stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '@', '#', 'rt', 'amp','https','co','http','u'])



#create separate word compilations for RT_from_user_list == True and RT_from_user_list == False

for el in [True,False]:

    series_tweets = new_tweets.loc[new_tweets['RT_from_user_list'] == el,'clean_text']

    #concatenate clean text column into a single string separated by spaces

    tweet_str = series_tweets.str.cat(sep = ' ')

    #create a lower case word list if the word is not in the stop lists and is alphanumeric

    word_list = [i.lower() for i in wordpunct_tokenize(tweet_str) if i.lower() not in stop and i.isalpha()]

    #create a frequency distribution of bigrams in the current word list

    wordfreqdist = nltk.FreqDist(list(nltk.bigrams(word_list)))

    #find 30 most common bigrams

    mostcommon = wordfreqdist.most_common(30)

    #if RT_from_user_list == True

    if el:

        print('Top 30 Bigrams In Retweeted Tweets Originating from Russian User List:')

        print(mostcommon)

        print('\n')

    else:

        print('Top 30 Bigrams In Retweeted Tweets Originating from Outside of Russian User List:')

        print(mostcommon)
#add a column of the tweet's year to the new_tweets dataframe

dates = pd.to_datetime(new_tweets['created_str'])

new_tweets['years'] = dates.dt.year



year_list = [2014,2015,2016,2017]

for year in year_list:

    #get clean_text for tweets for each year

    year_tweets = new_tweets.loc[new_tweets['years'] == year,'clean_text']

    #print(year_tweets.head())

    tweet_str = year_tweets.str.cat(sep = ' ')

    #create a lower case word list if the word is not in the stop lists and is alphanumeric

    word_list = [i.lower() for i in wordpunct_tokenize(tweet_str) if i.lower() not in stop and i.isalpha()]

    #create a frequency distribution of bigrams in the current word list

    wordfreqdist = nltk.FreqDist(list(nltk.bigrams(word_list)))

    #find 30 most common bigrams

    mostcommon = wordfreqdist.most_common(30)

    print('The most common words for {} are:'.format(str(year)))

    print(mostcommon)

#Collect tweets containing 'Donald Trump' from 2015, and print the first 10

DJT_2015 = new_tweets.loc[(new_tweets['text'].str.contains('Donald Trump')) & (new_tweets['years']==2015), 'text']

for el in DJT_2015.head(10).values:

    print(el)
#Collect tweets containing 'Jeb Bush' from 2015 and print first 10

JB_2015 = new_tweets.loc[(new_tweets['text'].str.contains('Jeb Bush')) & (new_tweets['years']==2015), 'text']

for el in JB_2015.head(10).values:

    print(el)
#collect and print tweets containing 'Rand Paul' from 2015

RP_2015 = new_tweets.loc[(new_tweets['text'].str.contains('Rand Paul')) & (new_tweets['years']==2015), 'text']

for el in RP_2015.head(10).values:

    print(el)