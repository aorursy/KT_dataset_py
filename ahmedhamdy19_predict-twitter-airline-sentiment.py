import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import re

from wordcloud import WordCloud,STOPWORDS

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.feature_extraction.text import CountVectorizer

from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder

import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')



from keras.preprocessing.text import Tokenizer

# Packages for modeling

from keras import models

from keras import layers

from keras import regularizers



Tweet= pd.read_csv("../input/Tweets.csv")

Tweet.head()
Tweet.info()
Tweet = Tweet.loc[: , ['airline_sentiment', 

                         'airline_sentiment_confidence',

                         'negativereason',

                         'negativereason_confidence',              

                         'name',

                         'text',

                         'tweet_coord',

                         'tweet_created',

                         'airline']]

Tweet = Tweet.set_index('tweet_created')

Tweet.head()
Tweet.describe()
Tweet.groupby('airline')['airline_sentiment'].count()
Tweet.groupby(['airline','airline_sentiment']).count().iloc[:,0]
ax = Tweet.groupby(['airline','airline_sentiment']).count().iloc[:,0].unstack(0).plot(kind = 'bar', title = 'Airline Ratings via Twitter')

ax.set_xlabel('Ratings')

ax.set_ylabel('Ratings Count')
def percentages(df, rating = 'negative'):

    if rating == 'negative':

        i = 0

        column = 'Percent Negative Ratings'

    elif rating == 'neutral':

        i = 1

        column = 'Percent Neutral Ratings'

    elif rating == 'positive':

        i = 2

        column = 'Percent Positive Ratings'

        

    #Count of all tweet ratings for each airline (negative, neutral, positive)

    each_airline_ratings_counts = df.groupby(['airline','airline_sentiment']).count().iloc[:,0]

    #Rating tweet total index for each airline:

    #American i

    #Delta i + 3

    #southwest i + 6

    #US Airways i + 9

    #United i + 12

    #Virgin i + 15



    #Count of total tweets about an airline

    total_airline_ratings_counts = df.groupby(['airline'])['airline_sentiment'].count()

    #Airline index in total tweets:

    #American 0

    #Delta 1

    #Southwest 2

    #US Airways 3

    #United 4

    #Virgin 5





    #Create a dictionary of percentage of rating tweets = (each_airline_ratings_counts / total_airline_ratings_counts)

    tweet_ratings_dict = {'American':each_airline_ratings_counts[i] / total_airline_ratings_counts[0],

                'Delta':each_airline_ratings_counts[i + 3] / total_airline_ratings_counts[1],

                'Southwest': each_airline_ratings_counts[i + 6] / total_airline_ratings_counts[2],

                'US Airways': each_airline_ratings_counts[i + 9] / total_airline_ratings_counts[3],

                'United': each_airline_ratings_counts[i + 12] / total_airline_ratings_counts[4],

                'Virgin': each_airline_ratings_counts[i + 15] / total_airline_ratings_counts[5]}



    #make a dataframe from the dictionary

    percent_tweet_ratings = pd.DataFrame.from_dict(tweet_ratings_dict, orient = 'index')

    

    #have to manually set column name when using .from_dict() method

    percent_tweet_ratings.columns = [column]

        

    return percent_tweet_ratings
#Create a df called negative that contains the percent negatives by calling the function above

percent_negative_ratings = percentages(Tweet, 'negative')



#Create a df called neutral that contains the percent neutrals by calling the function above

percent_neutral_ratings = percentages(Tweet, 'neutral')



#Create a df called positive that contains the percent positives by calling the function above

percent_positive_ratings= percentages(Tweet, 'positive')



def merging_airlines_ratings_dataframes(x,y,z):



    concatenate_airlines_ratings_dataframes = pd.concat([x,y,z], axis = 1)

    return concatenate_airlines_ratings_dataframes



#concatenate all 3 dataframes of percent ratings

percent_ratings_dataframes_concatenated = merging_airlines_ratings_dataframes(percent_neutral_ratings, percent_negative_ratings, percent_positive_ratings)

print(percent_ratings_dataframes_concatenated)
#graph all of airlines ratings dataframes

ax = percent_ratings_dataframes_concatenated.plot(kind = 'bar', stacked = True, rot = 0, figsize = (15,6))

#set x label

ax.set_xlabel('Airlines')

#set y label

ax.set_ylabel('Percentages')

#move the legend to the bottom of the graph

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),

          fancybox=True, shadow=True, ncol=5)



plt.show()
airline_column = list(Tweet.reset_index().iloc[6750:6755,8])

tweet_text_column = list(Tweet.reset_index().iloc[6750:6755,6])



for pos, item in enumerate(airline_column):

    print('Airline as entered: ' + str(item))

    print('The tweet text: ')

    print(tweet_text_column[pos], '\n''\n')
Tweet = Tweet.iloc[:,0:6]

Tweet.head()
import re





# We create new column called 'Airline'Then we extract the right airline from the tweet text by applying regular expression function to the 'text' column

Tweet['Airline'] = Tweet.text.apply(lambda x: re.findall('\@\w+', x)[0])



#get all unique twitter tags and the count for how many times it appears in the column

twitter_text_tags = np.unique(Tweet.Airline, return_counts = True)



#compile twitter_text_tags so that it lists the unique tag and its total count side by side instead of 2 seperate arrays

twitter_tags_count = list(zip(twitter_text_tags[0],twitter_text_tags[1]))

twitter_tags_count
airline_companies_list = ['@virginamerica','@united','@southwestair','@americanair','@jetblue','@usairways']

    

# We compile a regex search to seperate out only the airline tag and ignoring other users tags in the text

# We are ignoring case, or capitaliztion  in order to negate all the uniquess we encountered in the list above

airlines = re.compile('|'.join(airline_companies_list), re.IGNORECASE)

    

#We apply the compiled regex search and remove the twitter tag '@'

#for example, the following code takes @AmericanAir and returns AmericanAir

Tweet['Airline'] = Tweet.Airline.apply(lambda x: np.squeeze(re.findall(airlines, x))).str.split('@').str[1]

print(list(Tweet.Airline.head(10)))
Tweet_df_without_airline_first = Tweet.reset_index()

Tweet_df_rows_without_airline_first = Tweet_df_without_airline_first[Tweet_df_without_airline_first.Airline.isnull()].text.apply(lambda x: re.findall('\@\w+', x))

Tweet_df_rows_without_airline_first
#reset the index of our dataframe

Tweet = Tweet.reset_index()



#compile a list of index locations of the tweets that return null and set their airline value to the appropriate

#airline referenced in the tweet

united = [737,868,1088,4013]

southwest = [4604,5614,5615,6136,6362]

jetblue = [6796,6811,6906]

usairways = [7330, 8215,10243,10517,10799,10864,10874,10876,11430]

american = [11159,12222,12417,12585,13491,13979]

delta = [12038, 12039]

Tweet.set_value(united,'Airline','united')

Tweet.set_value(southwest,'Airline','southwestair')

Tweet.set_value(jetblue,'Airline','jetblue')

Tweet.set_value(usairways,'Airline','usairways')

Tweet.set_value(american,'Airline','americanair')

Tweet.set_value(delta,'Airline','delta')

    

#Since all airlines tweets are camel case in different orders, make all airlines uppercase so they are all equal

Tweet.Airline = Tweet.Airline.apply(lambda x: x.upper())

    

#create a dictionary to map the all uppercase airlines to the proper naming convention

Tweet_map_airline = {'AMERICANAIR':'American Airlines',

                'JETBLUE':'Jet Blue',

                'SOUTHWESTAIR':'Southwest Airlines',

                'UNITED': 'United Airlines',

                'USAIRWAYS': 'US Airways',

                'VIRGINAMERICA':'Virgin Airlines',

                'DELTA':'Delta Airlines'}

    

#map the uppercase airlines to the proper naming convention

Tweet.Airline = Tweet.Airline.map(Tweet_map_airline)



#display our new airlines!!!

np.unique(Tweet.Airline)
Tweet_conf_df = Tweet[Tweet.airline_sentiment_confidence >= 0.51 ]

#create a copy of our original dataframe and reset the index

date = Tweet_conf_df.reset_index()

#convert the Date column to pandas datetime

date.tweet_created = pd.to_datetime(date.tweet_created)

#Reduce the dates in the date column to only the date and no time stamp using the 'dt.date' method

date.tweet_created = date.tweet_created.dt.date

Tweet_conf_df = date

print(Tweet_conf_df.info())

Tweet_conf_df.head(10)
tweet_df_test = Tweet_conf_df.groupby(['Airline','airline_sentiment']).count().iloc[:,0]

tweet_df_test
def percentages(df, rating = 'negative'):

    if rating == 'negative':

        i = 0

        column = 'Percent Negative Ratings'

    elif rating == 'neutral':

        i = 1

        column = 'Percent Neutral Ratings'

    elif rating == 'positive':

        i = 2

        column = 'Percent Positive Ratings'

        

    #Count of all tweet ratings for each airline (negative, neutral, positive), remove Delta since it only has 2 entries total

    each_airline_ratings_counts = df[df.Airline != 'Delta Airlines'].groupby(['Airline','airline_sentiment']).count().iloc[:,0]

    #Rating tweet total index for each airline:

    #American i

    #Jet Blue i + 3

    #southwest i + 6

    #US Airways i + 9

    #United i + 12

    #Virgin i + 15



    #Count of total tweets about an airline

    total_airline_ratings_counts = df[df.Airline != 'Delta Airlines'].groupby(['Airline'])['airline_sentiment'].count()

    #Airline index in total tweets:

    #American 0

    #Jet Blue 1

    #Southwest 2

    #US Airways 3

    #United 4

    #Virgin 5



    #Create a dictionary of percentage of rating tweets = (each_airline_ratings_counts / total_airline_ratings_counts)

    tweet_ratings_dict = {'American':each_airline_ratings_counts[i] / total_airline_ratings_counts[0],

                'Jet Blue':each_airline_ratings_counts[i + 3] / total_airline_ratings_counts[1],

                'Southwest': each_airline_ratings_counts[i + 6] / total_airline_ratings_counts[2],

                'US Airways': each_airline_ratings_counts[i + 9] / total_airline_ratings_counts[3],

                'United': each_airline_ratings_counts[i + 12] / total_airline_ratings_counts[4],

                'Virgin': each_airline_ratings_counts[i + 15] / total_airline_ratings_counts[5]}



    #make a dataframe from the dictionary

    percent_tweet_ratings = pd.DataFrame.from_dict(tweet_ratings_dict, orient = 'index')

    

    #have to manually set column name when using .from_dict() method

    percent_tweet_ratings.columns = [column]

        

    return percent_tweet_ratings
#Create a df called negative that contains the percent negatives by calling the function above

percent_negative_ratings = percentages(Tweet_conf_df, 'negative')



#Create a df called neutral that contains the percent neutrals by calling the function above

percent_neutral_ratings = percentages(Tweet_conf_df, 'neutral')



#Create a df called positive that contains the percent positives by calling the function above

percent_positive_ratings= percentages(Tweet_conf_df, 'positive')





#concatenate all 3 dataframes of percent ratings

percent_ratings_dataframes_concatenated = merging_airlines_ratings_dataframes(percent_neutral_ratings, percent_negative_ratings, percent_positive_ratings)

print(percent_ratings_dataframes_concatenated)
#graph all of airlines ratings dataframes

ax = percent_ratings_dataframes_concatenated.plot(kind = 'bar', stacked = True, rot = 0, figsize = (15,6))

#set x label

ax.set_xlabel('Airlines')

#set y label

ax.set_ylabel('Percentages')

#move the legend to the bottom of the graph

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),

          fancybox=True, shadow=True, ncol=5)



plt.show()
Tweet_conf_df_ngreason = Tweet_conf_df.reset_index().loc[:,['Airline','negativereason']].dropna().groupby(['Airline','negativereason']).size()

Tweet_conf_df_ngreason.unstack(0).plot(kind = 'bar', figsize = (15,6), rot = 70)
Tweet_conf_day_df = Tweet_conf_df.groupby(['tweet_created','Airline','airline_sentiment']).size()

Tweet_conf_day_df
Tweet_conf_day_df = Tweet_conf_day_df.reset_index()

#Remove delta since it only has 2 entries

Tweet_conf_day_df = Tweet_conf_day_df[Tweet_conf_day_df.Airline != 'Delta Airlines']

#filter to only negative ratings

Tweet_conf_day_df = Tweet_conf_day_df[Tweet_conf_day_df.airline_sentiment == 'negative'].reset_index()

Tweet_conf_day_df = Tweet_conf_day_df.iloc[:,1:5]

#groupby and plot data

ax2 = Tweet_conf_day_df.groupby(['tweet_created','Airline']).sum().unstack().plot(kind = 'bar', figsize = (15,6), rot = 70)

labels = ['American Airlines','Jet Blue','Southwest Airlines','US Airways','United Airlines','Virgin Airlines']

ax2.legend(labels = labels)

ax2.set_xlabel('Date')

ax2.set_ylabel('Negative Tweets')

plt.show()

#We will filter the data to be the data with the negative ratings

Tweet_text_cloud_df=Tweet_conf_df[Tweet_conf_df['airline_sentiment']=='negative']

words = ' '.join(Tweet_text_cloud_df['text'])

#we will remove the links , tags and RT from the text

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

#then we will visualize the cleaned data by word cloud visualizations

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                     ).generate(cleaned_word)



plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
#We will filter the data to be the data with the positive ratings

Tweet_text_cloud_df=Tweet_conf_df[Tweet_conf_df['airline_sentiment']=='positive']

words = ' '.join(Tweet_text_cloud_df['text'])

#we will remove the links , tags and RT from the text

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

#then we will visualize the cleaned data by word cloud visualizations

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                     ).generate(cleaned_word)



plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
#We will filter the data to be the data with the neutral ratings

Tweet_text_cloud_df=Tweet_conf_df[Tweet_conf_df['airline_sentiment']=='neutral']

#we will remove the links , tags and RT from the text

words = ' '.join(Tweet_text_cloud_df['text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

#then we will visualize the cleaned data by word cloud visualizations

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                     ).generate(cleaned_word)



plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
#let's filter tweets text by applying tweet_to_words function to 'text' column 

def tweet_to_words(tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words ))



Tweet_conf_df.text = Tweet_conf_df.text.apply(lambda x: tweet_to_words(x))

# We will code te values in the 'airline_sentiment' column to be numeric values

Tweet_conf_df['airline_sentiment'] = Tweet_conf_df['airline_sentiment'].replace('negative', 0)

Tweet_conf_df['airline_sentiment'] = Tweet_conf_df['airline_sentiment'].replace('neutral', 1)

Tweet_conf_df['airline_sentiment'] = Tweet_conf_df['airline_sentiment'].replace('positive', 2)



vect = CountVectorizer(analyzer = "word")

## Create sparse matrix from the vectorizer

dt_features= vect.fit_transform(Tweet_conf_df['text'])

text_transformed = pd.DataFrame(dt_features.toarray(), columns=vect.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(text_transformed, Tweet_conf_df['airline_sentiment'], test_size=0.2, random_state=456)

# Train a logistic regression

log_reg = LogisticRegression(C=1.0, dual=True, penalty="l2").fit(X_train, y_train)

# Predict the labels

y_predicted = log_reg.predict(X_test)



print('our score is:',  log_reg.score(X_test,y_test))

# Print accuracy score and confusion matrix on test set

print('Accuracy on the test set: ', accuracy_score(y_test, y_predicted))

print(confusion_matrix(y_test, y_predicted)/len(y_test))
print(classification_report(y_predicted, y_test))
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(Tweet_conf_df.text, Tweet_conf_df.airline_sentiment, test_size=0.1, random_state=37)

tk = Tokenizer(num_words= 10000,

               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',

               lower=True,

               split=" ")

tk.fit_on_texts(X_train_d)



X_train_seq = tk.texts_to_sequences(X_train_d)

X_test_seq = tk.texts_to_sequences(X_test_d)



def one_hot_seq(seqs, nb_features = 10000):

    ohs = np.zeros((len(seqs), nb_features))

    for i, s in enumerate(seqs):

        ohs[i, s] = 1.

    return ohs



X_train_oh = one_hot_seq(X_train_seq)

X_test_oh = one_hot_seq(X_test_seq)



le = LabelEncoder()

y_train_le = le.fit_transform(y_train_d)

y_test_le = le.transform(y_test_d)

y_train_oh = to_categorical(y_train_le)

y_test_oh = to_categorical(y_test_le)



X_train_rest, X_valid, y_train_rest, y_valid = train_test_split(X_train_oh, y_train_oh, test_size=0.1, random_state=37)
drop_model = models.Sequential()

drop_model.add(layers.Dense(64, kernel_initializer = 'uniform', activation='relu', input_shape=(10000,)))

drop_model.add(layers.Dropout(0.5))

drop_model.add(layers.Dense(64,kernel_initializer = 'uniform', activation='relu'))

drop_model.add(layers.Dropout(0.5))

drop_model.add(layers.Dense(3, activation='softmax'))



print(drop_model.summary())
drop_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

drop_model.fit(X_train_rest,y_train_rest, batch_size = 64, nb_epoch = 10)