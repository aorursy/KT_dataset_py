import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from IPython.display import HTML
import requests, json, os
print(os.listdir("../input"))

def load_data():
    try:
        #with open('tweet_master.json') as json_file:  
        with open('../input/twitter-data-tweets/final_tweet_master.json') as json_file:  
            tweet_json = json.load(json_file)
        #with open('tweet_master.json') as json_file:  
        with open('../input/twitter-data-users/final_user_master.json') as json_file:  
            usr_json = json.load(json_file)

    except Exception as e:
        print("Loading Data from Azure - data files do not exist locally")
        print("Warning - these files were since removed by the host. Cheers!")
        '''
        final_url = "https://eumarharvardfiles.blob.core.windows.net/cscis109/final_tweets_master_withNLP.json"
        fina_content = requests.get(final_url).content

        more_url = "https://eumarharvardfiles.blob.core.windows.net/cscis109/more_final_tweets_master_withNLP.json"
        more_content = requests.get(more_url).content

        tweet_json = json.load(io.StringIO(fina_content.decode('utf-8')))
        tweet_json_more =json.load(io.StringIO(more_content.decode('utf-8')))
        '''

    return(pd.read_json(tweet_json), pd.read_json(usr_json))


tweet_df, user_df = load_data()
print(tweet_df.shape, user_df.shape)
user_df.set_index('screen_name',drop=False, inplace=True)

bots = user_df[user_df['known_bot'] == True].copy()
verifieds = user_df[user_df['verified'] == True].copy()
unknowns = user_df[np.logical_and(user_df['known_bot'] != True, user_df['verified'] != True)].copy()

tweets_bot_users = tweet_df.join(bots,'user_screen_name',rsuffix='user',how='right')
tweets_verified_users = tweet_df.join(verifieds,'user_screen_name',rsuffix='user',how='right')
tweets_unknown_users = tweet_df.join(unknowns,'user_screen_name',rsuffix='user',how='right')

tweets_unknown_users["TypeOfUser"] = "Unknown"
tweets_bot_users["TypeOfUser"] = "Bot"
tweets_verified_users["TypeOfUser"] = "Verified"

tweets_unknown_users["TypeOfUser"] = "Unknown"
tweets_bot_users["TypeOfUser"] = "Bot"
tweets_verified_users["TypeOfUser"] = "Verified"

new_tweet_df = pd.concat([tweets_unknown_users,tweets_bot_users, tweets_verified_users], ignore_index=True)
# Prepare DF
new_tweet_df["nlp_key_phrases"] = ""
new_tweet_df["nlp_count_key_phrases"] = 0
new_tweet_df["nlp_sentiment_score"] = 0.0

# Select English/UK twitter users
mask = np.logical_or(new_tweet_df["lang"] == "en", new_tweet_df["lang"] == "uk")

new_tweet_df_english = new_tweet_df[mask].copy()
new_tweet_df_english.reset_index(drop=True, inplace=True)
## THE CODE below uses Microsoft Azure APIs to detect sentiment and key phrases from each Tweet
#https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/quickstarts/python#extract-key-phrases
text_analytics_base_url = "https://eastus.api.cognitive.microsoft.com/text/analytics/v2.0/"
sentiment_api_url = text_analytics_base_url + "sentiment"
key_phrase_api_url = text_analytics_base_url + "keyPhrases"

subscription_key = "SORRY - CAN'T GIVE THIS OUT! Get your own and stick it in here :)"

start_pos = 0 
total = len(new_tweet_df_english)
'''
while start_pos <= total:

    try:
        index1 = start_pos
        index2 = start_pos + 1
        index3 = start_pos + 2
        index4 = start_pos + 3
        index5 = start_pos + 4
        index6 = start_pos + 5
        index7 = start_pos + 6
        index8 = start_pos + 7
        index9 = start_pos + 8
        index10 = start_pos + 9        

        print("Started Processing: Range({0}, {1})".format(index1, index10))
            
        documents = {'documents' : [
          {'id': index1, 'language': 'en', 'text': new_tweet_df_english.loc[index1].text},
          {'id': index2, 'language': 'en', 'text': new_tweet_df_english.loc[index2].text},
          {'id': index3, 'language': 'en', 'text': new_tweet_df_english.loc[index3].text},
          {'id': index4, 'language': 'en', 'text': new_tweet_df_english.loc[index4].text},
          {'id': index5, 'language': 'en', 'text': new_tweet_df_english.loc[index5].text},
          {'id': index6, 'language': 'en', 'text': new_tweet_df_english.loc[index6].text},
          {'id': index7, 'language': 'en', 'text': new_tweet_df_english.loc[index7].text},
          {'id': index8, 'language': 'en', 'text': new_tweet_df_english.loc[index8].text},
          {'id': index9, 'language': 'en', 'text': new_tweet_df_english.loc[index9].text},
          {'id': index10, 'language': 'en', 'text': new_tweet_df_english.loc[index10].text}
        ]}


        headers   = {"Ocp-Apim-Subscription-Key": subscription_key}
        response_key_phrase  = requests.post(key_phrase_api_url, headers=headers, json=documents)
        key_phrases = response_key_phrase.json()

        response_sentiment  = requests.post(sentiment_api_url, headers=headers, json=documents)
        sentiment = response_sentiment.json()


        for document in key_phrases["documents"]:
            id_doc = int(document["id"])
            num_phrases = len(document["keyPhrases"])
            phrases_doc = ",".join(document["keyPhrases"])

            new_tweet_df_english.loc[id_doc, "nlp_key_phrases"] = phrases_doc
            new_tweet_df_english.loc[id_doc, "nlp_count_key_phrases"] = num_phrases    

            #print(id_doc,new_tweet_df.loc[id_doc, "nlp_key_phrases"],  new_tweet_df.loc[id_doc, "nlp_count_key_phrases"])

        for document in sentiment["documents"]:
            id_doc = int(document["id"])
            score = document["score"]

            new_tweet_df_english.loc[id_doc, "nlp_sentiment_score"] = float(score)

            #print (id_doc, new_tweet_df.loc[id_doc, "nlp_sentiment_score"])

        print("Finished Processing: Range({0}, {1})".format(index1, index10))
        
        start_pos = start_pos + 10
    except Exception as e:
        print(f'SOME ERROR OCCURRED...PASSING!!!')
        print(e)
        start_pos = start_pos + 10
        pass
'''
# Errors aren't pretty; commenting this out since above wasn't run.
'''
filter_df = new_tweet_df_english["nlp_key_phrases"] != ""
tweets_with_nlp = new_tweet_df_english[filter_df]
with open('more_final_tweets_master_withNLP.json', 'w') as outfile:  
    json.dump(new_tweet_json , outfile)
'''
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string

print("Let's examine how many users are bots.")
print("\nRaw Counts:")
print(new_tweet_df.groupby("TypeOfUser").count()['known_bot'])
print("\nPercents:")
usr_sum = sum(new_tweet_df.groupby("TypeOfUser").count()['known_bot'])
print(new_tweet_df.groupby("TypeOfUser").count()['known_bot']/usr_sum*100)

print("\nHmmm.")
tt = TweetTokenizer()

print("Original, non-retweeted count of bot tweets: {}.".format(len(new_tweet_df)))
new_tweet_df.dropna(subset=['text'], inplace=True)
print("Count after removing empty tweets: {}.\n".format(len(new_tweet_df)))


useful_cols = ['followers_count','known_bot','text','tokens','name',
               'is_tweet', 'tweet_length'] # Unavailable values: 'nlp_key_phrases','nlp_sentiment_score','nlp_count_key_phrases',
new_tweet_df['tokens'] = new_tweet_df['text'].apply(tt.tokenize)

new_tweet_df['tweet_length'] = new_tweet_df['tokens'].str.len()

bot_texts = new_tweet_df.loc[new_tweet_df.known_bot == True][useful_cols]
real_texts= new_tweet_df.loc[new_tweet_df.known_bot == False][useful_cols]
print("\nLet's look at the distribution of tweet lengths by bot vs tweet lengths by real people. \
We achieve this by tokenizing the tweet sentences with the NLTK package (natural language processing library).\
Next we group by name and find the mean tweet length by user name.\n")
tweet_len_by_bot = bot_texts.groupby(['name']).tweet_length.mean().sort_values(ascending=False)
tweet_len_by_usr = real_texts.groupby(['name']).tweet_length.mean().sort_values(ascending=False)


fig, ax = plt.subplots(1,2, figsize= (16,8))
ax[0].hist([tweet_len_by_bot,tweet_len_by_usr], bins = int(max(tweet_len_by_bot)), 
        label=["Bot Tweets","People Tweets"], alpha=0.3)
ax[0].set_title("Histogram of Mean Tweet Word Length per User(Bot)")
ax[0].set_ylabel("Tweets")
ax[0].set_xlabel("Count")
ax[0].legend()

ax[1].boxplot([bot_texts.groupby(['name']).tweet_length.mean(), real_texts.groupby(['name']).tweet_length.mean()], 
              labels = ['Bot', 'Person'])
ax[1].set_title("Barplot of Tweet Length by User")
ax[1].set_ylabel("Counts")
ax[1].set_xlabel("User Type")
ax[1].legend()

print("The histogram isn't very illuminating; the barplot indicates real people's tweets have a considerably wider word length span, where 75% of tweets by non-bots\
range from 6-30 words per tweet. Bots typically keep their conversations between 10-25 words in length. They both average to around 16-18 words.") 
bot_words = bot_texts.groupby(['name']).tokens.agg(sum)
usr_words = real_texts.groupby(['name']).tokens.agg(sum)
bot_words = pd.DataFrame(bot_words)
usr_words = pd.DataFrame(usr_words)

bot_words.columns = ['words']
usr_words.columns = ['words']

stop_words  = stopwords.words('english') + list(string.punctuation) + [' ','rt',"\'", "...", "..","`",'\"', '–', '’', "I'm", '…','""','“','”']

# Construct list of cleaned words
usr_words['cleaned_words'] = [[word for word in words if word.lower() not in stop_words] 
                                for words in usr_words['words']]
bot_words['cleaned_words'] = [[word for word in words if word.lower() not in stop_words] 
                              for words in bot_words['words']]
freq_per_usr = FreqDist(list([a for b in usr_words.cleaned_words.tolist() for a in b]))
freq_per_bot = FreqDist(list([a for b in bot_words.cleaned_words.tolist() for a in b]))

# Most common words
common_words_bot = pd.DataFrame(freq_per_bot.most_common())
common_words_usr = pd.DataFrame(freq_per_usr.most_common())

cols = ["Words", "Count"]
common_words_bot.columns = cols
common_words_usr.columns = cols

common_words_usr['Frequency'] = common_words_usr['Count']/len(common_words_usr)
common_words_bot['Frequency'] = common_words_bot['Count']/len(common_words_bot)
print("The following calculation was conducted on words greater than two letters. This removes silly 1-emoji \
tweets and the such.\n")

filter1 = (common_words_usr['Words'].str.len()>=3)
filter2 = (common_words_bot['Words'].str.len()>=3)


filtered_usr = common_words_usr.loc[filter1]
filtered_bot = common_words_bot.loc[filter2]

print("\nThe top 15 words used in real tweets (out of {} unique words)::\n".format(len(filtered_usr)))
print(filtered_usr[:15])
print("\nThe top 15 word used by all bots (out of {} unique words):\n".format(len(filtered_bot)))
print(filtered_bot[:15])
naughty_words = filtered_bot[:10]

# Set these to 0
for word in naughty_words['Words']:
    new_tweet_df[word] = 0 # Set to 0
    new_tweet_df[word] = new_tweet_df.apply(lambda row: row['tokens'].count(word), axis=1) # Fill if counted
    
text_by_names = new_tweet_df.groupby(['name']).sum()[naughty_words['Words']]
to_join = new_tweet_df[['name','known_bot']].drop_duplicates().set_index('name')
text_by_names=text_by_names.join(to_join, how='inner').drop_duplicates()
print(text_by_names.head())
bot_texts2 = text_by_names.loc[text_by_names.known_bot == True].join(tweet_len_by_bot, how='inner')
usr_texts2= text_by_names.loc[text_by_names.known_bot == False].join(tweet_len_by_usr, how='inner')

tweet_len_by_bot = pd.DataFrame(tweet_len_by_bot)
tweet_len_by_usr = pd.DataFrame(tweet_len_by_usr)

tweet_len_by_bot.columns = ['mean_tweet_length']
tweet_len_by_usr.columns = ['mean_tweet_length']


fig, ax = plt.subplots(2,5, figsize= (20,8))
ax = ax.ravel()

for i, word in enumerate(naughty_words['Words']):
    ax[i].hist([bot_texts2[word],
                usr_texts2[word]], 
               label=["Bot","Real"])
    ax[i].set_title(word)
    ax[i].set_ylabel('# of Times Word Used')
    ax[i].set_xlabel("# of Accounts")

fig.legend()
plt.tight_layout()
usr_texts2 = usr_texts2.join(tweet_len_by_usr)
bot_texts2 = bot_texts2.join(tweet_len_by_bot)

for word in naughty_words['Words']:
    usr_texts2[word+"_freq"] = usr_texts2[word]/usr_texts2['mean_tweet_length']
    bot_texts2[word+"_freq"] = bot_texts2[word]/bot_texts2['mean_tweet_length']


fig, ax = plt.subplots(2,5, figsize= (20,8))
ax = ax.ravel()

for i, word in enumerate(naughty_words['Words']):
    ax[i].boxplot([bot_texts2[word+"_freq"],
                usr_texts2[word+"_freq"]], labels = ['Bot', 'Person'])
    ax[i].set_title(word)
    ax[i].set_ylabel('Count')
    ax[i].set_xlabel("Accounts")
plt.tight_layout()
