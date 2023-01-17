import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#IRAhandle_tweets_1 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_1.csv")

#IRAhandle_tweets_2 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_2.csv")

#IRAhandle_tweets_3 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_3.csv")

#IRAhandle_tweets_4 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_4.csv")

#IRAhandle_tweets_5 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_5.csv")

#IRAhandle_tweets_6 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_6.csv")

#IRAhandle_tweets_7 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_7.csv")

#IRAhandle_tweets_8 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_8.csv")

#IRAhandle_tweets_9 = pd.read_csv("../input/russian-troll-tweets/IRAhandle_tweets_9.csv")
#Read all CSV files into a list

prefix = '../input/russian-troll-tweets/IRAhandle_tweets_'

ext = '.csv'

frames = [pd.read_csv(prefix + str(i) + ext) for i in range(1,10)]
#Concatenate the dataframes

tweets = pd.concat(frames)
# We can print out the titles of all the columns, this can help us organize our information

print(list(tweets.columns))
#Print Tweets

print(tweets)
#Display Tweets

tweets
#Filter by year 

tweets = tweets[tweets['publish_date'].str.contains('2016')]



tweets
#Filter by Language 

tweets = tweets[tweets['language'].str.contains('English')]



tweets
#Filter by Content 

tweets = tweets[tweets['content'].str.contains('Trump')]



tweets
#Drop Columns

tweets = tweets.drop(['external_author_id', 'post_type','updates','new_june_2018','retweet','harvested_date'], axis=1)



tweets
#Select Columns

tweets = tweets[['content']]



tweets
class WordCounter:

  def __init__(self):

    self.words = {}

    self.blacklist = []



  def add_word(self, word: str) -> int:

    '''

    Adds word if it doesn't exist, increment counter if we already have it.

    Return the count of that word.

    '''

    

    if word in self.words:

      self.words[word] = self.words[word] + 1

    elif word in self.blacklist:

      return 0  

    else:

      self.words[word] = 1

    return self.words[word]



    def getitem(self, word):

      return self.words[word]
#Example using split()

w = tweets.iloc[0]['content']

W2 = tweets.iloc[0]['content'].split()

print(w)

print(W2)
#import progress bar and initialize progress bar 

from tqdm.notebook import tqdm

pb = tqdm(total=len(tweets))





word_counter = WordCounter()

word_counter.blacklist = ['in','you','the','The','I','is','on','a','A','to','To','of','for','and','it',

                          'will','with','that','this','be','at','as','he','his','by','not','they',

                          '-','&','about','has','from','was','have','who','all','say','my','out',

                          'says','up','but','we','like','are','if','our','via','�','This','your','an',]



for index, row in tweets.iterrows():

    pb.update()

    rows = row["content"].split()

    

    for w in rows:

        word_counter.add_word(w)





print(word_counter.words)    
# lambda ARGUMENT: do stuff with ARGUMENT 

# EXAMPLE: add = lambda x, y: x + y

sList = sorted(word_counter.words.items(),key=lambda item: item[1],reverse=True)



mainDict ={key:value for key,value in sList}



mainDict
# turn the mainDict into a list with just the keys and print the top 20 words

topTwenty = list(mainDict.keys())



print(topTwenty[:20])
#Create a for loop to iterate through the dictionary

#Print out both Keys and values

counter = 0

for key,value in mainDict.items():

        if counter > 19:

            break

        print(value,'\t',key)

        counter+=1
#You can also turn the mainDict into a list with just the keys and print the top 20

Topkeys = list(mainDict.keys())

topVal = list(mainDict.values())

kv = list(zip(Topkeys[:20],topVal[:20]))



print(kv)

Final = pd.DataFrame(kv,columns = ["Count","Word"])



Final