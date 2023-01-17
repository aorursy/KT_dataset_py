# Necessary packages
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Plotting
import re, io, json, requests # Essentialls
import tweepy as tp # API to interact with twitter
import nltk # API for feature engineering (NLP tools)
# Note: if this is your first time using nltk,consider running:
#nltk.download('stopwords') 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 17:03:17 2018
@author: andrewcaide
"""

'''
# Original code, stored in 'twitter_credentials.py':
global CONSUMER_KEY
global CONSUMER_SECRET
global ACCESS_TOKEN
global ACCESS_TOKEN_SECRET

CONSUMER_KEY = "cxxxxxxxxxxxxxf"
CONSUMER_SECRET = "xxxxxxxxk"
ACCESS_TOKEN = "41xxxxxxxxxxxxxxxxxxxxxxxM"
ACCESS_TOKEN_SECRET = "1xxxxxxxxx2"
''' #---------------------- END ORIGINAL FILE -----------------------

# Work-around:
class twitter_creds:
    def __init__(self, credential_list):
        self.CONSUMER_KEY = credential_list[0]#CONSUMER_KEY
        self.CONSUMER_SECRET = credential_list[1]#CONSUMER_SECRET
        self.ACCESS_TOKEN = credential_list[2]#ACCESS_TOKEN
        self.ACCESS_TOKEN_SECRET = credential_list[3]#ACCESS_TOKEN_SECRET

        
# ***********************************************************************************
# ***********************************************************************************

#                         !! IMPORTANT !!
#                    NEEDS TO BE EDITED BY **YOU**
# Delete the string and replace it with a string containing your specific key

CONSUMER_KEY = "your_consumer_key [EDIT THIS]"
CONSUMER_SECRET = "your_consumer_secret_key [EDIT THIS]"
ACCESS_TOKEN = "your_access_token [EDIT THIS]"
ACCESS_TOKEN_SECRET = "your_access_token_secret_key [EDIT THIS]"

# ***********************************************************************************
# ***********************************************************************************

# Keep the rest untouched.
credentials = [CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN, ACCESS_TOKEN_SECRET]
    
twitter_credentials = twitter_creds(credentials)
print("Consumer key: {}".format(twitter_credentials.CONSUMER_KEY))
print("Access token: {}".format(twitter_credentials.ACCESS_TOKEN))
# Helper functions
def authenticate_twitter_app():
    
    # Authentication
    consumer_key = twitter_credentials.CONSUMER_KEY 
    consumer_secret = twitter_credentials.CONSUMER_SECRET 
    auth = tp.OAuthHandler(consumer_key, consumer_secret)

    # token stuff
    access_token = twitter_credentials.ACCESS_TOKEN 
    access_token_secret = twitter_credentials.ACCESS_TOKEN_SECRET 
    auth.set_access_token(access_token, access_token_secret)
    return(auth)

def get_user_timeline_tweets(twitter_client, user_list, num_tweets):

    tweets = []
    for user in user_list:
        # This returns tweets & re-tweets, 10 at a time
        # We need to research more about getting larger volume back in time with max_id, since_id, etc.
        print(f'Getting {num_tweets} tweets for {user}. ', end = '')
        try:
            for tweet in tp.Cursor(twitter_client.user_timeline, id=user).items(num_tweets):
                tweets.append(tweet)    
        except tp.RateLimitError:
            print(f'SLEEPING DUE TO RATE LIMIT ERROR!!!!')
            time.sleep(15 * 60)
        except Exception as e:
            print(f'SOME ERROR OCCURRED...PASSING!!!')
            print(e.__doc__)
            pass
    return(tweets)
         
def produce_status_LoDs(statuses):
    
    tweet_LoD = []
    user_LoD = []
    for status in statuses:
        
        tweet_dict = {}
        user_dict = {}
        
        tweet_dict['user_id'] = status.author.id
        tweet_dict['user_screen_name'] = status.author.screen_name
        #tweet_dict['created_at'] = status.created_at.isoformat()
        tweet_dict['created_at'] = str(status.created_at)
        tweet_dict['id'] = status.id
        tweet_dict['id_str'] = status.id_str
        tweet_dict['text'] = status.text
        tweet_dict['source'] = status.source
        tweet_dict['truncated'] = status.truncated
        tweet_dict['retweet_count'] = status.retweet_count
        tweet_dict['favorite_count'] = status.favorite_count
        tweet_dict['lang'] = status.lang
        tweet_dict['is_tweet'] = ((re.search('RT', status.text) == None))
        ###Preimium API only?:  tweet_dict['retweeted_status'] = status.retweeted_status
        ###Preimium API only?:  tweet_dict['reply_count'] = status.reply_count
        ###Premium API only?:  tweet_dict['possibly_sensitive'] = status.possibly_sensitive
        
        tweet_LoD.append(tweet_dict)
                
        # user data
        user_dict['id'] = status.author.id
        #user_dict['id_str'] = status.author.id_str
        user_dict['name'] = status.author.name
        user_dict['screen_name'] = status.author.screen_name
        user_dict['location'] = status.author.location
        #user_dict['url'] = status.author.url
        user_dict['description'] = status.author.description
        user_dict['verified'] = status.author.verified
        user_dict['followers_count'] = status.author.followers_count
        user_dict['listed_count'] = status.author.listed_count
        user_dict['favourites_count'] = status.author.favourites_count
        user_dict['statuses_count'] = status.author.statuses_count
        #user_dict['created_at'] = status.author.created_at.isoformat()
        user_dict['created_at'] = str(status.author.created_at)
        #user_dict['utc_offset'] = status.author.utc_offset
        user_dict['time_zone'] = status.author.time_zone
        user_dict['lang'] = status.author.lang
        
        # Non-twitter, enriched field that dev team is manually managing
        if status.author.verified:
            user_dict['known_bot'] = False
        else:
            user_dict['known_bot'] = False # We will change to true when importing known bots
        user_LoD.append(user_dict)
        
    return(tweet_LoD, user_LoD)
auth = authenticate_twitter_app()
twitter_client = tp.API(auth)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 22:45:11 2018
@authors: andrewcaide and eumarassis
bot list pulled from:
    https://www.nbcnews.com/tech/social-media/now-available-more-200-000-deleted-russian-troll-tweets-n844731
"""

tweets_url = "http://nodeassets.nbcnews.com/russian-twitter-trolls/tweets.csv"
tweets_content = requests.get(tweets_url).content

bots_url = "http://nodeassets.nbcnews.com/russian-twitter-trolls/users.csv"
bots_content = requests.get(bots_url).content

bots = pd.read_csv(io.StringIO(bots_content.decode('utf-8')))
tweets = pd.read_csv(io.StringIO(tweets_content.decode('utf-8')))

#print("Cursory examination of the tweets dataframe:")
#print(tweets.dtypes)

def produce_bot_LoDs(bots, tweets):
    '''
    Read in dataframe of bots and their tweets. Organize them according to Mark's format. 
    
    Args: 
        dataframes - Bots, Tweets
    
    Returns: 
        Cleaned dataframe, with an extra column: 'Bot_Status' = True
    '''
    # Fix tweets:
    tweet_colnames = ['created_at','favorite_count','id','id_str','is_tweet','lang','retweet_count',
                 'source','text','truncated','user_screen_name']
    
    # Ask mark to see if he can pull out hashtags from his tweets!!
    tweets['truncated'] = False
    tweets['is_tweet'] = tweets['text'].apply(lambda x: False if str(x).find('RT') == -1 else True)
    tweets = tweets.rename(columns={'tweet_id': 'id'}) 
    tweets['id_str'] = str(tweets['id'])
    tweets['user_screen_name'] = tweets['user_key']
    tweets['lang'] = 'en'
    #tweets['created_at'] = tweets['created_str']
    
    tweet_output = tweets[tweet_colnames]
    # Fix users:
    user_colnames = ['created_at','description','favourites_count','followers_count','id',
                      'lang','listed_count','location','name','screen_name','statuses_count',
                      'time_zone','verified']
    
    bots_output = bots[user_colnames]
    bots_output['known_bot'] = True
    return(bots_output, tweet_output)

bots_Clean, tweets_Clean = produce_bot_LoDs(bots, tweets)
print(bots_Clean.shape)
# This might not run; it has to be executed locally.
user_json  = bots_Clean.to_json(orient='records')
tweet_json = tweets_Clean.to_json(orient='records')

'''
with open('data/b_tweet_table_out.json', 'w') as outfile:  
    json.dump(tweet_json, outfile)
    
with open('data/b_user_table_out.json', 'w') as outfile:  
    json.dump(user_json, outfile)
'''
# Uncomment above if you'd like to save the data.
# Define list of non-verified users. This will be our verification data.
nv_user_list=['@oovoo_javer_ceo','@slactochile','@jaymijams','@ChefDoubleG',
'@mrpotato','@Rcontreras777','@MissMaseline','@mike434prof','@NonativeEuan',
'@mbspyder','@vaggar99','@AfifaAssel','@esruben','@Victorhuvaldez','@JesiaQuispe',
'@TurnbowRosemary','@todaav','@Pasho53013866','@tonyaba18632641','@ghostsignal1',
'@chubbyleena','@genre_addis','@DarrrellWalraven','@onegearrico','@abadreen',
'@somerice','@unsaltCarthage','@Cmiln01','@Kitter_44','@ashish3vedi',
'@HugoMunissa','@TODthegiant','@LissyBee4','@anna_adamcova','@jerwinbroas2',
'@Queenprominent','@IndianhawkFB','@7998472','@rjerome217','@CharlesNcharg14',
'@Truthseeker1237','@guywpt','@bernoroel','@DavidOrr4','@backworldsman1',
'@jimmythecoat','@wrwveit','@TriggaGhj','@duckmesick','@tyjopow','@mskoch',
'@jaspect_wan','@WiseSparticus','@Mr_AdiSingh','@Live9Fortknox','@mrfridberg',
'@vibolnet','@paulanderson801','@Supanovawhatevs','@politicalpatfan','@DAvidofny1',
'@Tvat_64','@S_Nenov','@HglundNiklas','@LBoertjes','@anBiabhail','@iantuck99',
'@JumahSaisi','@QteleOluwatobi1','@woodgrovect','@LeeThecritch','@mkinisa1',
'@Anfieldvianne','@DonUbani','@JardyRaines','@BagbyCarole','@JopiHuangShen',
'@scottwms84','@gander99','@biller_jon','@aeal_ve','@DesjardinsKarla','@LBonxe',
'@joey_gomez','@anthoamick844','@Brettwadeart','@zac_slocumb','@NatoNogo','@Twu76',
'@Monoclops37','@dwhite612','@_bwright','@InsaneGamer1983','@AsaWatts6','@Niallpolke',
'@84newsnerny','@BrownWilliamF','@MariusD53205774']

# Define list of verified users. Will use these accounts as confirmed 'non-bot's.
v_user_list=['@BarackObama','@rihanna','@realDonaldTrump','@secupp','@ChairmanKimNK',
'@taylorswift13','@ladygaga','@TheEllenShow','@Cristiano','@YouTube','@katyperry',
'@jtimberlake','@KimKardashian','@ArianaGrande','@britneyspears','@cnnbrk','@BillGates',
'@narendramodi','@Oprah','@SecPompeo','@nikkihaley','@SamSifton','@FrankBruni',
'@The_Hank_Poggi','@krassenstein','@TheJordanRachel','@MrsScottBaio',
'@ClaireBerlinski','@java','@JakeSherman','@jaketapper','@jakeowen','@AndrewCMcCarthy',
'@tictoc','@thedailybeast','@mitchellvii','@GadSaad','@Joy_Villa','@RashanAGary',
'@DallasFed','@Gab.ai','@bigleaguepolitics','@Circa','@EmilyMiller','@francesmartel',
'@andersoncooper','@nico_mueller','@NancyGrace','@washingtonpost','@ThePSF', '@pnut',
'@EYJr','@MCRofficial','@RM_Foundation','@tomwaits','@burbunny','@justinbieber',
'@TherealTaraji','@duttypaul','@AvanJogia','@AlecJRoss','@s_vakarchuk','@elongmusk',
'@StephenColletti','@jem','@tonyparker','@vitorbelfort','@jeff_green22',
'@TomJackson57','@robbiewilliams','@AshleyMGreene','@edhornick','@mattdusk',
'@ReggieEvans30','@RachelNichols1','@AndersFoghR','@PalmerReport',
'@KAKA','@Robbie_OC','@josiahandthe','@OKKenna','@CP3','@crystaltamar',
'@MichelleDBeadle','@Jonnyboy77','@kramergirl','@johnwoodRTR','@StevePeers',
'@AdamSchefter','@georgelopez','@CharlieDavies9','@Nicole_Murphy',
'@vkhosla','@NathanPacheco','@SomethingToBurn','@jensstoltenberg','@Devonte_Riley',
'@FreddtAdu','@Erik_Seidel','@Pamela_Brunson','@MMRAW','@russwest44','@shawnieora',
'@wingoz','@ToddBrunson','@NathanFillion','@LaurenLondon','@francescadani',
'@howardhlederer','@MrBlackFrancis','@GordonKljestan','@thehitwoman','@KeriHilson',
'@druidDUDE','@jimjonescapo','@myfamolouslife','@PAULVANDYK','@SteveAustria',
'@bandofhoreses','@jaysean','@justdemi','@MaryBonoUSA','@PaulBrounMD','@jrich23','@Eve6',
'@st_vincent','@Padmasree','@jamiecullum','@GuyKawasaki','@PythonJones','@sffed',
'@howardlindzon','@xonecole','@AlisonSudol','@SuzyWelch','@topchefkevin','@MarcusCooks',
'@Rick_Bayless','@ShaniDavis','@scottylago','@danielralston','@crystalshawanda',
'@TheRealSimonCho','@ItsStephRice','@IvanBabikov','@DennyMdotcom','@TFletchernordic',
'@RockneBru86','@JuliaMancuso','@RyanOBedford','@speedchick428','@JennHeil',
'@katadamek','@kathryn_kang','@alejandrina_gr','@RaymondArroyo','@JonHaidt',
'@DKShrewsbury','@faisalislam','@miqdaad','@michikokakutani','@mehdirhasan','@AbiWilks',
'@hugorifkind','@kylegriffin1','@timothy_stanley','@NAXWELL','@PT_Dawson','@MaiaDunphy',
'@zachheltzel','@KatyWellhousen','@NicholasHoult','@ryanbroems','@LlamaGod','@boozan',
'@DarrenMattocks','@BraulioAmado','@bernierobichaud','@ThisisSIBA','@Jill_Perkins3',
'@D_Breitenstein','@George_McD','@RedAlurk','@NickRobertson10','@kevinvu','@Henry_Kaye',
'@Chris_Biele','@tom_watson','@MikeSegalov','@edballs','@TalbertSwan','@eugenegu',
'@Weinsteinlaw','@BrittMcHenry','@ava','@McFaul','@DaShanneStokes','@funder',
'@BrunoAmato_1','@DirkBlocker','@TrevDon','@DavidYankovich','@KirkDBorne','@JohnLegere',
'@JustinPollard','@MattDudek','@CoachWash56','@RexxLifeRaj','@SageRosenfels18']


print("Number of non-verified users: {}".format(len(nv_user_list)))
print("Number of verified users: {}".format(len(v_user_list)))
# To accurately classify our users, let's pull at least 100 tweets from each account.
num_tweets = 100

# Helper function to get fixed number of tweets and put in results
def get_tweets(twitter_client,v_user_list, num_tweets):
    statuses = get_user_timeline_tweets(twitter_client,v_user_list, num_tweets)
    
    # Create list to write to json file
    tweet_LoD, user_LoD = produce_status_LoDs(statuses)
    
    # Put in DF in case you skip the out/in via json below
    tweet_df = pd.DataFrame(tweet_LoD)
    user_df = pd.DataFrame(user_LoD)
    return(tweet_df, user_df)

# Get verified users, write them to HD
v_tweet_df, v_user_df  = get_tweets(twitter_client,v_user_list, num_tweets)

user_json = v_user_df.to_json(orient='records')
tweet_json = v_tweet_df.to_json(orient='records')
'''
with open('v_tweet_table_out.json', 'w') as outfile:  
    json.dump(tweet_json, outfile)
with open('v_user_table_out.json', 'w') as outfile:  
    json.dump(user_json, outfile)
'''

# Get unverified users, write them to HD
nv_tweet_df, nv_user_df  = get_tweets(twitter_client,nv_user_list, num_tweets)

user_json = nv_user_df.to_json(orient='records')
tweet_json = nv_tweet_df.to_json(orient='records')
'''
with open('data/v_tweet_table_out.json', 'w') as outfile:  
    json.dump(tweet_json, outfile)
with open('data/v_user_table_out.json', 'w') as outfile:  
    json.dump(user_json, outfile)
'''
# Uncomment above if you'd like to save the data.
user_df = nv_user_df.append(v_user_df) #, ignore_index=True)
tweet_df = nv_tweet_df.append(v_tweet_df) #, ignore_index=True)

b_tweet_df = pd.read_json(tweet_json)
b_user_df = pd.read_json(user_json)
'''
with open('data/b_tweet_table_out.json') as json_file:  
    tweet_json = json.load(json_file)

with open('data/b_user_table_out.json') as json_file:  
    user_json = json.load(json_file)
    
tweets_Clean = pd.read_json(tweet_json)
bots_Clean = pd.read_json(user_json)
'''

final_user_dffinal_us  = user_df.append(bots_Clean)
final_tweet_df = tweet_df.append(tweets_Clean)

'''
with open('data/final_tweet_master.json', 'w') as outfile:  
    json.dump(tweet_json, outfile)
    
with open('data/final_user_master.json', 'w') as outfile:  
    json.dump(user_json, outfile)
'''
# Uncomment above if you'd like to save the data.