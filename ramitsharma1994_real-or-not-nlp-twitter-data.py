import warnings 
warnings.filterwarnings("ignore")
# importing the libraries 

import string
import re
import numpy as np 
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns 
import emoji
import os
from collections import defaultdict 

from tqdm import tqdm
import statistics 
from statistics import mode, mean

from mlxtend.evaluate import confusion_matrix
from mlxtend.classifier import StackingClassifier

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, recall_score, log_loss, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn_pandas import CategoricalImputer
import xgboost as xgb

from mlxtend.plotting import plot_confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.utils import np_utils


from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

os.listdir()
# Lets load our training and test datasets 

stop_words = stopwords.words("english")

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# increasing the limit of number of rows to be displayed
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_colwidth', -1)
# Creating a class where I will do my preprocessing and apply simple machine learning models

word_net_lemmatizer = WordNetLemmatizer()
w_token = nltk.tokenize.WhitespaceTokenizer()

#creating a class
class Ensembler(object):
    def __init__(self, model_dict, num_folds, optimize = accuracy_score, vectorizer_type = "TfidfVectorizer"):
        
        """
        :param model_dict: dictionary consisting of different models
        :param num_folds : the number of folds 
        :param optimize  : the function to optimize, ex. accuracy score, classification report
        :vectorizer_type : type of vectorizer to use (CountVectorizer or TfidfVectorizer or Word Embedding)
        """
        
        #Initializing
        self.model_dict = model_dict
        self.num_folds = num_folds
        self.optimize = optimize
        self.vectorizer_type = vectorizer_type
        
        
        self.training_data = None
        self.training_target = None
        self.xtrain = None
        self.test_data = None
        self.lbl_enc = None
        self.train_pred_dict = None
        self.test_pred_dict = None
        self.num_classes = None
       
        
    # Creating a simple function to clean the tweets text (like removing emojis, links, punctuations and also doing spell check) 
    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        #removing emojis and unwanted text
        text = emoji.get_emoji_regexp().sub(r"", text)
        text = text.translate(str.maketrans("","", string.punctuation))
        text = text.strip(" ")
        text = text.lower()
        
        #replacing incorrect words with correct words (can also use spellcheck library)
        text = re.sub("rockyfire", 'rocky fire', text)
        text = re.sub('cafire', 'california fire', text)
        text = re.sub('goooooooaaaaaal', 'goal', text)
        text = re.sub('bbcmtd','bbc midlands', text)
        text = re.sub('africanbaze',"african blaze", text)
        text = re.sub('newsnigeria', "news nigeria", text)
        text = re.sub('\x89ûó', "uo", text)
        text = re.sub('superintende', "superintendent", text)
        text = re.sub('southridgelife', "south ridge life", text)
        text = re.sub('carolinaåêablaze', "carolina blaze", text)
        text = re.sub('elbestia',"el bestia", text)
        text = re.sub('\x89û', "u", text)
        text = re.sub('news24680', "news 24680", text)
        text = re.sub('nxwestmidlands', "west midlands", text)
        text = re.sub('nashvilletraffic', "nashville traffic", text)
        text = re.sub('personalinjury', "personal injury", text)
        text = re.sub('otleyhour', 'otley hour', text)
        text = re.sub('caraccidentlawyer', 'car accident lawyer', text)
        text = re.sub('teeû', 'teen', text)
        text = re.sub('bigrigradio', 'big radio', text)
        text = re.sub('sleepjunkies', 'sleep junky', text)
        text = re.sub('hwymagellan', 'highway magellan', text)
        text = re.sub('080615', ' ', text)
        text = re.sub('110358', ' ', text)
        text = re.sub('accidentwho', 'accident who', text)
        text = re.sub('truckcrash', 'truck crash', text)
        text = re.sub('crashgt', 'crash gate', text)
        text = re.sub('i540', 'I 540', text)
        text = re.sub('fyi', 'for your information', text)
        text = re.sub('cadfyi', 'cad for your information', text)
        text = re.sub(' n ', ' and ', text)
        text = re.sub('confû', 'conf', text)
        text = re.sub('nh3a', 'national highway 3a', text)
        text = re.sub('damagewpd1600', 'damage wpd 1600', text)
        text = re.sub('nearfatal', 'near fatal', text)
        text = re.sub('southaccident', 'south accident', text)
        text = re.sub('rdconsider', 'consider', text)
        text = re.sub('your4state', 'your for state', text)
        text = re.sub('measuresarrestpastornganga', 'measures arrest pastor nganga', text)
        text = re.sub('aftershockdelo', 'after shock delo', text)
        text = re.sub('trapmusic', 'trap music', text)
        text = re.sub('icesû', 'ices', text)
        text = re.sub('growingupspoiled', 'growing up spoiled', text)
        text = re.sub('kjfordays', 'kj for days', text)
        text = re.sub('wisdomwed', 'wisdom wed', text)
        text = re.sub('ã¢', 'ac', text)
        text = re.sub('fullã¢', 'full ac', text)
        text = re.sub('esquireattire', 'esquire attire', text)
        text = re.sub('wdyouth', 'wd youth', text)
        text = re.sub('onfireanders', 'on fire anders', text)
        text = re.sub('aftershockorg', 'after shock org', text)
        text = re.sub('watchthevideo', 'watch the video', text)
        text = re.sub('wednesdayû', 'Wednesday', text)
        text = re.sub('freakyû', 'freaky', text)
        text = re.sub('airplaneåê29072015', 'airplane', text)
        text = re.sub('canûªt', 'can not', text)
        text = re.sub('randomthought', 'random thought', text)
        text = re.sub('rejectdcartoons', 'rejected cartoons', text)
        text = re.sub('û÷minimum', 'minimum', text)
        text = re.sub('wageûª', 'wage', text)
        text = re.sub('andû', 'and', text)
        text = re.sub('celticindeed', 'celtic indeed', text)
        text = re.sub('viralspell', 'viral spell', text)
        text = re.sub('suregod', 'sure god', text)
        text = re.sub('breakfastone', 'break fast tone', text)
        text = re.sub('uptotheminute', 'upto the minute', text)
        text = re.sub('stormbeard', 'storm beard', text)
        text = re.sub('steellord', 'steel lord', text)
        text = re.sub('fantasticfourfant4sticwhatever', 'fantastic four whatever', text)
        text = re.sub('starmade', 'star made', text)
        text = re.sub('signatureschange', 'signatures change', text)
        text = re.sub('petitiontake', 'petition take', text)
        text = re.sub('dieplease', 'die please', text)
        text = re.sub('warmbodies', 'warm bodies', text)
        text = re.sub('geekapocalypse', 'geek apocalypse', text)
        text = re.sub('doublecups', 'double cups', text)
        text = re.sub('wifekids', 'wife kids', text)
        text = re.sub('whitewalkers', 'white walkers', text)
        text = re.sub('historicchurch', 'historic church', text)
        text = re.sub('newsintweets', 'news in tweets', text)
        text = re.sub('griefûª', 'grief', text)
        text = re.sub('countynews', 'county news', text)
        text = re.sub('û÷politics', 'politics', text)
        text = re.sub('chicagoarea', 'chicago area', text)
        text = re.sub('theadvocatemag', 'the advocate mag', text)
        text = re.sub('arsonû', 'arson', text)
        text = re.sub('cloudygoldrush', 'cloudy gold rush', text)
        text = re.sub('uniteblue', 'unite blue', text)
        text = re.sub('tonightûªs', 'tonights', text)
        text = re.sub('arsonistmusic', 'arsonist music', text)
        text = re.sub('slimebeast', 'slime beast', text)
        text = re.sub('bestcomedyvine', 'best comedy vine', text)
        text = re.sub('localarsonist', 'local arsonist', text)
        text = re.sub('nativehuman', 'native human', text)
        text = re.sub('myreligion', 'my religion', text)
        text = re.sub('ûïhatchet', 'hatchet', text)
        text = re.sub('controlû', 'control', text)
        text = re.sub('4suspected', 'four suspected', text)
        text = re.sub('acebreakingnews', 'ace breaking news', text)
        text = re.sub('attackshare', 'attack share', text)
        text = re.sub('obamadhs', 'obama dhs', text)
        text = re.sub('blazerfan', 'blazer fan', text)
        text = re.sub('benothing', 'be nothing', text)
        text = re.sub('daytonarea', 'dayton area', text)
        text = re.sub('messeymetoo', 'messy me too', text)
        text = re.sub('robotcoingame', 'robot coin game', text)
        text = re.sub('freebitcoin', 'free bitcoin', text)
        text = re.sub('sportsroadhouse', 'sports road house', text)
        text = re.sub('weddinghour', 'wedding hour', text)
        text = re.sub('ghostoftheav', 'ghost of the av', text)
        text = re.sub('fuelgas', 'fuel gas', text)
        text = re.sub('û÷avalancheûª', 'avlanche', text)
        text = re.sub('coloradoavalanche', 'colorado avlanche', text)
        text = re.sub('mildmannered', 'mild mannered', text)
        text = re.sub('neur0sis', 'neurosis', text)
        text = re.sub('cbsbigbrother', 'cbs big brother', text)
        text = re.sub('sexydragonmagic', 'sexy dragon magic', text)
        text = re.sub('detroitpls', 'detroit please', text)
        text = re.sub('postbattle', 'post battle', text)
        text = re.sub('httû', ' ', text)
        text = re.sub('foxnewû', 'foxnews', text)
        text = re.sub('bioterû', 'bio terrorism', text)
        text = re.sub('infectiousdiseases', 'infectious diseases', text)
        text = re.sub('thelonevirologi', 'the lone virology', text)
        text = re.sub('clergyforced', 'clergy forced', text)
        text = re.sub('bioterrorismap', 'bio terrorism', text)
        text = re.sub('digitalhealth', 'digital health', text)
        text = re.sub('bioterrorismim', 'bio terrorism', text)
        text = re.sub('hostageamp2', 'hostage and 2', text)
        text = re.sub('wbioterrorismampuse', 'bioterrorism puse', text)
        text = re.sub('harvardu', 'harvard university', text)
        text = re.sub('irandeal', 'iran deal', text)
        text = re.sub('cdcgov', 'cdc government', text)
        text = re.sub('raisinfingers', 'raising fingers', text)
        text = re.sub('skywars', 'sky wars', text)
        text = re.sub('agochicago', 'ago chicago', text)
        text = re.sub('thisispublichealth', 'this is public health', text)
        text = re.sub('sothwest', 'south west', text)
        text = re.sub('weekold', 'week old', text)
        text = re.sub('fireû', 'fire', text)
        text = re.sub('artisteoftheweekfact', 'artist of the week fact', text)
        text = re.sub('clubbanger', 'club banger', text)
        text = re.sub('listenlive', 'listen live', text)
        text = re.sub('weatherstay', 'weather stay', text)
        text = re.sub('transcendblazing', 'transcend blazing', text)
        text = re.sub('stoponesounds', 'stop one sounds', text)
        text = re.sub('stickynyc', 'sticky new york city', text)
        text = re.sub('95roots', '95 roots', text)
        text = re.sub('blazingben', 'blazing ben', text)
        text = re.sub('shouout', 'shout out', text)
        text = re.sub('s3xleak', 'sex leak', text)
        text = re.sub('ph0tos', 'photos', text)
        text = re.sub('exp0sed', 'exposed', text)
        text = re.sub('notû', 'not', text)
        text = re.sub('blazingelwoods', 'blazing el woods', text)
        text = re.sub('funkylilshack', 'funky little shack', text)
        text = re.sub('wellgrounded', 'well grounded', text)
        text = re.sub('sodamntrue', 'so damn true', text)
        text = re.sub('hopeinhearts', 'hope in hearts', text)
        text = re.sub('onlyftf', 'only for this Friday', text)
        text = re.sub('robsimss', 'rob sims', text)
        text = re.sub('cantmisskid', 'can not miss kid', text)
        text = re.sub('yahooschwab', 'yahoo schwab', text)
        text = re.sub('fiascothat', 'fiasco that', text)
        text = re.sub('harperanetflixshow', 'harper net flix show', text)
        text = re.sub('stopharper', 'stop harper', text)
        text = re.sub('graywardens', 'gray wardens', text)
        text = re.sub('realhotcullen', 'real hot cullen', text)
        text = re.sub('developmentû', 'development', text)
        text = re.sub('iclowns', 'I clowns', text)
        text = re.sub('2iclown', 'two I clown', text)
        text = re.sub('revolutionblight', 'revolution light', text)
        text = re.sub('healthweekly1', 'health weekly', text)
        text = re.sub('amateurnester', 'amateur nester', text)
        text = re.sub('parksboardfacts', 'parks board facts', text)
        text = re.sub('stevenontwatter', 'steven on twitter', text)
        text = re.sub('pussyxdestroyer', 'pussy destroyer', text)
        text = re.sub('radioriffrocks', 'radio riff rocks', text)
        text = re.sub('tweet4taiji', 'tweet for taiji', text)
        text = re.sub('blizzardfans', 'blizzard fans', text)
        text = re.sub('blizzardgamin', 'blizzard gaming', text)
        text = re.sub('fairx818x', 'fair', text)
        text = re.sub('playoverwatch', 'play over watch', text)
        text = re.sub('blizzardcs', 'blizzards', text)
        text = re.sub('blizzarddraco', 'blizzard draco', text)
        text = re.sub('lonewolffur', 'lone wolf fur', text)
        text = re.sub('bubblycuteone', 'bubbly cute one', text)
        text = re.sub('nailreal', 'nail real', text)
        text = re.sub('bookanother', 'book another', text)
        text = re.sub('chamberedblood', 'chambered blood', text)
        text = re.sub('speakingfromexperience', 'speaking from experience', text)
        text = re.sub('decisionsondecisions', 'decisions on decisions', text)
        text = re.sub('dangerousbeans', 'dangerous beans', text)
        text = re.sub('5sos', '5 sos', text)
        text = re.sub('everwhe', 'everywhere', text)
        text = re.sub('bloodû', 'blood', text)
        text = re.sub('butterlondon', 'butter london', text)
        text = re.sub('bbloggers', 'bloggers', text)
        text = re.sub('resigninshame', 'resign in shame', text)
        text = re.sub('kingûªs', 'kings', text)
        text = re.sub('û÷the', 'the', text)
        text = re.sub('towerûª', 'tower', text)
        text = re.sub('thedarktower', 'the dark tower', text)
        text = re.sub('bdisgusting', ' disgusting', text)
        text = re.sub('wwwbigbaldhead', 'big bald head', text)
        text = re.sub('jessienojoke', 'jessie no joke', text)
        text = re.sub('chxrmingprince', 'charming prince', text)
        text = re.sub('indiansfor', 'indians for', text)
        text = re.sub('bloodymonday', 'bloody Monday', text)
        text = re.sub('tvshowtime', 'tv showtime', text)
        text = re.sub('machinegunkelly', 'machine gun kelly', text)
        text = re.sub('thisdayinhistory', 'this day in history', text)
        text = re.sub('taylorswift13', 'taylor swift 13', text)
        text = re.sub('musicadvisory', 'music advisory', text)
        text = re.sub('weûªre', 'we are', text)
        text = re.sub('weûªve', 'we have', text)
        text = re.sub('theboyofmasks', 'the boy of masks', text)
        text = re.sub('gentlementhe', 'gentle men the', text)
        text = re.sub('kalinandmyles', 'kalin and myles', text)
        text = re.sub('kalinwhite', 'kalin white', text)
        text = re.sub('givebackkalinwhiteaccount', 'give back kalin white account', text)
        text = re.sub('princessduck', 'princess duck', text)
        text = re.sub('dogûªs', 'dogs', text)
        text = re.sub('hopefulbatgirl', 'hopeful bat girl', text)
        text = re.sub('piperwearsthepants', 'piper wears the pants', text)
        text = re.sub('questergirl', 'quester girl', text)
        text = re.sub('readû', 'read', text)
        text = re.sub('bagû', 'bag', text)
        text = re.sub('deliciousvomit', 'delicious vomit', text)
        text = re.sub('today4got', 'today forgot', text)
        text = re.sub('ovofest', 'ovo fest', text)
        text = re.sub('2k15', '2015', text)
        text = re.sub('officialrealrap', 'official real rap', text)
        text = re.sub('im2ad', 'I am too ad', text)
        text = re.sub('bodybagging', 'body bagging', text)
        text = re.sub('womengirls', 'women girls', text)
        text = re.sub('boomerangtime', 'boomerang time', text)
        text = re.sub('û÷institute', 'institute', text)
        text = re.sub('peaceûª', 'peace', text)
        text = re.sub('moving2k15', 'moving 2015', text)
        text = re.sub('expertwhiner', 'expert whiner', text)
        text = re.sub('shopûªs', 'shop', text)
        text = re.sub('cutekitten', 'cute kitten', text)
        text = re.sub('catsofinstagram', 'cats of instagram', text)
        text = re.sub('summerinsweden', 'summer in sweden', text)
        text = re.sub('ûïparties', 'parties', text)
        text = re.sub('drivingû', 'driving', text)
        text = re.sub('youûªll', 'you all', text)
        text = re.sub('û÷body', 'body', text)
        text = re.sub('bagsûª', 'bags', text)
        text = re.sub('whatcanthedo', 'what can the do', text)
        text = re.sub('70year', '70 years', text)
        text = re.sub('hatchetwielding', 'hatchet wielding', text)
        text = re.sub('invadedbombed', 'invaded bombed', text)
        text = re.sub('bombedout', 'bombed out', text)
        text = re.sub('elephantintheroom', 'elephant in the room', text)
        text = re.sub('abombed', 'a bombed', text)
        text = re.sub('tblack', 'black', text)
        text = re.sub('bellybombed', 'belly bombed', text)
        text = re.sub('teamstream', 'team stream', text)
        text = re.sub('beyondgps', 'beyond gps', text)
        text = re.sub('cityamp3others', 'city and three others', text)
        text = re.sub('scheduleû', 'schedule', text)
        text = re.sub('moscowghost', 'moscow ghost', text)
        text = re.sub('banthebomb', 'ban the bomb', text)
        text = re.sub('setting4success', 'setting for success', text)
        text = re.sub('pearlharbor', 'pearl harbor', text)
        text = re.sub('push2left', 'push to left', text)
        text = re.sub('snapharmony', 'snap harmony', text)
        text = re.sub('australiaûªs', 'australia', text)
        text = re.sub('huffpostarts', 'huffpost arts', text)
        text = re.sub('jewishpress', 'jewish press', text)
        text = re.sub('bloopandablast', 'bloop and a blast', text)
        text = re.sub('traintragedy', 'train tragedy', text)
        text = re.sub('slingnews', 'sling news', text)
        text = re.sub('urgentthere', 'urgent there', text)
        text = re.sub('blacklivesmatter', 'black lives matter', text)
        text = re.sub('fewmoretweets', 'few more tweets', text)
        text = re.sub('9newsmornings', '9news mornings', text)
        text = re.sub('strikesstrikes', 'strikes', text)
        text = re.sub('doctorfluxx', 'doctor flux', text)
        text = re.sub('thestrain', 'the strain', text)
        text = re.sub('newyorkû', 'new york', text)
        text = re.sub('tweetlikeitsseptember11th2001', 'tweet like it is september 11th 2011', text)
        text = re.sub('222pm', '2:22 pm', text)
        text = re.sub('ppsellsbabyparts', 'pp sells baby parts', text)
        text = re.sub('bbcintroducing', 'bbc introducing', text)
        text = re.sub('giantgiantsound', 'giant giant sound', text)
        text = re.sub('threealarm', 'three alarm', text)
        text = re.sub('3alarm', 'three alarm', text)
        text = re.sub('wildlionx3', 'wild lion', text)
        text = re.sub('cubstalk', 'cub stalk', text)
        text = re.sub('letsfootball', 'lets football', text)
        text = re.sub('totteham', 'tottenham', text)
        text = re.sub('thatûªs', 'that is', text)
        text = re.sub('burnfat', 'burn fat', text)
        text = re.sub('rvaping101', 'vaping 101', text)
        text = re.sub('lightningcaused', 'lightening caused', text)
        text = re.sub('mountainsû', 'mountains', text)
        text = re.sub('sniiiiiiff', 'sniff', text)
        text = re.sub('youûªre', 'you are', text)
        text = re.sub('foxnewsvideo', 'foxnews video', text)
        text = re.sub('forestservice', 'forest service', text)
        text = re.sub('buildingsûówe', 'buildings we', text)
        text = re.sub('progress4ohio', 'progress for ohio', text)
        text = re.sub('slashandburn', 'slash and burn', text)
        text = re.sub('jamaicaobserver', 'jamaica observer', text)
        text = re.sub('cnewslive', 'cnews live', text)
        text = re.sub('appreciativeinquiry', 'appreciative inquiry', text)
        text = re.sub('standwithpp', 'stand with pp', text)
        text = re.sub('scaryeven', 'scary even', text)
        text = re.sub('attackclose', 'attack close', text)               
        
        return text

    # Creating another function to lemmatize the text 
    def lemmatizer(self, text):
        lis = []
        words = text.split()
        for word in words: 
            if(word not in stop_words):
                lis.append(word_net_lemmatizer.lemmatize(word, pos = 'v'))
            else:
                continue        
        return lis

    # Creating function to create embedding_index which we will use for word embeddings
    def word_embedding_index(self):
        # loading the glove vectors in a dictionary
        embeddings_index = {}
        f = open('/kaggle/input/glove42b300dtxt/glove.42B.300d.txt', encoding = 'utf-8')
        print("Starting Embedding")
        for line in (f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = coefs
        f.close()
        print("Embedding Done")
        return embeddings_index
        
    
    
    # This function will create a normalized vector for the whole sentence
    def sen2vect(self, s, embedding_index):
        embeddings_index = embedding_index 
        words= str(s)       #typecasting to string
        words = word_tokenize(words)    #creating a word_token from the sentence/tweets that we have
        words = [w for w in words if not w in stop_words]    #removing the stopwords from tweet
        words = [w for w in words if w.isalpha()]            #only keeping the words
    
        M = []
        
        #using for loop to store all the coefficients in a list
        for w in words:
            try:
                M.append(embeddings_index[w])               #using embedding_index creating a token from that sentence
            except:
                continue
        
        M = np.array(M)
        v = M.sum(axis = 0)
        #this is used to normalize the vector
        if(type(v) !=np.ndarray):
            return np.zeros(300)
        return v/np.sqrt((v ** 2).sum())
    
    
    # Function definition for plotting the confusion matrix 
    def conf_matrix(self, data, Labels = ['No-Disaster', 'Disaster']):
        matrix_data = data
        count = 1
        # using for loop to access the data (its a dictionary)
        for name, data in matrix_data.items():
        
            # setting the figure size
            plt.figure(figsize=(8,12), facecolor='white')
        
            # assigning the number of subplots and with each count increment the axis
            ax = plt.subplot(2, 1,count)
        
            # using seaborn heatmap to plot confusion matrix
            sns.heatmap(data, xticklabels= Labels, yticklabels= Labels, annot = True, ax = ax, fmt= 'g')
        
            # setting the title name
            ax.set_title("Confusion Matrix of {}".format(name))
        
            plt.xlabel("Predicted Class") #setting the x label
            plt.ylabel("True Class")   # setting the y label
            plt.show()
            count = count +1   # incrementor to increase the value of axis with each loop 


    #Here creating a function for preprocessing, all the preprocessing steps are done here (cleaning text, lemmatizing, word _embedding)
    def preprocess(self, training_data, test_data):
        """
        :param training_data: training_data in tabular format
        :param test_data    : test_data in tabular format
        :return             : return preprocessed data for modelling
        """
        #initializing variables
        self.training_data = training_data
        self.test_data = test_data
        self.vectorizer_type = self.vectorizer_type
        
        #Here I am using a clean_text() function to remove html strings and emojis
        self.training_data['text'] = self.training_data["text"].apply(lambda x: self.clean_text(x))
        self.test_data['text'] = self.test_data['text'].apply(lambda x: self.clean_text(x))
            
        # Lets use lemmatize function to lemmatize the text
        self.training_data['text'] = self.training_data["text"].apply(lambda x: " ".join(self.lemmatizer(x)))
        self.test_data['text'] = self.test_data['text'].apply(lambda x: " ".join(self.lemmatizer(x)))
            
        self.training_data['text']
        # Here I am also dropping all the columns except for the text column
        self.training_data = self.training_data.drop(['id', 'keyword','location'],axis = 1)
        self.test_data = self.test_data.drop(['id', 'keyword', 'location'], axis = 1)
        
        if(self.vectorizer_type == "TfidfVectorizer"):
            #lets use tfidf vectorizer to convert categorical columns to numerical columns
            self.tfv = TfidfVectorizer(min_df = 3, max_features = 300, strip_accents="unicode", analyzer="word", token_pattern=r'\w{1,}', ngram_range= (1,3), use_idf= 1, smooth_idf = 1, sublinear_tf = 1, stop_words='english')
            
            #fitting the tfidf on the training data
            self.tfv.fit(self.training_data['text'])
            
            #transforming both train and test data using tfidf vectorizer
            train_transformed = self.tfv.transform(self.training_data['text'])
            test_transformed = self.tfv.transform(self.test_data['text'])
                
        elif(self.vectorizer_type == "CountVectorizer"):
            #lets use tfidf vectorizer to convert categorical columns to numerical columns
            self.ctv =CountVectorizer(max_features = 300,analyzer= 'word', token_pattern=r'\w{1,}',ngram_range= (1,3), stop_words='english')
            
            #fitting the count vectorizer on the training data
            self.ctv.fit(self.training_data['text'])
            
            #transforming both train and test data using countVectorizer
            train_transformed = self.ctv.transform(self.training_data['text'])
            test_transformed = self.ctv.transform(self.test_data['text'])
                
        elif(self.vectorizer_type == "WordEmbedding"):
            #Calling my word_embedding_index function that I initialized above to create embedding index
            embeddings_index = self.word_embedding_index()
            print("Transforming the dataset")
            
            #Now, using my embedding_index, transforming both train and test data using sen2vect function.
            train_transformed = [self.sen2vect(x, embeddings_index) for x in (self.training_data['text'])]
            test_transformed = [self.sen2vect(x, embeddings_index) for x in (self.test_data['text'])]
            
            #converting both the train and test transformed to numpy array
            train_transformed = np.array(train_transformed)
            test_transformed = np.array(test_transformed)
            print("Dataset Transformed. Next step Modelling")
        
        #returning 
        return self.training_data, self.test_data, train_transformed, test_transformed
    
    
    #Creating a fit_predict_train_valid function, here i am splitting the train data into train and valid and applying simple
    #machine learning models
    def fit_predict_train_valid(self,train_transformed, test_transformed, model_dict, vectorizer_type = "TfidfVectorizer"):
        
        #initalizing 
        self.train_transformed =train_transformed
        self.test_transformed  = test_transformed
        self.model_dict = model_dict    
         
        #using stratifiedKFold to split the train data into train and valid
        skf = StratifiedKFold(n_splits = self.num_folds, random_state = 0, shuffle =True)
        
        #Initializng some required lists and dictionaries
        count = 0   #this will keep in check the number of folds
        single_model_score = {}    #storing all the accuracy scores in a dictionary
        stack_model_score = []    #a list to store all the scores of stack model
        model_list = []            #creating a list of models that we are using
        mean_dict = {}           #creating a mean_dict to store the mean of accuracy scores of all the models
        
        #creating a for loop to split our data into train and valid
        for train_index, valid_index in skf.split(train_transformed, self.training_data['target']):
            count = count+1
            print("Training for Fold: {}".format(count))
            
            #splitting into xtrain, xvalid, ytrain, yvalid
            xtrain, xvalid = train_transformed[train_index], train_transformed[valid_index]
            ytrain, yvalid = self.training_data['target'][train_index], self.training_data['target'][valid_index]
            
            #using for loop to run all the models inside the model_dict dictionary
            for name, model in model_dict.items():
                
                #initializing the model
                model = model
                #fitting the model
                model.fit(xtrain, ytrain)
                
                #predicting the xvalid from our model
                y_pred = model.predict(xvalid)
                
                #calculating the accuracy scores
                score = accuracy_score(yvalid, y_pred)
                
                #using if else function to store the accuracy scores in the dictionary. Storing at as a list of values
                if(name not in single_model_score):
                    single_model_score[name] = [score]
                else: 
                    single_model_score[name].append(score)
                    
                #appening all the models in model_list
                model_list.append(model)
                
                #printing the accuracy score of each model in every fold
                print("\tAccuracy score of {}:{:.3f}".format(name, score))
            
            # here i am using stacking classifier, an ensemble technique to see if we can improve the accuracy 
            stack_classifier = StackingClassifier(classifiers = model_list, meta_classifier= xgb.XGBClassifier(n_estimators =  400, learning_rate =0.1, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',nthread=4, scale_pos_weight=1))
            
            #fitting the stack_classifier on xtrain and ytrain
            stack_classifier.fit(xtrain, ytrain)    
            
            #predicting the results of xvalid using stack_classifier
            final_pred = stack_classifier.predict(xvalid)
            
            #calculating the accuracy score of our stack_Classifier
            stack_score = accuracy_score(yvalid, final_pred)
            
            #using stack_model_Score list to store the accuracy scores in different folds
            stack_model_score.append(stack_score)
            
            #printing the accuracy score of stack classifier with each fold
            print("\tAccuracy score by Stacking all classifier is: {:.3f}".format(stack_score))
        
     
        #using for loop to print the mean_accuracy scores of all the models in all the folds
        for name, score in single_model_score.items():
            print("Mean Accuracy score of {} is: {:.3f}".format(name, np.mean(score)))
            #storing all the mean accuracies in a dictionary
            mean_dict[name] = np.mean(score)
        print("Mean Accuracy score of Stacking of classifier is:{:.3f} ".format(mean(stack_model_score)))
        
        #returning mean_dict and model_list which we will use in our next function
        return mean_dict, model_dict
    
    
    #this function will use the best model from our previous function and train the whole training dataset and then will
    #do a prediction on the unseen test data and will create a submission file for kaggle
    def final_train_predict(self, train_transformed, ytrain_target, test_transformed, model_dict, score_dict, submission):
        
        #initializing
        self.train_transformed = train_transformed
        self.test_transformed = test_transformed
        self.model_dict = model_dict
        
        sample = submission
        mean_score_dict = score_dict
        y_true = ytrain_target
        
        #initializing the lists that are required to plot the roc curve
        model_list = []
        auc_list = []
        fpr_list = []
        tpr_list = []
        
        #from the mean_dict we are getting the model from which we got the maximum accuracies in our validation data
        modelName_max_accuracy = max(mean_score_dict, key = mean_score_dict.get)
        
        #using for loop to go through all the models in our model_dict dictionary
        for name, model in self.model_dict.items():
            
            #appending the model in a list which we will use for stacking classifier
            model_list.append(model)
            
            #using if statement to only use the model from which we got the maximum accuracy
            if(modelName_max_accuracy in name):
                #initializing the model
                model = model
                #fitting the whole train data in our model
                model.fit(train_transformed, y_true)
                
                #predicting the train data and test data using our model
                ytrain_pred = model.predict(train_transformed)
                ytest_pred = model.predict(test_transformed)
                
                #creating a confusion matrix from our predicitions. Since, test data is unseen we will use our training data for that
                confusionMatrix = confusion_matrix(y_true, ytrain_pred)
                
                #calculating fpr, tpr and auc scores
                fpr, tpr, threshold = roc_curve(y_true, ytrain_pred)
                auc1 = auc(fpr, tpr)
                
                #storing all the scores in a list which we will use later for plotting the roc curve
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                auc_list.append(auc1)
                
                #printing the accuracy score of our model on train data
                print("Accuracy score of {} is:{:.3f}".format(modelName_max_accuracy, accuracy_score(y_true, ytrain_pred)))
                
                #creating a csv submission file
                sample['target'] = ytest_pred
                sample.to_csv(modelName_max_accuracy+'_'+self.vectorizer_type+'_submission.csv', index = False)
            
            #else statement to continue if the model didn't preform good in validation data
            else: 
                continue
            
            
            #Lets try the whole training dataset on stacking classifier
            stack_classifier = StackingClassifier(classifiers = model_list, meta_classifier= xgb.XGBClassifier(n_estimators =  400, learning_rate =0.1, max_depth=5, min_child_weight=1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',nthread=4, scale_pos_weight=1))
            
            #fitting the training data on stacking classifier
            stack_classifier.fit(train_transformed, y_true)
            
            #predicting the train and test data results using stacking classifier
            stack_ytrain_pred = stack_classifier.predict(train_transformed)    
            stack_ytest_pred = stack_classifier.predict(test_transformed)
            
            #creating a confusion matrix from our prediction
            confusionMat = confusion_matrix(y_true, stack_ytrain_pred)
            
            #calculating the fpr, tpr and auc scores from the results of stacking classifier
            fpr_stack, tpr_stack, threshold_stack = roc_curve(y_true, stack_ytrain_pred)
            auc_stack = auc(fpr_stack, tpr_stack)
            
            #storing all the results in a list
            fpr_list.append(fpr_stack)
            tpr_list.append(tpr_stack)
            auc_list.append(auc_stack)
            
            #printing the accuracy of my stack_Classifier on training data
            print("Accuracy score of {} is:{:.3f}".format("StackingClassifier", accuracy_score(y_true, stack_ytrain_pred)))
            
            #creating a csv submission file for stack of classifiers
            sample['target'] = stack_ytest_pred
            sample.to_csv("Stack_"+self.vectorizer_type+"_submission.csv", index = False)
            
            #plotting the confusion matrix using the con_matrix function initialized before the preprocess steps
            matrix_data = {modelName_max_accuracy: confusionMatrix, "StackingClassifier": confusionMat}
            self.conf_matrix(matrix_data)
            
            #returning required lists
            return fpr_list, tpr_list, auc_list
        
    
#Here I am using the above class to implement countVectorizer
vect = "CountVectorizer"

#initializing the model dictionary that we will use for modeling
model_dict = {"Logistic Regression": LogisticRegression(C = 1.0, max_iter = 10000), 
              "Random Forest Classifier": RandomForestClassifier(n_estimators = 600),
              "XGBoost Classifier": xgb.XGBClassifier(n_estimators =  400, learning_rate =0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                 nthread=4, scale_pos_weight=1)}

#initializing an object for class
ensCount = Ensembler(model_dict = model_dict, num_folds = 5, optimize=accuracy_score, vectorizer_type =vect)
#preprocessing using original train and test data. Return values from this function are, preprocessed training set, test set and 
#transformed train set and test set

training, testing, train_tran_count, test_tran_count =  ensCount.preprocess(train, test)
#splitting the data and training on xtrain, ytrain and predicting on xvalid, yvalid. using the transformed data for modeling
#Return values from this function are mean_accuracy_Score dictionary and list of models

mean_dict, model_list = ensCount.fit_predict_train_valid(train_tran_count, test_tran_count, model_dict = model_dict)
#modeling on the whole training data and predicting on the unseen test data. Also, creating submission file.
#Return values from this function are list of fpr, tpr and auc scores

fpr_count_list, tpr_count_list, auc_count_list = ensCount.final_train_predict(train_tran_count, training['target'], test_tran_count, model_list,
                                                                  mean_dict, sample)
#Here I am implementing Tfidf Vectorizer

vect = "TfidfVectorizer"

#initializing model dictionary. these are the models which we will use in our class for training and prediciton
model_dict = {"Logistic Regression": LogisticRegression(C = 1.0, max_iter = 10000), 
              "Random Forest Classifier": RandomForestClassifier(n_estimators = 600),
              "XGBoost Classifier": xgb.XGBClassifier(n_estimators =  400, learning_rate =0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                 nthread=4, scale_pos_weight=1)}

#initializing a class object 
ensTfidf = Ensembler(model_dict = model_dict, num_folds = 5, optimize=accuracy_score, vectorizer_type =vect)
#this is a preprocessing step. using original train and test data to preprocess.

training, testing, train_tran_tfidf, test_tran_tfidf =  ensTfidf.preprocess(train, test)
#using the fit_predict_train_valid function from class, splitting the data into train and validation and modelling on those

mean_dict, model_list = ensTfidf.fit_predict_train_valid(train_tran_tfidf, test_tran_tfidf, model_dict = model_dict)
#here, I am training on the whole train transformed data and predicting on the unseen data. 

fpr_tfidf_list, tpr_tfidf_list, auc_tfidf_list = ensTfidf.final_train_predict(train_tran_tfidf, training['target'], test_tran_tfidf, model_list,
                                                                  mean_dict, sample)
# Here I am implementing the glove word embedding from stanford

vect = "WordEmbedding"

#againg initializing the model_dictionary that we will use in our class
model_dict = {"Logistic Regression": LogisticRegression(C = 1.0, max_iter = 10000), 
              "Random Forest Classifier": RandomForestClassifier(n_estimators = 600),
              "XGBoost Classifier": xgb.XGBClassifier(n_estimators =  400, learning_rate =0.1, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic',
                                 nthread=4, scale_pos_weight=1)}

#creating an object for our ensembler class
ensWord = Ensembler(model_dict = model_dict, num_folds = 5, optimize=accuracy_score, vectorizer_type =vect)
#preprocessing the data on our train and test set.

training, testing, train_tran, test_tran =  ensWord.preprocess(train, test)
#here i am splitting the data and creating predictive models on train and valid sets.

mean_dict, model_list = ensWord.fit_predict_train_valid(train_tran, test_tran, model_dict = model_dict)
#here, i am using the whole training dataset for training and predicting on the test data. fpr, tpr and auc are the returned values

fpr_word_list, tpr_word_list, auc_word_list = ensWord.final_train_predict(train_tran, training['target'], test_tran, model_list,
                                                                  mean_dict, sample)
# function for creating a confusion matrix outside the class

def confusionMatrix(data, Labels = ['No-Disaster', 'Disaster']):
            
    # setting the figure size
    plt.figure(figsize=(4,4), facecolor='white')
        
    # using seaborn heatmap to plot confusion matrix
    sns.heatmap(data, xticklabels= Labels, yticklabels= Labels, annot = True, fmt= 'g')
    
    # setting the title name
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class") #setting the x label
    plt.ylabel("True Class")   # setting the y label
    plt.show()
#converting the transformed train and test that we are getting from above class into numpy arrays
train_tran1 = np.array(train_tran)
test_tran1 = np.array(test_tran)

#doing one hot encoding on target values in training set
y_train_enc = np_utils.to_categorical(training['target'])
testing['target'] = ""
print(train_tran1.shape)
print(test_tran1.shape)
# creating a 3 layer simple nn 

#initializing 
model = Sequential()

#creating a dense layer with 300 input dimensions and giving 400 as output
model.add(Dense(400, input_dim = 300, activation = "relu"))
#adding dropout, this actually improved the accuracy a little bit
model.add(Dropout(0.60))
#this will normalize the input values
model.add(BatchNormalization())

#adding another layer
model.add(Dense(400, activation ='relu'))
#again adding a droput and batchnormalization
model.add(Dropout(0.8))
model.add(BatchNormalization())

#dding the output layer with softmax activation
model.add(Dense(2))
model.add(Activation('softmax'))


# compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#this will monitor the validity accuracy and will do an earlystop if accuracy doesnt increase much
earlystop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 3, 
                        verbose = 0, mode = 'auto')

#fitting the model on our transformed train data
model.fit(train_tran1, y=y_train_enc,validation_split=0.3, batch_size = 128, epochs = 30, verbose=1, callbacks = [earlystop])
# list all data in history
print(model.history.history.keys())
# summarize history for accuracy
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# predict probabilities for test set
y_pred_proba = model.predict(test_tran1, verbose=0)

# predicting classes for test set
y_pred = model.predict_classes(test_tran1, verbose=0)

#predicting classes for our train dataset
ytrain_pred = model.predict_classes(train_tran1)

#confusion matrix for our simple keras model on training set
con_mat = confusion_matrix(training['target'], ytrain_pred)

#creating a submission file for our model 
sample1 =pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sample1['target'] = y_pred
sample1.to_csv("SimpleKerasModel.csv", index = False)

#also creating a classification report for this model
print("Classification report for training: ")
print(classification_report(training['target'], ytrain_pred))
print("\n Confusion Matrix for training is: ")
confusionMatrix(con_mat)
# lets use keras tokenizer this time
token1 = text.Tokenizer(num_words = None)
max_len = 70 

#fitting the tokenizer on training dataset
token1.fit_on_texts(list(training['text']))

#converting the text to sequence of integers on both training and test data
train_seq = token1.texts_to_sequences(training['text'])
test_seq = token1.texts_to_sequences(testing['text'])

# zero pad the sequences 
train_pad = sequence.pad_sequences(train_seq, maxlen = max_len)
test_pad = sequence.pad_sequences(test_seq, maxlen = max_len)

#creating a word index
word_index1 = token1.word_index
#creating a glove word embedding index(dictionary)
embeddings_index = {}
f = open('/kaggle/input/glove42b300dtxt/glove.42B.300d.txt', encoding = 'utf-8')
for line in (f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
    
f.close()
        
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index1)+1, 300))

#taking the word from the word index that we created using keras tokenizer
for word, i in tqdm(word_index1.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #storing it in a dictionary with a value of embedding index of that word
        embedding_matrix[i] = embedding_vector
# Creating a LSTM model with glove embeddings
model = Sequential()

#adding an embedding layer with an input dimension of 300 and with wights as the coefficients of golve word index
model.add(Embedding(len(word_index1)+1, 300, weights = [embedding_matrix],
        input_length = max_len, trainable = False))

#here using spatial dropout. Removing specifing part of element from all channels
model.add(SpatialDropout1D(0.3))
#adding LSTM layer
model.add(LSTM(300, dropout = 0.3, recurrent_dropout = 0.3))

#adding another Dense layer 
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.8))
#adding another dense layer
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.8))

#and then the output layer
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss= 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#this will monitor the validity accuracy and will do an earlystop if accuracy doesnt increase much
earlystop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 3, 
                        verbose = 0, mode = 'auto')

# Fitting the model with early stopping callback
model.fit(train_pad, y = y_train_enc, batch_size = 32, epochs = 100,
    verbose = 1, validation_split = 0.2, callbacks = [earlystop])
model.history.history.keys()
# list all data in history
print(model.history.history.keys())
# summarize history for accuracy
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#predicting probabilities for test set
y_pred_proba = model.predict(test_pad, verbose=0)

# predicting classes for test set
y_pred = model.predict_classes(test_pad, verbose=0)

#creating a submission file
sample1['target'] = y_pred
sample1.to_csv("Lstm_keras.csv", index = False)

#predicting the classes of training set
ytrain_pred = model.predict_classes(train_pad)

#confusion matrix for train set
con_mat = confusion_matrix(training['target'], ytrain_pred)

#also printing classification report for this model
print("Classification report for training: ")
print(classification_report(training['target'], ytrain_pred))
print("\n Confusion Matrix for training is: ")
confusionMatrix(con_mat)
#calculating the fpr, tpr and auc for plotting the roc curve
fpr_keras, tpr_keras, thresholds_keras = roc_curve(training['target'], ytrain_pred)
auc_keras = auc(fpr_keras, tpr_keras)
#!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
#importing a tokenization file from github. actually writing it on my local machine.

import urllib.request
url = 'https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py'
filename = 'myfile.py'
urllib.request.urlretrieve(url, filename)
#importing the require libraries

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import myfile
#creating a bert encoding function 
def bert_encode(texts, tokenizer, max_len=512):
    #creating a list for tokens, masks and segments of text
    all_tokens = []
    all_masks = []
    all_segments = []
    
    #for loop to go through the text
    for text in texts:
        text = tokenizer.tokenize(text)   #this will tokanize the text 
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"] #adding cls and sep token to the beginning and end to the text
        pad_len = max_len - len(input_sequence)       # deciding the padding lenghth
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)   #creating tokens
        tokens += [0] * pad_len                           #adding a padding to match the length of tokens
        pad_masks = [1] * len(input_sequence) + [0] * pad_len      #creating a padding for masked tokens
        segment_ids = [0] * max_len                          
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

#creating a function to build a model
def build_model(bert_layer, max_len=512):
    #taking the input, mask and segment and creating ids
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    #creating a sequence of output and then building a model
    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    #initializing and compiling 
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
#downloading bert layer
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
#creating a vocab file from bert layer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = myfile.FullTokenizer(vocab_file, do_lower_case)
testing['text'].shape
#encoding the text using bert encode function
train_input = bert_encode(training.text.values, tokenizer, max_len=160)
#encoding the test data using bert encode function
test_input = bert_encode(testing.text.values, tokenizer, max_len=160)
#assigning target labels with target values
train_labels = train.target.values
len(train_input)
#building a model
model = build_model(bert_layer, max_len = 160)
model.summary()
#training model with bert layer
train_history = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=16
)

model.save('model.h5')
#test predicition on unseen data
test_pred = model.predict(test_input)

#creating a submission file
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission['target'] = test_pred.round().astype(int)
submission.to_csv('submission_bert.csv', index=False)
#predicting the classes of training set
ytrain_pred = model.predict(train_input)

ytrain_pred = [1 if(i>0.5) else 0 for i in ytrain_pred]

#confusion matrix for train set
con_mat = confusion_matrix(train_labels, np.array(ytrain_pred))

#also printing classification report for this model
print("Accuracy of training: {}".format(accuracy_score(train_labels, ytrain_pred)))
print("Classification report for training: ")
print(classification_report(train_labels, ytrain_pred))
print("\n Confusion Matrix for training is: ")
confusionMatrix(con_mat)
#calculating the fpr, tpr and auc for plotting the roc curve
fpr_bert, tpr_bert, thresholds_bert = roc_curve(train_labels, ytrain_pred)
auc_bert = auc(fpr_bert, tpr_bert)
auc_count_list
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_count_list[0], tpr_count_list[0], label='CountVectorizer (area = {:.3f})'.format(auc_count_list[0]))
plt.plot(fpr_tfidf_list[0], tpr_tfidf_list[0], label='TfidfVectorizer (area = {:.3f})'.format(auc_tfidf_list[0]))
plt.plot(fpr_word_list[0], tpr_word_list[0], label='WordEmbedding (area = {:.3f})'.format(auc_word_list[0]))
plt.plot(fpr_word_list[1], tpr_word_list[1], label='StackedClassifier (area = {:.3f})'.format(auc_word_list[1]))
plt.plot(fpr_bert, tpr_bert, label='Bert Algorithm (area = {:.3f})'.format(auc_bert))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
