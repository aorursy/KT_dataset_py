# import library 

import gc

import re

import operator 

import numpy as np

import pandas as pd

import seaborn as sns

from gensim.models import KeyedVectors

from sklearn import model_selection

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, Input, Dense, CuDNNLSTM, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.layers import add, Dropout

from keras.optimizers import RMSprop, Adam

from keras.models import Model

from keras.callbacks import EarlyStopping, LearningRateScheduler

from keras.preprocessing import text, sequence

from keras import callbacks

from sklearn.model_selection import train_test_split
# read data 

train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')



# set x and y 

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'



# IDENTITY_COLUMNS (신원을 파악할 수 있는 변수)

IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness']



# AUX_COLUMNS (additional toxicity subtype attributes) 

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']



# pre-trained embedding models

EMBEDDING_FILES = [

    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',

    '../input/glove840b300dtxt/glove.840B.300d.txt'

]
train_df = train_df.dropna()

len(train_df)
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words



ft_common_crawl = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'

embeddings_index = KeyedVectors.load_word2vec_format(ft_common_crawl)
vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index) # out of vocab (unknown)

print(oov[:10])
contraction_mapping1 = {

  "Trump's" : 'trump is',"'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

  'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

  'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',

  'can;t': 'cannot','can;t;ve': 'cannot have',

  'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

  "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

  'couldn;t;ve': 'could not have','couldn´t': 'could not',

  'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',

  'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

  'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',

  'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',

  "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

  'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',

  "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',

  "he'd've": 'he would have',"he'll": 'he will',

  "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',

  'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',

  'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',

  "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',

  'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',

  'how’s': 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will',

  'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not',

  'isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',

  "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',

  'it’d': 'it would','it’ll': 'it will','it’s': 'it is',

  'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',

  'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',

  'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',

  'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',

  'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

  "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',

  'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',

  'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',

  'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',

  "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',

  'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',

  'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',

  "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',

  'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

  'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',

  "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',

  'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',

  "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',

  'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',

  'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',

  'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',

  'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',

  'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

  "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',

  'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',

  "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',

  'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',

  'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is',

  'what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',

  'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',

  "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',

  'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

  'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',

  'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',

  'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

  'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',

  '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',

  "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",

  "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",

  "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",

  "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",

  "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",

  "i'ma": "i am","is'nt": "is not","‘I":'I',

  'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\

  'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\

  'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\

  'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\

  'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\

  'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\

  'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',

  'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop',

}
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

contraction_mapping.update(contraction_mapping1)

print("number of contraction",len(contraction_mapping))



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))



vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index)

print(oov[:10])
train_df['comment_text'][:10]
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])    

    for p in punct:

        text = text.replace(p, f' {p} ')     

    return text



punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"_":" ", "`":" "}



train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))



vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index)

print(oov[:10])
train_df['comment_text'][:10]
swear_words = [

    ' 4r5e ',' 5h1t ',' 5hit ',' a55 ',' anal ',' anus ',' ar5e ',' arrse ',' arse ',' ass ',' ass-fucker ',' asses ',' assfucker ',' assfukka ',' asshole ',' assholes ',' asswhole ',' a_s_s ',' b!tch ',' b00bs ',' b17ch ',' b1tch ',' ballbag ',' balls ',' ballsack ',' bastard ',' beastial ',' beastiality ',' bellend ',' bestial ',' bestiality ',' biatch ',' bitch ',' bitcher ',' bitchers ',' bitches ',' bitchin ',' bitching ',' bloody ',' blow job ',' blowjob ',' blowjobs ',' boiolas ',' bollock ',' bollok ',' boner ',' boob ',' boobs ',' booobs ',' boooobs ',' booooobs ',' booooooobs ',' breasts ',' buceta ',' bugger ',' bum ',' bunny fucker ',' butt ',' butthole ',' buttmuch ',' buttplug ',' c0ck ',' c0cksucker ',' carpet muncher ',' cawk ',' chink ',' cipa ',' cl1t ',' clit ',' clitoris ',' clits ',' cnut ',' cock ',' cock-sucker ',' cockface ',' cockhead ',' cockmunch ',' cockmuncher ',' cocks ',' cocksuck ',' cocksucked ',' cocksucker ',' cocksucking ',' cocksucks ',' cocksuka ',' cocksukka ',' cok ',' cokmuncher ',' coksucka ',' coon ',' cox ',' crap ',' cum ',' cummer ',' cumming ',' cums ',' cumshot ',' cunilingus ',' cunillingus ',' cunnilingus ',' cunt ',' cuntlick ',' cuntlicker ',' cuntlicking ',' cunts ',' cyalis ',' cyberfuc ',' cyberfuck ',' cyberfucked ',' cyberfucker ',' cyberfuckers ',' cyberfucking ',' d1ck ',' damn ',' dick ',' dickhead ',' dildo ',' dildos ',' dink ',' dinks ',' dirsa ',' dlck ',' dog-fucker ',' doggin ',' dogging ',' donkeyribber ',' doosh ',' duche ',' dyke ',' ejaculate ',' ejaculated ',' ejaculates ',' ejaculating ',' ejaculatings ',' ejaculation ',' ejakulate ',' f u c k ',' f u c k e r ',' f4nny ',' fag ',' fagging ',' faggitt ',' faggot ',' faggs ',' fagot ',' fagots ',' fags ',' fanny ',' fannyflaps ',' fannyfucker ',' fanyy ',' fatass ',' fcuk ',' fcuker ',' fcuking ',' feck ',' fecker ',' felching ',' fellate ',' fellatio ',' fingerfuck ',' fingerfucked ',' fingerfucker ',' fingerfuckers ',' fingerfucking ',' fingerfucks ',' fistfuck ',' fistfucked ',' fistfucker ',' fistfuckers ',' fistfucking ',' fistfuckings ',' fistfucks ',' flange ',' fook ',' fooker ',' fuck ',' fucka ',' fucked ',' fucker ',' fuckers ',' fuckhead ',' fuckheads ',' fuckin ',' fucking ',' fuckings ',' fuckingshitmotherfucker ',' fuckme ',' fucks ',' fuckwhit ',' fuckwit ',' fudge packer ',' fudgepacker ',' fuk ',' fuker ',' fukker ',' fukkin ',' fuks ',' fukwhit ',' fukwit ',' fux ',' fux0r ',' f_u_c_k ',' gangbang ',' gangbanged ',' gangbangs ',' gaylord ',' gaysex ',' goatse ',' God ',' god-dam ',' god-damned ',' goddamn ',' goddamned ',' hardcoresex ',' hell ',' heshe ',' hoar ',' hoare ',' hoer ',' homo ',' hore ',' horniest ',' horny ',' hotsex ',' jack-off ',' jackoff ',' jap ',' jerk-off ',' jism ',' jiz ',' jizm ',' jizz ',' kawk ',' knob ',' knobead ',' knobed ',' knobend ',' knobhead ',' knobjocky ',' knobjokey ',' kock ',' kondum ',' kondums ',' kum ',' kummer ',' kumming ',' kums ',' kunilingus ',' l3itch ',' labia ',' lmfao ',' lust ',' lusting ',' m0f0 ',' m0fo ',' m45terbate ',' ma5terb8 ',' ma5terbate ',' masochist ',' master-bate ',' masterb8 ',' masterbat3 ',' masterbate ',' masterbation ',' masterbations ',' masturbate ',' mo-fo ',' mof0 ',' mofo ',' mothafuck ',' mothafucka ',' mothafuckas ',' mothafuckaz ',' mothafucked ',' mothafucker ',' mothafuckers ',' mothafuckin ',' mothafucking ',' mothafuckings ',' mothafucks ',' mother fucker ',' motherfuck ',' motherfucked ',' motherfucker ',' motherfuckers ',' motherfuckin ',' motherfucking ',' motherfuckings ',' motherfuckka ',' motherfucks ',' muff ',' mutha ',' muthafecker ',' muthafuckker ',' muther ',' mutherfucker ',' n1gga ',' n1gger ',' nazi ',' nigg3r ',' nigg4h ',' nigga ',' niggah ',' niggas ',' niggaz ',' nigger ',' niggers ',' nob ',' nob jokey ',' nobhead ',' nobjocky ',' nobjokey ',' numbnuts ',' nutsack ',' orgasim ',' orgasims ',' orgasm ',' orgasms ',' p0rn ',' pawn ',' pecker ',' penis ',' penisfucker ',' phonesex ',' phuck ',' phuk ',' phuked ',' phuking ',' phukked ',' phukking ',' phuks ',' phuq ',' pigfucker ',' pimpis ',' piss ',' pissed ',' pisser ',' pissers ',' pisses ',' pissflaps ',' pissin ',' pissing ',' pissoff ',' poop ',' porn ',' porno ',' pornography ',' pornos ',' prick ',' pricks ',' pron ',' pube ',' pusse ',' pussi ',' pussies ',' pussy ',' pussys ',' rectum ',' retard ',' rimjaw ',' rimming ',' s hit ',' s.o.b. ',' sadist ',' schlong ',' screwing ',' scroat ',' scrote ',' scrotum ',' semen ',' sex ',' sh!t ',' sh1t ',' shag ',' shagger ',' shaggin ',' shagging ',' shemale ',' shit ',' shitdick ',' shite ',' shited ',' shitey ',' shitfuck ',' shitfull ',' shithead ',' shiting ',' shitings ',' shits ',' shitted ',' shitter ',' shitters ',' shitting ',' shittings ',' shitty ',' skank ',' slut ',' sluts ',' smegma ',' smut ',' snatch ',' son-of-a-bitch ',' spac ',' spunk ',' s_h_i_t ',' t1tt1e5 ',' t1tties ',' teets ',' teez ',' testical ',' testicle ',' tit ',' titfuck ',' tits ',' titt ',' tittie5 ',' tittiefucker ',' titties ',' tittyfuck ',' tittywank ',' titwank ',' tosser ',' turd ',' tw4t ',' twat ',' twathead ',' twatty ',' twunt ',' twunter ',' v14gra ',' v1gra ',' vagina ',' viagra ',' vulva ',' w00se ',' wang ',' wank ',' wanker ',' wanky ',' whoar ',

    ' whore ',' willies ',' willy ',' xrated ',' xxx '

]

replace_with_fuck = []



for swear in swear_words:

    if swear[1:(len(swear)-1)] not in embeddings_index:

        replace_with_fuck.append(swear)

        

replace_with_fuck = '|'.join(replace_with_fuck)



def handle_swears(text):

    text = re.sub(replace_with_fuck, ' fuck ', text)

    return text



train_df['comment_text'] = train_df['comment_text'].apply(lambda x: handle_swears(x))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: handle_swears(x))



vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index)

print(oov[:10])
train_df['comment_text'][:10]
mispell_dict1 = {'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump','utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',}

mispell_dict = {'whattsup': 'WhatsApp', 'whatasapp':'WhatsApp', 'whatsupp':'WhatsApp','whatcus':'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat':'what','Whwhat': 'What', 'whatshapp':'WhatsApp', 'howhat':'how that','Whybis':'Why is', 'laowhy86':'Foreigners who do not respect China','Whyco-education':'Why co-education',"Howddo":"How do", 'Howeber':'However', 'Showh':'Show',"Willowmagic":'Willow magic', 'WillsEye':'Will Eye', 'Williby':'will by','pretextt':'pre text','aɴᴅ':'and','amette':'annette','aᴛ':'at','Tridentinus':'mushroom','dailycaller':'daily caller', "™":'trade mark'}

mispell_dict.update(mispell_dict1)

print("number of mispell",len(mispell_dict))



def correct_spelling(x, dic):

    for word in dic.keys():

        if word in x:

            x = x.replace(word, dic[word])

    return x



train_df['comment_text'] = train_df['comment_text'].apply(lambda x: correct_spelling(x, mispell_dict))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: correct_spelling(x, mispell_dict))



vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index)

print(oov[:10])
train_df['comment_text'][:10]
def remove_space(text):

    """

    remove extra spaces and ending space if any

    """

    for space in spaces:

        text = text.replace(space, ' ')

    text = text.strip()

    text = re.sub('\s+', ' ', text)

    return text



spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad', '\xa0']



train_df['comment_text'] = train_df['comment_text'].apply(lambda x: remove_space(x))

test_df['comment_text'] = test_df['comment_text'].apply(lambda x: remove_space(x))



vocab = build_vocab(train_df['comment_text'])

oov = check_coverage(vocab, embeddings_index)

print(oov[:10])
train_df['comment_text'][:10]
# data split (train / test / x / y)

x_train = train_df[TEXT_COLUMN].astype(str)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str)
# ---

# token (text -> seq num)

# ---



# keras token func

# num_words: the maximum number of words to keep

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

tokenizer = Tokenizer(filters=CHARS_TO_REMOVE)



# fit_on_texts(=Updates internal vocabulary based on a list of texts)

tokenizer.fit_on_texts(list(x_train) + list(x_test))



# texts_to_sequences(=Transforms each text in texts to a sequence of integers)

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)



# pad_sequences(=Pads sequences to the same length)

MAX_LEN = 220 # 문장 최대 길이 설정 (set the number of X columns)

x_train = pad_sequences(x_train, maxlen=MAX_LEN)

x_test = pad_sequences(x_test, maxlen=MAX_LEN)
# ---

# embedding

# ---

  

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix



# glove + fasttext 

embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
# ---

# modeling

# ---



# bi-LSTM

def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model



# train options 

NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4



# sample_weights

for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)



sample_weights = np.ones(len(x_train), dtype=np.float32)

sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1) # rowsum

sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1) # target samples 

sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) # non-target samples

sample_weights /= sample_weights.mean() # scale by mean 



# train model 

checkpoint_predictions = []

weights = []

for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=2,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)

        

# predictions

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



# submission

submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)