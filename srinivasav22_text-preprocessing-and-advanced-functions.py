import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import emoji

#Count vectorizer for N grams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Nltk for tekenize and stopwords
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

#Loading data
df=pd.read_csv('../input/nlp-getting-started/train.csv')
df.head()
def missing_value_of_data(data):
    total=data.isnull().sum().sort_values(ascending=False)
    percentage=round(total/data.shape[0]*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

missing_value_of_data(df)
#  Reason for 0 percentage value = Round of 1 divided by 27481 will be 0
df=df.dropna()
def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
count_values_in_column(df,'target')
def unique_values_in_column(data,feature):
    unique_val=pd.Series(data.loc[:,feature].unique())
    return pd.concat([unique_val],axis=1,keys=['Unique Values'])

unique_values_in_column(df,'target')
    
def duplicated_values_data(data):
    dup=[]
    columns=data.columns
    for i in data.columns:
        dup.append(sum(data[i].duplicated()))
    return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])
duplicated_values_data(df)
df.describe()
def find_url(string): 
    text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
    return "".join(text) # converting return value from list to string
sentence="I love spending time at https://www.kaggle.com/"
find_url(sentence)
df['url']=df['text'].apply(lambda x:find_url(x))
def find_emoji(text):
    emo_text=emoji.demojize(text)
    line=re.findall(r'\:(.*?)\:',emo_text)
    return line
sentence="I love ⚽ very much 😁"
find_emoji(sentence)

# Emoji cheat sheet - https://www.webfx.com/tools/emoji-cheat-sheet/
# Uniceode for all emoji : https://unicode.org/emoji/charts/full-emoji-list.html
df['emoji']=df['text'].apply(lambda x: find_emoji(x))
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


sentence="Its all about \U0001F600 face"
print(sentence)
remove_emoji(sentence)
df['text']=df['text'].apply(lambda x: remove_emoji(x))
def find_email(text):
    line = re.findall(r'[\w\.-]+@[\w\.-]+',str(text))
    return ",".join(line)
sentence="My gmail is abc99@gmail.com"
find_email(sentence)
df['email']=df['text'].apply(lambda x: find_email(x))
def find_hash(text):
    line=re.findall(r'(?<=#)\w+',text)
    return " ".join(line)
sentence="#Corona is trending now in the world" 
find_hash(sentence)
df['hash']=df['text'].apply(lambda x: find_hash(x))
def find_at(text):
    line=re.findall(r'(?<=@)\w+',text)
    return " ".join(line)
sentence="@David,can you help me out"
find_at(sentence)
df['at_mention']=df['text'].apply(lambda x: find_at(x))
def find_number(text):
    line=re.findall(r'[0-9]+',text)
    return " ".join(line)
sentence="2833047 people are affected by corona now"
find_number(sentence)
df['number']=df['text'].apply(lambda x: find_number(x))
def find_phone_number(text):
    line=re.findall(r"\b\d{10}\b",text)
    return "".join(line)
find_phone_number("9998887776 is a phone number of Mark from 210,North Avenue")
df['phone_number']=df['text'].apply(lambda x: find_phone_number(x))
def find_year(text):
    line=re.findall(r"\b(19[40][0-9]|20[0-1][0-9]|2020)\b",text)
    return line
sentence="India got independence on 1947."
find_year(sentence)
df['year']=df['text'].apply(lambda x: find_year(x))
def find_nonalp(text):
    line = re.findall("[^A-Za-z0-9 ]",text)
    return line
sentence="Twitter has lots of @ and # in posts.(general tweet)"
find_nonalp(sentence)
df['non_alp']=df['text'].apply(lambda x: find_nonalp(x))
def find_punct(text):
    line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
    string="".join(line)
    return list(string)
example="Corona virus have kiled #24506 confirmed cases now.#Corona is un(tolerable)"
print(find_punct(example))
df['punctuation']=df['text'].apply(lambda x : find_punct(x))
def stop_word_fn(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text) 
    non_stop_words = [w for w in word_tokens if not w in stop_words] 
    stop_words= [w for w in word_tokens if w in stop_words] 
    return stop_words
example_sent = "This is a sample sentence, showing off the stop words filtration."
stop_word_fn(example_sent)
df['stop_words']=df['text'].apply(lambda x : stop_word_fn(x))
def ngrams_top(corpus,ngram_range,n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    total_list=words_freq[:n]
    df=pd.DataFrame(total_list,columns=['text','count'])
    return df
ngrams_top(df['text'],(1,1),n=10)
ngrams_top(df['text'],(2,2),n=10)
ngrams_top(df['text'],(3,3),n=10)
def rep(text):
    grp = text.group(0)
    if len(grp) > 1:
        return grp[0:1] # can change the value here on repetition
def unique_char(rep,sentence):
    convert = re.sub(r'(\w)\1+', rep, sentence) 
    return convert
sentence="heyyy this is loong textttt sooon"
unique_char(rep,sentence)
df['unique_char']=df['text'].apply(lambda x : unique_char(rep,x))
def find_dollar(text):
    line=re.findall(r'\$\d+(?:\.\d+)?',text)
    return " ".join(line)

# \$ - dollar sign followed by
# \d+ one or more digits
# (?:\.\d+)? - decimal which is optional
sentence="this shirt costs $20.56"
find_dollar(sentence)
df['dollar']=df['text'].apply(lambda x : find_dollar(x))
#Number greater than 930
def num_great(text): 
    line=re.findall(r'9[3-9][0-9]|[1-9]\d{3,}',text)
    return " ".join(line)
sentence="It is expected to be more than 935 corona death and 29974 observation cases across 29 states in india"
num_great(sentence)
#Number greater than 930 (Just part of example)
df['num_great']=df['text'].apply(lambda x : num_great(x))
# Number less than 930
def num_less(text):
    only_num=[]
    for i in text.split():
        line=re.findall(r'^(9[0-2][0-0]|[1-8][0-9][0-9]|[1-9][0-9]|[0-9])$',i) # 5 500
        only_num.append(line)
        all_num=[",".join(x) for x in only_num if x != []]
    return " ".join(all_num)
sentence="There are some countries where less than 920 cases exist with 1100 observations"
num_less(sentence)
#Number greater than 930 (Just part of example)
df['num_less']=df['text'].apply(lambda x : num_less(x))
def or_cond(text,key1,key2):
    line=re.findall(r"{}|{}".format(key1,key2), text) 
    return " ".join(line)
sentence="sad and sorrow displays emotions"
or_cond(sentence,'sad','sorrow')
# Looks for sorrow or sad word
df['sad_or_sorrow']=df['text'].apply(lambda x : or_cond(x,'sad','sorrow'))
def and_cond(text):
    line=re.findall(r'(?=.*do)(?=.*die).*', text) 
    return " ".join(line)
print("Both string present:",and_cond("do or die is a motivating phrase"))
print("Only one string present :",and_cond('die word is other side of emotion'))
# Looks for do and die both else empty
df['do_and_die']=df['text'].apply(lambda x : and_cond(x))
# mm-dd-yyyy format 
def find_dates(text):
    line=re.findall(r'\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b',text)
    return line

sentence="Todays date is 04/28/2020 for format mm/dd/yyyy, not 28/04/2020"
find_dates(sentence)
df['dates']=df['text'].apply(lambda x : find_dates(x))
def only_words(text):
    line=re.findall(r'\b[^\d\W]+\b', text)
    return " ".join(line)

sentence="the world population has grown from 1650 million to 6000 million"
only_words(sentence)
df['only_words']=df['text'].apply(lambda x : only_words(x))
def only_numbers(text):
    line=re.findall(r'\b\d+\b', text)
    return " ".join(line)
sentence="the world population has grown from 1650 million to 6000 million"
only_numbers(sentence)
df['only_num']=df['text'].apply(lambda x : only_numbers(x))
# Extracting word with boundary
def boundary(text):
    line=re.findall(r'\bneutral\b', text)
    return " ".join(line)
sentence="Most tweets are neutral in twitter"
boundary(sentence)
df['bound']=df['text'].apply(lambda x : boundary(x))
def search_string(text,key):
    return bool(re.search(r''+key+'', text))
sentence="Happy Mothers day to all Moms"
search_string(sentence,'day')
df['search_day']=df['text'].apply(lambda x : search_string(x,'day'))
def pick_only_key_sentence(text,keyword):
    line=re.findall(r'([^.]*'+keyword+'[^.]*)', text)
    return line
sentence="People are fighting with covid these days.Economy has fallen down.How will we survice covid"
pick_only_key_sentence(sentence,'covid')
df['pick_senence']=df['text'].apply(lambda x : pick_only_key_sentence(x,'covid'))
def pick_unique_sentence(text):
    line=re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', text)
    return line
sentence="I thank doctors\nDoctors are working very hard in this pandemic situation\nI thank doctors"
pick_unique_sentence(sentence)
df['pick_unique']=df['text'].apply(lambda x : pick_unique_sentence(x))
def find_capital(text):
    line=re.findall(r'\b[A-Z]\w+', text)
    return line
sentence="World is affected by corona crisis.No one other than God can save us from it"
find_capital(sentence)
df['caps_word']=df['text'].apply(lambda x : find_capital(x))
df['text_length']=df['text'].str.split().map(lambda x: len(x))
df[['text','text_length']].sample(3)
df['char_length']=df['text'].str.len()
df[['text','char_length']].sample(3)
def find_id(text):
    line=re.findall(r'\bIND(\d+)', text)
    return line
sentence="My company id is IND50120.And I work under Asia region"
find_id(sentence)
df['get_id']=df['text'].apply(lambda x : find_id(x))
my_string_rows = df[df['text'].str.contains("good")]
my_string_rows[['text']].sample(3)
!pip install webcolors
import webcolors
def find_color(string): 
    text = re.findall('\#(?:[0-9a-fA-F]{3}){1,2}',string)
    conv_name=[]
    for i in text:
        conv_name.append(webcolors.hex_to_name(i))
    return conv_name
sentence="Find the color of #00FF00 and #FF4500"
find_color(sentence)
def remove_tag(string):
    text=re.sub('<.*?>','',string)
    return text
sentence="Markdown sentences can use <br> for breaks and <i></i> for italics"
remove_tag(sentence)
def ip_add(string):
    text=re.findall('\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',string)
    return text
sentence="An example of ip address is 125.16.100.1"
ip_add(sentence)
def mac_add(string):
    text=re.findall('(?:[0-9a-fA-F]:?){12}',string)
    return text
#https://stackoverflow.com/questions/26891833/python-regex-extract-mac-addresses-from-string/26892371
sentence="MAC ADDRESSES of this laptop - 00:24:17:b1:cc:cc .Other details will be mentioned"
mac_add(sentence)
def subword(string,sub): 
    text=re.findall(sub,string)
    return len(text)
sentence = 'Fundamentalism and constructivism are important skills'
subword(sentence,'ism') # change subword and try for others
def lat_lon(string):
    text=re.findall(r'^[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)$',string)
    if text!=[]:
        print("[{}] is valid latitude & longitude".format(string))
    else:
        print("[{}] is not a valid latitude & longitude".format(string))
lat_lon('28.6466772,76.8130649')
lat_lon('2324.3244,3423.432423')
def valid_pan(string):
    text=re.findall(r'^([A-Z]){5}([0-9]){4}([A-Z]){1}$',string)
    if text!=[]:
        print("{} is valid PAN number".format(string))
    else:
        print("{} is not a valid PAN number".format(string))
valid_pan("ABCSD0123K")
valid_pan("LEcGD012eg")
def valid_phone_code(string):
    text=re.findall(r'^([0-9]){2}(-)([0-9]){2}(-)(\d+)$',string)
    if text!=[]:
        print("{} is valid Indian Phone number wth country code".format(string))
    else:
        print("{} is not a valid Indian Phone number wth country code".format(string))
valid_phone_code('91-44-23413627')
valid_phone_code('291-4456-23413627')
def pos_look_ahead(string,A,B):
    pattern = re.compile(''+A+'(?=\s'+B+')')
    match = pattern.search(string)
    print("position:{} Matched word:{}".format(match.span(),match.group()))
pos_look_ahead("I love kaggle. I love DL","love","DL")
def neg_look_ahead(string,A,B):
    pattern = re.compile(''+A+'(?!\s'+B+')')
    match = pattern.search(string)
    print("position:{} Matched word:{}".format(match.span(),match.group()))
neg_look_ahead("I love kaggle. I love DL","love","DL")
def pos_look_behind(string,A,B):
    pattern = re.compile("(?<="+A+"\s)"+B+"")
    match = pattern.search(string)
    print("position:{} Matched word: {}".format(match.span(),match.group()))
pos_look_behind("i love nlp.everyone likes nlp","love","nlp")
# the word "nlp" that do come after "love"
def neg_look_behind(string,A,B):
    pattern = re.compile("(?<!"+A+"\s)"+B+"")
    match = pattern.search(string)
    print("position:{} Matched word: {}".format(match.span(),match.group()))
neg_look_behind("i love nlp.everyone likes nlp","love","nlp")
# the word "nlp" that doesnt come after "love"
def find_domain(string): 
    text = re.findall(r'\b(\w+[.]\w+)',string)
    return text
sentence="WHO provides valid information about covid in their site who.int . UNICEF supports disadvantageous childrens. know more in unicef.org"
find_domain(sentence)
def find_percent(string): 
    text = re.findall(r'\b(100|[1-9][0-9]|[0-9])\%',string)
    return text
sentence="COVID recovery rate has been increased to 76%.And death rate drops to 2% from 3%"
find_percent(sentence)
df.sample(5)
# We will see empty values too as most of text may not have related feature.You can filter and check.