import numpy as np
import pandas as pd
import re
import spacy
from tqdm import tqdm
import string
import ast
from collections import Counter
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")
len(nlp.Defaults.stop_words) #No of Stop Words in Spacy Package
data_dir = '../input/'
train = pd.read_csv(data_dir + "train.csv")
overview = pd.read_csv(data_dir + "game_overview.csv")
test = pd.read_csv(data_dir + "test.csv")
train.head()
overview.head()
overview_games = set(overview.title.unique())
test_games = set(test.title.unique())
train_games = set(train.title.unique())
print("Games Titles present in the train but not in overview", len(train_games - overview_games))
print("Games Titles present in the test but not in overview", len(test_games - overview_games))
train = pd.merge(train,overview,left_on='title',right_on='title')
train.head()
test = pd.merge(test,overview,left_on='title',right_on='title')
test.head()
isascii = lambda s: len(s) == len(s.encode()) #Lambda function that compares length
non_ascii_idx = [idx for idx,user_review in enumerate(train['user_review'].values) if not isascii(user_review)]
train.iloc[non_ascii_idx] #Heart Emojis are visible hence these strings have non ascii chars
import unicodedata
def remove_accented_chars(text):
    """Takes string as input and return a normalized version of string with ascii chars only.""" 
    """Non-ascii replaced with equivalent ascii."""
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8', 'ignore')
    return text
#Let's covert an example
train.loc[non_ascii_idx[0],'user_review']
remove_accented_chars(train.loc[non_ascii_idx[0],'user_review'])
CONTRACTION_MAP = {
"ain't": "is not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because",
"could've": "could have","couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not",
"don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not",
"he'd": "he would","he'd've": "he would have","he'll": "he will","he'll've": "he he will have","he's": "he is",
"how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would","I'd've": "I would have",
"I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","i'd": "i would","i'd've": "i would have",
"i'll": "i will","i'll've": "i will have","i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would",
"it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us",
"ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have",
"must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not",
"needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have",
"shan't": "shall not","sha'n't": "shall not","shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
"she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have","so's": "so as","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there would",
"there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have",
"they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
"wasn't": "was not","we'd": "we would","we'd've": "we would have","we'll": "we will","we'll've": "we will have",
"we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have",
"what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have",
"where'd": "where did","where's": "where is","where've": "where have","who'll": "who will",
"who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is","why've": "why have",
"will've": "will have","won't": "will not","won't've": "will not have","would've": "would have","wouldn't": "would not",
"wouldn't've": "would not have","y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have",
"y'all're": "you all are","y'all've": "you all have","you'd": "you would","you'd've": "you would have",
"you'll": "you will","you'll've": "you will have","you're": "you are","you've": "you have"}
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
expand_contractions("This can't be true.")
remove_url = lambda text : re.sub("http[s]*://[^\s]+"," ",text)
remove_url("Hey!! Checkout the site here https://www.google.com or https://youtube.com")
def cleaning_spaces_punct_specialchars(text):
    text_sent = re.sub("[-!\"#$%&'()*+,./:;<=>?@\][^_`|}{~']"," ",text) #replaces any of these characters with space
    text_sent = text_sent.replace("\\"," ")
    text_sent = re.sub(r'\s+', ' ',text_sent) #Replaces more than one space with single space
    return text_sent
cleaning_spaces_punct_specialchars("Hi! Chir@ag Ho    w are \\you?")
import spacy
nlp = spacy.load("en_core_web_lg")
sents = ['This is to play the games.', 'This is playing the games.', 'This is where we played the games.']
docs = nlp.pipe(sents) #use nlp.pipe("sentence") for only single sentence
for doc in docs:
    for token in doc:
        print(token.lemma_, end=" ")
    print("")
def clean_the_corpus(dataframe,col_name):
    corpus = list(dataframe[col_name].values)
    text_sent = [expand_contractions(sent) for sent in corpus]
    text_sent = [re.sub("http[s]*://[^\s]+"," ",text) for text in text_sent]
    text_sent = [remove_accented_chars(sent) for sent in text_sent]
    text_sent = [re.sub("[-!\"#$%&'()*+,./:;<=>?@\][^_`|}{~']"," ",text) for text in text_sent]
    text_sent = [text.replace("\\"," ") for text in text_sent]
    text_sent = [re.sub(r'\s+', ' ',sent) for sent in text_sent]
    text_sent = list(map(str.lower,text_sent))
    docs = nlp.pipe(text_sent,disable=["ner","parser"])
    cleaned_corpus = []
    for doc in docs:
        doc_text = []
        for token in doc:
            if token.lemma_ != '-PRON-':
                doc_text.append(token.lemma_)
            else:
                doc_text.append(token.text)
        cleaned_corpus.append(doc_text)
    cleaned_corpus = [" ".join(cleaned_text) for cleaned_text in cleaned_corpus]
    return cleaned_corpus
cleaned_title = clean_the_corpus(train,"title")
cleaned_user_review = clean_the_corpus(train, "user_review")
cleaned_overview = clean_the_corpus(train,"overview")
list(set(cleaned_title))[:5]
train['user_review'].values[0]
cleaned_user_review[0]
train['tags'].values[0] #Double Quotes in the beginning and end show it's a string
print(type(train['tags'].values[0]))
def tag_cleaner(df,col_name):
    tags = df[col_name].values #list of all strings(tags) assosiated with each row.
    tags = [ast.literal_eval(tag) for tag in tags] #Evaluates the strings into list of string for each row
    tags = ["_".join(set(tag)) for tag in tags] #If we join by space then the tags like horror and psychological horror
    #won't have any differences.
    return np.array(tags) 
cleaned_tag = tag_cleaner(train,"tags")
cleaned_tag[0]
train_cleaned = pd.DataFrame()
train_cleaned['title'] = cleaned_title
train_cleaned['user_review'] = cleaned_user_review
train_cleaned['overview'] = cleaned_overview
train_cleaned['tag'] = cleaned_tag
train_cleaned['year'] = train['year'].astype(str)
train_cleaned['user_suggestion'] = train['user_suggestion'].values
train_cleaned.head()
test_cleaned_title = clean_the_corpus(test,"title")
test_cleaned_user_review = clean_the_corpus(test, "user_review")
test_overview = clean_the_corpus(test, "overview")
test_cleaned_tag = tag_cleaner(test,"tags")
test_review_id = test['review_id']
test_cleaned = pd.DataFrame()
test_cleaned['review_id'] = test_review_id
test_cleaned['title'] = test_cleaned_title
test_cleaned['user_review'] = test_cleaned_user_review
test_cleaned['overview'] = test_overview
test_cleaned['tag'] = test_cleaned_tag
test_cleaned.head()
