import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import string
import re

import nltk
from nltk.corpus import wordnet,stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud, STOPWORDS
import matplotlib.cm as cm

import unicodedata
#from contractions import CONTRACTION_MAP
inp_df = pd.read_excel('../input/automatic-ticket-assignment-using-nlp/Automatic Ticket Assignment.xlsx')
inp_df.head()
inp_df.shape
#Since the caller column is not useful, we can drop it

inp_df.drop(columns='Caller',inplace=True)
inp_df.head()
#Check for Null and drop null rows - Since null count is less

inp_df.isnull().sum()
inp_df.dropna(inplace=True)
# Drop duplicate rows

inp_df[inp_df.duplicated(['Short description','Description'])]
inp_df.drop_duplicates(['Short description','Description'],inplace=True)
inp_df['Assignment group'].value_counts()
#Plot line graph and view counts of each tickets

plt.subplots(figsize = (20,5))

sns.countplot(x='Assignment group', data=inp_df,order = inp_df['Assignment group'].value_counts().index)
plt.xlabel('Assignment Group') 
plt.ylabel('Count') 
plt.xticks(rotation=90)
plt.title('Tickets Distribution')

plt.show()
# Lets skip GRP_0 and visualize the count

temp_df1 = inp_df[inp_df['Assignment group'] != 'GRP_0']

plt.subplots(figsize = (20,5))

sns.countplot(x='Assignment group', data=temp_df1,order = temp_df1['Assignment group'].value_counts().index)
plt.xlabel('Assignment Group') 
plt.ylabel('Count') 
plt.xticks(rotation=90)
plt.title('Tickets Distribution - Excluding GRP_0')

plt.show()
temp_df2 = pd.DataFrame(inp_df['Assignment group'].value_counts())
temp_df2 = temp_df2.T
temp_df2
inp_df['Count'] = inp_df.apply(lambda row: temp_df2[row['Assignment group']] , axis=1)

inp_df.loc[inp_df['Count'] <= 200 , "Assignment group"] = "GRP_X"
#Plot line graph and view counts of each tickets - After GRP_X

plt.subplots(figsize = (20,5))

sns.countplot(x='Assignment group', data=inp_df,order = inp_df['Assignment group'].value_counts().index)
plt.xlabel('Assignment Group') 
plt.ylabel('Count') 
plt.title('Tickets Distribution - With GRP_X')

plt.show()
inp_df['Assignment group'].value_counts()
inp_df.drop(columns='Count',inplace=True)
inp_df.head()
#Lets create temp csv to see the data as of now

#inp_df.to_csv('temp_data.csv')
#Lets merge both the columns into Single one

inp_df['Full Description'] = inp_df['Short description'] + ' '+ inp_df['Description']
inp_df.drop(columns=['Short description','Description'],inplace=True)
inp_df.head()
#Remove numbers
def removeNumbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text


#Replace Contractions

contraction_patterns = [ (r'won\'t', 'will not'),(r'didn\'t', 'did not'),(r'didnt', 'did not'), (r'can\'t', 'cannot'),(r'cant', 'cannot'), (r'i\'m', 'i am'), (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                         (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'), (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), (r'dont', 'do not'), (r'wont', 'will not') ]
def replaceContraction(text):
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        (text, count) = re.subn(pattern, repl, text)
    return text
'''
def replaceContraction(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
'''

#Replace Negations with Antonym
def replace(word, pos=None):
    """ Creates a set of all antonyms for the word and if there is only one antonym, it returns it """
    antonyms = set()
    for syn in wordnet.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None

def replaceNegations(text):
    """ Finds "not" and antonym for the next word and if found, replaces not and the next word with the antonym """
    i, l = 0, len(text)
    words = []
    while i < l:
        word = text[i]
        if word == 'not' and i+1 < l:
            ant = replace(text[i+1])
            if ant:
                words.append(ant)
                i += 2
                continue
        words.append(word)
        i += 1
    return words

def antonym(text):
    tokens = nltk.word_tokenize(text)
    tokens = replaceNegations(tokens)
    text = " ".join(tokens)
    return text


#Remove Stopwords
stoplist = stopwords.words('english')
stoplist.remove('no')
stoplist.remove('not')
def stp_words(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if (w not in stoplist):
            finalTokens.append(w)
    text = " ".join(finalTokens)
    return text

#Remove mail related words
mail_words_list = ['hi','hello','com','gmail','cc','regards','thanks']
def mail_words(text):
    finalTokens = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if (w not in mail_words_list):
            finalTokens.append(w)
    text = " ".join(finalTokens)
    return text

#Lemmatization
stemmer = PorterStemmer() #set stemmer
lemmatizer = WordNetLemmatizer() # set lemmatizer

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

#Remove accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
for index, row in inp_df.iterrows():
    # remove numbers
    proc_des = removeNumbers(row['Full Description'])
    
    #remove punctuations
    #translator = str.maketrans('', '', string.punctuation)
    #proc_des = proc_des.translate(translator) 
    proc_des = re.sub(r"\W", " ", proc_des, flags=re.I)
    proc_des = proc_des.replace('_',' ')
    
    #replace contractions
    proc_des = replaceContraction(proc_des)
    
    #remove accents
    proc_des = remove_accented_chars(proc_des)
    
    #convert to lower case
    proc_des = proc_des.lower()
    
    #replace negation with antonym - Skipping this for our case, as it couldnt process words like cannot
    #proc_des = antonym(proc_des)
    
    #remove stopwords
    proc_des = stp_words(proc_des)
    
    #remove mail related words
    proc_des = mail_words(proc_des)
    
    #check whether the language is English
    #lang = detect(proc_des)
    
    #lemmatization
    proc_des = lemmatize_sentence(proc_des)
    
    #create new column and add updated data
    #inp_df.set_value(index, 'Full Description - After', proc_des)
    inp_df.at[index, 'Full Description - After']= proc_des
    #inp_df.set_value(index, 'Language', lang)
inp_df.head()
#Lets create temp csv to see the data as of now

#inp_df.to_csv('temp_data1.csv')
stopwords = set(STOPWORDS)

#function to create Word Cloud
def show_wordcloud(data, title):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
for grp in inp_df['Assignment group'].unique().tolist():
    text_Str = inp_df['Full Description - After'][inp_df['Assignment group'].isin([grp])].tolist()
    show_wordcloud(text_Str, '\n'+ grp +' - WORD CLOUD')
    print('===============================================================================================================')

