

!pip install emot

import pandas as pd

import nltk

import re

from emot.emo_unicode import UNICODE_EMO, EMOTICONS

#nltk.download('words')

df=pd.read_csv('../input/tweets-text/text_emotion.csv')

df['text'] = df['text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

df.text.replace({r'[^\x00-\x7F]+':''}, regex=True, inplace=True)

df['text']=df['text'].str.strip()

df['text'] = df['text'].str.replace('#','')

df['text'] = df['text'].str.replace('@','')

df['text'] = df['text'].str.replace('_','')

df['text'] = df['text'].str.replace('&','')

df['text'] = df['text'].str.replace('*','')

df['text'] = df['text'].str.replace('(','')

df['text'] = df['text'].str.replace(')','')

df['text'] = df['text'].str.replace('|','')

df['text'] = df['text'].str.replace('~','')

df['text'] = df['text'].str.replace('Via','')

df['text'] = df['text'].str.replace('via','')

df['text'] = df['text'].str.replace('Click to','')

df['text'] = df['text'].str.replace('Click here','')

df['text'] = df['text'].str.replace('{','')

df['text'] = df['text'].str.replace('}','')

df['text'] = df['text'].str.replace('click to','')

df['text'] = df['text'].str.replace('click here','')

df['text'] = df['text'].str.replace('Click to','')

df['text'] = df['text'].str.replace('Click here','')

df['text'] = df['text'].str.replace('{','')

df['text'] = df['text'].str.replace('}','')

df['text'] = df['text'].str.replace('click to','')

df['text'] = df['text'].str.replace('click here','')

df['text'] = df['text'].str.replace('""','')

df['text'] = df['text'].str.replace('/','')

df['text'] = df['text'].str.replace('"','')

df['text'] = df['text'].str.replace('-','')

df['text'] = df['text'].str.replace(':','')

df['text'] = df['text'].str.replace('.','')

df['text'] = df['text'].str.replace('=','equals to')

df['text'] = df['text'].str.replace(',','')

df['text'] = df['text'].str.replace('its','it is')

df['text'] = df['text'].str.replace('it\'s','it is')

df['text'] = df['text'].str.replace('It\'s','It is')

df['text'] = df['text'].str.replace('Don\'t','Do not')

df['text'] = df['text'].str.replace('We\'re','We are')

#df['text'] = df['text'].str.replace('it \' s,'')

df['text']
# sorting by first name 

df.sort_values("text", inplace = True) 

  

# dropping ALL duplicte values 

df.drop_duplicates(subset ="text", 

                     keep = False, inplace = True) 


# Converting emojis to words

def convert_emojis(text):

    for emot in UNICODE_EMO:

        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()))

        return text

# Converting emoticons to words    

def convert_emoticons(text):

    for emot in EMOTICONS:

        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)

        return text

# Example

#text = "Hello :-) :-)"

#convert_emoticons(text)

#text1 = "Hilarious 😂"

#convert_emojis(text1)

# Passing both functions to 'text_rare'



df['text'] = df['text'].apply(convert_emoticons)

df['text'] = df['text'].apply(convert_emojis)
import nltk

nltk.download('words')

words = set(nltk.corpus.words.words())



#df['text']="aj bhi wahi khade ho gaye,jaha se kabhi hum I am a good boy."



df_t = pd.DataFrame(columns=['text'])



#words = ['what','is','your','name']

df['text'] = [" ".join(w for w in nltk.wordpunct_tokenize(x) 

                       if w.lower() in words or not w.isalpha()) 

                       for x in df['text']]
df['text']=df['text'].str.strip()
import re

cList = {

  "let's": "let us", 

  "It's": "It is",

  "Here's": "Here is",

  "Here ' s": "Here is",

  "it's": "it is",

  "can't": "can not",

  "Can't": "Can not",

  "it's": "it is",

  "we're": "we are",

  "that's": "that is",

  "That's": "That is",       

  "Let's": "let us",

  "here's": "here is",   

  "ain't": "am not",

  "ain ' t": "am not",

  "aren't": "are not",

  "aren ' t": "are not",

  "can't": "cannot",

  "can ' t": "cannot",

  "can't ' ve": "cannot have",

  "'cause": "because",

  "' cause": "because",

  "could've": "could have",

  "could ' ve": "could have",

  "couldn't": "could not",

  "couldn ' t": "could not",

  "couldn't've": "could not have",

  "couldn't ' ve": "could not have",

  "didn't": "did not",

  "didn ' t": "did not",

  "doesn't": "does not",

  "doesn ' t": "does not",

  "don't": "do not",

  "don ' t": "do not",

  "hadn't": "had not",

  "hadn ' t": "had not",

  "hadn't've": "had not have",

  "hadn't ' ve": "had not have",

  "hasn't": "has not",

  "hasn ' t": "has not",

  "haven't": "have not",

  "haven ' t": "have not",

  "he'd": "he would",

  "he ' d": "he would",

  "he'd've": "he would have",

  "he'll": "he will",

  "He'll": "he will",

  "He ' ll": "he will",

  "he ' ll": "he will",

  "he'll've": "he will have",

  "he's": "he is",

  "how'd": "how did",

  "how'd'y": "how do you",

  "how'll": "how will",

  "how's": "how is",

  "I'd": "I would",

  "I'd've": "I would have",

  "I'll": "I will",

  "I'll've": "I will have",

  "I'm": "I am",

  "I ' m": "I am",

  "I've": "I have",

  "isn't": "is not",

  "it'd": "it had",

  "it'd've": "it would have",

  "it'll": "it will",

  "it'll've": "it will have",

  "it's": "it is",

  "let's": "let us",

  "ma'am": "madam",

  "mayn't": "may not",

  "might've": "might have",

  "mightn't": "might not",

  "mightn't've": "might not have",

  "must've": "must have",

  "mustn't": "must not",

  "mustn't've": "must not have",

  "needn't": "need not",

  "needn't've": "need not have",

  "o'clock": "of the clock",

  "oughtn't": "ought not",

  "oughtn't've": "ought not have",

  "shan't": "shall not",

  "sha'n't": "shall not",

  "shan't've": "shall not have",

  "she'd": "she would",

  "she'd've": "she would have",

  "she'll": "she will",

  "she'll've": "she will have",

  "she's": "she is",

  "should've": "should have",

  "shouldn't": "should not",

  "shouldn't've": "should not have",

  "so've": "so have",

  "so's": "so is",

  "that'd": "that would",

  "that'd've": "that would have",

  "that's": "that is",

  "there'd": "there had",

  "there'd've": "there would have",

  "there's": "there is",

  "they'd": "they would",

  "they'd've": "they would have",

  "they'll": "they will",

  "they'll've": "they will have",

  "they're": "they are",

  "they've": "they have",

  "to've": "to have",

  "wasn't": "was not",

  "we'd": "we had",

  "we'd've": "we would have",

  "we'll": "we will",

  "we'll've": "we will have",

  "we're": "we are",

  "we've": "we have",

  "weren't": "were not",

  "what'll": "what will",

  "what'll've": "what will have",

  "what're": "what are",

  "what's": "what is",

  "what've": "what have",

  "when's": "when is",

  "when've": "when have",

  "where'd": "where did",

  "where's": "where is",

  "where've": "where have",

  "who'll": "who will",

  "who'll've": "who will have",

  "who's": "who is",

  "who've": "who have",

  "why's": "why is",

  "why've": "why have",

  "will've": "will have",

  "won't": "will not",

  "won't've": "will not have",

  "would've": "would have",

  "wouldn't": "would not",

  "wouldn't've": "would not have",

  "y'all": "you all",

  "y'alls": "you alls",

  "y'all'd": "you all would",

  "y'all'd've": "you all would have",

  "y'all're": "you all are",

  "y'all've": "you all have",

  "you'd": "you had",

  "you'd've": "you would have",

  "you'll": "you you will",

  "you'll've": "you you will have",

  "you're": "you are",

  "You're": "You are",

  "ma'am": "madam",

  "Ma'am": "Madam",

  "you've": "you have"

}



c_re = re.compile('(%s)' % '|'.join(cList.keys()))



def expandContractions(text, c_re=c_re):

    def replace(match):

        return cList[match.group(0)]

    return c_re.sub(replace, text)



# examples

for index, row in df.iterrows():

    #row['text']=expandContractions(row['text'])

    #print(row['text'])

    df.at[index,'text'] = expandContractions(row['text'])



df['text']
#df['text']=' '.join(df['text'].split())

for index, row in df.iterrows():

    #row['text']=expandContractions(row['text'])

    #print(row['text'])

    df.at[index,'text'] = ' '.join(row['text'].split())
import string

df['text'] = df['text'].str.strip(string.punctuation)
df['text']=df['text'].str.strip("/")

df['text']=df['text'].str.lstrip(",")

df['text']=df['text'].str.rstrip(",")

#df['text']=df['text'].str.rstrip(".")
df['text'] = df['text'].str.lstrip(string.punctuation)

df['text'] = df['text'].str.rstrip(string.punctuation)
import numpy as np

df['text'].replace('', np.nan, inplace=True)

df.dropna(subset = ['text'],inplace=True)
df['text']
df['text']=df['text'].str.lstrip(" ")

df['text']=df['text'].str.lstrip(" ")

df['text']=df['text'].str.lstrip(",")

df['text']=df['text'].str.rstrip(",")

df['text']=df['text'].str.lstrip(" ")

df['text']=df['text'].str.lstrip(" ")

import numpy as np

df['text'].replace('', np.nan, inplace=True)

df.dropna(subset = ['text'],inplace=True)
print(df.shape)
df.to_csv('train_data.csv')