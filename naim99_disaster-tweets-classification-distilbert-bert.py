import numpy as np

import pandas as pd

import torch

import transformers as ppb # pytorch transformers

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import re
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df_test = pd.read_csv('../input/nlp-getting-started/test.csv') 
"""df['text'] = df['text'].str.lower() #lowercase

df_test['text'] = df_test['text']

df['text'] = df['text'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

df_test['text'] = df_test['text'].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

# remove numbers

#remove.............. (#re sub / search/ ..)

df['text'] = df['text'].apply(lambda elem: re.sub(r"\d+", "", elem))

df_test['text'] = df_test['text'].apply(lambda elem: re.sub(r"\d+", "", elem))

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)







df['text'] = df['text'].apply(lambda x: remove_URL(x))

df_test['text'] = df_test['text'].apply(lambda x: remove_URL(x))

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



df['text'] = df['text'].apply(lambda x: remove_html(x))

df_test['text'] = df_test['text'].apply(lambda x: remove_html(x))

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b



import string

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



df['text'] = df['text'].apply(lambda x: remove_punct(x))

df_test['text'] = df_test['text'].apply(lambda x: remove_punct(x)) """ 
"""def text_to_wordlist(text, remove_stop_words=False, stem_words=False):

    # Clean the text, with the option to remove stop_words and to stem words.



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9]", " ", text)

    text = re.sub(r"what's", "", text)

    text = re.sub(r"What's", "", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"I'm", "I am", text)

    text = re.sub(r" m ", " am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"60k", " 60000 ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e-mail", "email", text)

    text = re.sub(r"\s{2,}", " ", text)

    text = re.sub(r"quikly", "quickly", text)

    text = re.sub(r" usa ", " America ", text)

    text = re.sub(r" USA ", " America ", text)

    text = re.sub(r" u s ", " America ", text)

    text = re.sub(r" uk ", " England ", text)

    text = re.sub(r" UK ", " England ", text)

    text = re.sub(r"india", "India", text)

    text = re.sub(r"switzerland", "Switzerland", text)

    text = re.sub(r"china", "China", text)

    text = re.sub(r"chinese", "Chinese", text) 

    text = re.sub(r"imrovement", "improvement", text)

    text = re.sub(r"intially", "initially", text)

    text = re.sub(r"quora", "Quora", text)

    text = re.sub(r" dms ", "direct messages ", text)  

    text = re.sub(r"demonitization", "demonetization", text) 

    text = re.sub(r"actived", "active", text)

    text = re.sub(r"kms", " kilometers ", text)

    text = re.sub(r"KMs", " kilometers ", text)

    text = re.sub(r" cs ", " computer science ", text) 

    text = re.sub(r" upvotes ", " up votes ", text)

    text = re.sub(r" iPhone ", " phone ", text)

    text = re.sub(r"\0rs ", " rs ", text) 

    text = re.sub(r"calender", "calendar", text)

    text = re.sub(r"ios", "operating system", text)

    text = re.sub(r"gps", "GPS", text)

    text = re.sub(r"gst", "GST", text)

    text = re.sub(r"programing", "programming", text)

    text = re.sub(r"bestfriend", "best friend", text)

    text = re.sub(r"dna", "DNA", text)

    text = re.sub(r"III", "3", text) 

    text = re.sub(r"the US", "America", text)

    text = re.sub(r"Astrology", "astrology", text)

    text = re.sub(r"Method", "method", text)

    text = re.sub(r"Find", "find", text) 

    text = re.sub(r"banglore", "Banglore", text)

    text = re.sub(r" J K ", " JK ", text)

    return(text) 



df['text'] = df['text'].apply(lambda x: text_to_wordlist(x))

df_test['text'] = df_test['text'].apply(lambda x: text_to_wordlist(x)) """ 
"""# replace strange punctuations and raplace diacritics

from unicodedata import category, name, normalize



def remove_diacritics(s):

    return ''.join(c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))

                  if category(c) != 'Mn')



special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',

                         "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',

                         '…': ' ... ', '\ufeff': ''}

def clean_special_punctuations(text):

    for punc in special_punc_mappings:

        if punc in text:

            text = text.replace(punc, special_punc_mappings[punc])

  

    text = remove_diacritics(text)

    return text



df['text'] = df['text'].apply(lambda x: remove_diacritics(x))

df['text'] = df['text'].apply(lambda x: clean_special_punctuations(x))

df_test['text'] = df_test['text'].apply(lambda x: clean_special_punctuations(x))

df_test['text'] = df_test['text'].apply(lambda x: remove_diacritics(x)) """
""""# clean numbers

def clean_number(text):

   

    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)

    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)

    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)

    

#     text = re.sub('[0-9]{5,}', '#####', text)

#     text = re.sub('[0-9]{4}', '####', text)

#     text = re.sub('[0-9]{3}', '###', text)

#     text = re.sub('[0-9]{2}', '##', text)

    

    return text



df['text'] = df['text'].apply(lambda x: clean_number(x))

df_test['text'] = df_test['text'].apply(lambda x: clean_number(x))"""
# de-contract the contraction

def decontracted(text):

    # specific

    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)

    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)

    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)

    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)



    # general

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)

    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)

    text = re.sub(r"n(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)re ", " are ", text)

    text = re.sub(r"(\'|\’)s ", " is ", text)

    text = re.sub(r"(\'|\’)d ", " would ", text)

    text = re.sub(r"(\'|\’)ll ", " will ", text)

    text = re.sub(r"(\'|\’)t ", " not ", text)

    text = re.sub(r"(\'|\’)ve ", " have ", text)

    return text



df['text'] = df['text'].apply(lambda x: decontracted(x))

df_test['text'] = df_test['text'].apply(lambda x: decontracted(x))
import string

regular_punct = list(string.punctuation)

extra_punct = [

    ',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',

    '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',

    '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',

    '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',

    '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',

    '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',

    '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',

    'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',

    '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',

    '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']

all_punct = list(set(regular_punct + extra_punct))

# do not spacing - and .

all_punct.remove('-')

all_punct.remove('.')



def spacing_punctuation(text):

    """

    add space before and after punctuation and symbols

    """

    for punc in all_punct:

        if punc in text:

            text = text.replace(punc, f' {punc} ')

    return text



df['text'] = df['text'].apply(lambda x: spacing_punctuation(x))

df_test['text'] = df_test['text'].apply(lambda x: spacing_punctuation(x))
mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']

mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))



mis_spell_mapping = {'whattsup': 'WhatsApp', 'whatasapp':'WhatsApp', 'whatsupp':'WhatsApp', 

                      'whatcus':'what cause', 'arewhatsapp': 'are WhatsApp', 'Hwhat':'what',

                      'Whwhat': 'What', 'whatshapp':'WhatsApp', 'howhat':'how that',

                      # why

                      'Whybis':'Why is', 'laowhy86':'Foreigners who do not respect China',

                      'Whyco-education':'Why co-education',

                      # How

                      "Howddo":"How do", 'Howeber':'However', 'Showh':'Show',

                      "Willowmagic":'Willow magic', 'WillsEye':'Will Eye', 'Williby':'will by'}

def spacing_some_connect_words(text):

    """

    'Whyare' -> 'Why are'

    """

    ori = text

    for error in mis_spell_mapping:

        if error in text:

            text = text.replace(error, mis_spell_mapping[error])

            

    # what

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)

    text = re.sub(r" (W|w)hat\S ", " What ", text)

    text = re.sub(r" \S(W|w)hat ", " What ", text)

    # why

    text = re.sub(r" (W|w)hy\S ", " Why ", text)

    text = re.sub(r" \S(W|w)hy ", " Why ", text)

    # How

    text = re.sub(r" (H|h)ow\S ", " How ", text)

    text = re.sub(r" \S(H|h)ow ", " How ", text)

    # which

    text = re.sub(r" (W|w)hich\S ", " Which ", text)

    text = re.sub(r" \S(W|w)hich ", " Which ", text)

    # where

    text = re.sub(r" (W|w)here\S ", " Where ", text)

    text = re.sub(r" \S(W|w)here ", " Where ", text)

    # 

    text = mis_connect_re.sub(r" \1 ", text)

    text = text.replace("What sApp", 'WhatsApp')

    

    

    return text



df['text'] = df['text'].apply(lambda x: spacing_some_connect_words(x))

df_test['text'] = df_test['text'].apply(lambda x: spacing_some_connect_words(x))
#https://www.kaggle.com/sunnymarkliu/more-text-cleaning-to-increase-word-coverage
df.head() 
#Importing pre-trained DistilBERT model and tokenizer

#model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')



## Want BERT instead of distilBERT? Uncomment the following line:

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')



# Load pretrained model/tokenizer

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

model = model_class.from_pretrained(pretrained_weights)
df = df[:4000]
#we’ll tokenize and process all sentences together as a batch 

tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0

for i in tokenized.values:

    if len(i) > max_len:

        max_len = len(i)



padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)

attention_mask.shape
#The `model()` function runs our sentences through BERT. The results of the processing will be returned into `last_hidden_states`.

input_ids = torch.tensor(padded)  

attention_mask = torch.tensor(attention_mask)



with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask)
features = last_hidden_states[0][:,0,:].numpy()

features[0].shape



labels = df['target']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
import sklearn

from sklearn.model_selection import GridSearchCV

parameters = {'C': np.linspace(0.0001, 100, 20)}

grid_search = GridSearchCV(LogisticRegression(), parameters)

grid_search.fit(train_features, train_labels)

print('best parameters: ', grid_search.best_params_)

print('best scrores: ', grid_search.best_score_)
lr_clf = LogisticRegression(C = 31.579015789473683)

lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)
from sklearn.dummy import DummyClassifier

clf = DummyClassifier()



scores = cross_val_score(clf, train_features, train_labels)

print("Dummy classifier score: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
tokenized_t = df_test['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
max_len = 0

for i in tokenized_t.values:

    if len(i) > max_len:

        max_len = len(i)
padded_t = np.array([i + [0]*(max_len-len(i)) for i in tokenized_t.values])

np.array(padded_t).shape
attention_mask_t = np.where(padded_t != 0, 1, 0)
input_ids = torch.tensor(padded_t)  

input_ids
attention_mask_t = torch.tensor(attention_mask_t)



with torch.no_grad():

    last_hidden_states = model(input_ids, attention_mask=attention_mask_t)
val_features = last_hidden_states[0][:,0,:].numpy() 
y_pred = lr_clf.predict(val_features)

y_pred
# Create submission file

submission = pd.DataFrame()

submission['id'] = df_test['id']

submission['target'] = y_pred

submission.to_csv('submission.csv', index=False)