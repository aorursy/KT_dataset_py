import pandas as pd

from bs4 import BeautifulSoup

from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import sent_tokenize

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud



import matplotlib.pyplot as plt



from collections import Counter



import nltk

import re
df=pd.read_csv("../input/customer-support-on-twitter/twcs/twcs.csv", nrows=500)
df.head()
df_text=df["text"]

df_text=pd.DataFrame(df_text)
df_text.rename(columns={'text': 'Text'},inplace=True)
df_text.shape
df_text
df_text['Text']=df_text['Text'].str.lower()

df_text['Text'].head()
#Here I am taking some sample texts to show how  preprocessing works

sent_0=df_text['Text'].values[0]

sent_50=df_text['Text'].values[50]

sent_90=df_text['Text'].values[90]
print(sent_0)

print("="*50)

print(sent_50)

print("="*50)

print(sent_90)

#Removing the punctuation of sentences single sentence 

result_90=re.sub(r'[\,\.\?\"\;\:\'\@\-\#]','',sent_90)

result_90

preprocessed_data=[]

for i in tqdm(df_text['Text'].values):

    i=re.sub(r'[\,\.\?\"\;\:\'\@\^\&\/\!\-\#\(\)\%]','',i)

    i=re.sub(r"http\S+",'',i)# remove urls from text python: https://stackoverflow.com/a/40823105/4084039

    #'+' means one or more

    i=re.sub(r'[0-9]','',i)

    i=BeautifulSoup(i,'lxml').get_text()

    preprocessed_data.append(i.strip())
preprocessed_data
# Since the text is stored in list [type(preprocessed_data)] therefore we have to convert it into dataframe.

preprocessed_data=pd.DataFrame(preprocessed_data)

preprocessed_data=preprocessed_data.set_axis(['Text'], axis=1, inplace=False)# Since the df did't have any col name

preprocessed_data.head()
def tokenize(text):

    tokens = word_tokenize(text) #W+ means that either a word character (A-Za-z0-9_) or a dash (-) can go there.

    return tokens

preprocessed_data['Text_token']=preprocessed_data['Text'].apply(lambda x:tokenize(x))

preprocessed_data.head()
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))

word_token=word_tokenize(sent_50)

word_50=[]

for word in word_token:

    if word not in stop_words:

        word_50.append(word)

print(word_token)

print(word_50)

print('='*100)

print(sent_50)
def remove_stopwords(txt_tokenized):

    txt_clean=[]

    for i in txt_tokenized:

        if i not in stop_words:

            txt_clean.append(i)

    return txt_clean

preprocessed_data['Text_stopwords']=preprocessed_data['Text_token'].apply(lambda x:remove_stopwords(x))

preprocessed_data
#Count of words in Text_stopwords values[1]

sent_1=preprocessed_data['Text_stopwords'].values[1]

sent_1

a=Counter(sent_1)

a
'''ALGORITHM

1. Create a list of the stopwords(more frequent words) for iteration

2. Extend(don't use append because it will create multiple lists) the list and store it to a new list

3. Find out the count of each words

4. Take the top 10 rows and store it in a dictionary

5. Extract the keys and store it in a list

6. Create  function and iterate through the whole list'''

preprocessed_data_list=preprocessed_data['Text_stopwords'].values.tolist()

final_list=[]

for i in range(len(preprocessed_data_list)):

    final_list.extend(preprocessed_data_list[i])

words_count=pd.Series(final_list).value_counts()





top_frequent_words=words_count.head(10).to_dict()

dict_keys=top_frequent_words.keys()

dict_keys=list(dict_keys)





words_count
def frequent_words(qw):

    frequent_list=[]

    for i in qw:

        if i not in dict_keys:

            frequent_list.append(i)

    return frequent_list

preprocessed_data['frequent_words']=preprocessed_data['Text_stopwords'].apply(lambda x:frequent_words(x))

preprocessed_data
print(df_text.values[0])

print('='*50)

print(preprocessed_data['Text_stopwords'].values[0])

print('='*50)

print(preprocessed_data['frequent_words'].values[0])
least_words=words_count.tail(10).to_dict()

least_words_keys=list(least_words.keys())





def rare_words(qr):

    rare_words=[]

    for i in qr:

        if i not in least_words_keys:

            rare_words.append(i)

    return rare_words

preprocessed_data['rare_words']=preprocessed_data['frequent_words'].apply(lambda x:rare_words(x))
preprocessed_data
print(df_text.values[0])

print('='*50)

print(preprocessed_data['Text_stopwords'].values[0])

print('='*50)

print(preprocessed_data['frequent_words'].values[0])

print('='*50)

print(preprocessed_data['rare_words'].values[0])
def lematized(lem):

    lemmatizer=WordNetLemmatizer()

    lematized=[]

   # r_words_50=preprocessed_data['rare_words'].values[50]

    for i in lem:

        lemmatized=lemmatizer.lemmatize(i)

        lematized.append(lemmatized)

    return lematized

preprocessed_data["Lemmatized_words"]=preprocessed_data['rare_words'].apply(lambda x:lematized(x))
preprocessed_data
'''ALGORITHM

    1. Convert the dataframe to string

        i. I converted {dataframe->list->string}

NOTE: Wordcloud accepts string format

    2. Import wordcloud and use th function'''



def dftolist(er):

    x=preprocessed_data['Text']

    all_words=[]

    for i in range(len(x)):

        a=preprocessed_data['Text'].values[i]

        all_words.append(a)

    return all_words

wc_list=dftolist(preprocessed_data['Text'])





def listtostring(we):

    str=' '

    return(str.join(we))

lt=listtostring(wc_list)

wc_text=lt

wordcloud=WordCloud(width=100,height=100,max_words=500,stopwords=stop_words,background_color='white',repeat=False).generate(wc_text)

plt.imshow(wordcloud)