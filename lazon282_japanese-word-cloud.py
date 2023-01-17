import pandas as pd



CSV_PATH = '/input/abe_tweet_nlp.csv'

df = pd.read_csv(CSV_PATH,index_col=0)

texts = list(df.text)

texts
from janome.tokenizer import Tokenizer

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from collections import Counter, defaultdict
#from sentence to words

def counter(texts):

    t = Tokenizer()

    words_count =  defaultdict(int)

    words = []

    for text in texts:

        tokens = t.tokenize(text)

        for token in tokens:

            #extract just noun

            pos = token.part_of_speech.split(',')[0]

            if pos == '名詞':

                words_count[token.base_form] +=1

                words.append(token.base_form)

    return words_count, words



words_count,words = counter(texts)
sorted(words_count.items(),key=lambda x: x[1],reverse=True)
#set stop words

with open('data/Japanese_stopword_list.txt','r') as f:

    stopword_list = f.readlines()



stopword_list_mod = []

for stopword in stopword_list:

    stopword_list_mod.append(stopword.replace("\n",""))

    

stopword_list_mod
#vectorize by frequency

from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(min_df=1,stop_words=stopword_list_mod) #vector generator

vector = vectorizer.fit_transform(words)  #text to freqeuency

words_name = vectorizer.get_feature_names() #name of word
#join for word cloud 

cloud_text = ' '.join(words)



#word cloud setting

fpath = "~/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"

wordcloud = WordCloud(background_color="white",font_path=fpath,width=900,height=600).generate(cloud_text)



plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#filter by term frequency

filtered_words = []



# words_count

limit_frequency = 500

for i in range(len(words_count)):

    if list(words_count.values())[i] < limit_frequency:

        filtered_words.append(list(words_count.keys())[i])



#filter by regex

import re



words_without_number = []

for fw in filtered_words:

    if not bool(re.search(r"[0-9]", str(fw))):

        words_without_number.append(fw)



words_without_alphabet = []

for fw in words_without_number:

    if not bool(re.search(r"[a-zA-Z]", str(fw))):

        words_without_alphabet.append(fw)





#join for word cloud analysis

cloud_text = ' '.join(words_without_alphabet)



#word cloud setting

fpath = "~/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc"

wordcloud = WordCloud(background_color="white",font_path=fpath,width=900,height=600).generate(cloud_text)



plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()