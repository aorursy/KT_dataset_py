import pandas as pd



#CSVを読み込む

df = pd.read_csv('../input/kojima-tweets/kojima_tweets.csv')



#データフレームを作成し、いいねが多い順に並べる

tweets_df = df.sort_values(by='Favourites', ascending=False)



#分析に必要なデータのみを取り出す

tweets_df = tweets_df[['Tweet', 'Retweets', 'Created Date', 'Favourites']]
tweets_df.head()
!pip install wordcloud
# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip
text_list = tweets_df['Tweet'].to_list()

text = ' '.join(text_list)

text
from janome.tokenizer import Tokenizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt 



full_text = text

 

t = Tokenizer()

tokens = t.tokenize(full_text)

 

word_list=[]

for token in tokens:

    word = token.surface

    partOfSpeech = token.part_of_speech.split(',')[0]

    partOfSpeech2 = token.part_of_speech.split(',')[1]

     

    if partOfSpeech == "名詞":

        if (partOfSpeech2 != "非自立") and (partOfSpeech2 != "代名詞") and (partOfSpeech2 != "数"):

            word_list.append(word)

 

words_wakati=" ".join(word_list)

#print(words_wakati)  

 

stop_words = ['https','co','ため']  

fpath = '/kaggle/working/NotoSansCJKjp-Regular.otf'  # 日本語フォント指定

 

wordcloud = WordCloud(

    font_path=fpath,

    width=900, height=600,   # default width=400, height=200

    background_color="white",   # default=”black”

    stopwords=set(stop_words),

    max_words=200,   # default=200

    min_font_size=4,   #default=4

    collocations = False   #default = True

    ).generate(words_wakati)

 

plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("word_cloud.png")

plt.show()