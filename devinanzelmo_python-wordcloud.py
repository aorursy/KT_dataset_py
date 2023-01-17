import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
import wordcloud
chat = pd.read_csv('../input/chat.csv')
chat.head()
# generates wordcloud over a time interval in the game, with the option to add stopwords, to remove some

# of the spam the clutters chat



def time_interval_wc(start_time, end_time, chat_df, extra_stop={''}):

    selected_chat = chat_df.query('time > @start_time and time < @ end_time')

    text = ' '.join(selected_chat['key'].astype(str).tolist())

    stopwords = set(wordcloud.STOPWORDS)

    stopwords = stopwords.union(extra_stop)

    return wordcloud.WordCloud(stopwords=stopwords,background_color='black',colormap='terrain', width=800, height=600).generate(text)
def show(im):

    plt.figure(figsize=(7,7), frameon=False)

    plt.imshow(im)  

    plt.axis('off')

    plt.show()
wc_tmp = time_interval_wc(-10000, 0, chat)

show(wc_tmp)
# add some more stopwords to get rid of very common items



some_stopwords = {'lol', 'fuck', 'ok', 'gg','gl','hf', 'glhf'}

wc_tmp = time_interval_wc(-10000, 0, chat,some_stopwords)

show(wc_tmp)
some_stopwords = {'lol', 'fuck', 'ok', 'gg','gl','hf', 'glhf', 'ty', 

                  'guy','GUY','Guy', 'go', 'wait','pl','pls'}

wc_tmp = time_interval_wc(-10000, 0, chat,some_stopwords)

show(wc_tmp)
some_stopwords = {'lol', 'fuck', 'ok', 'gg','gl','hf', 'glhf', 'ty', 

                  'guy','GUY','Guy', 'go', 'wait','pl','pls'}

wc_tmp = time_interval_wc(2000, 1000000, chat,some_stopwords)

show(wc_tmp)
some_stopwords = {'lol','ggwp','noob','XD' ,'wp','ez','fuck', 'ok', 'gg',

                  'gl','hf', 'glhf', 'ty', 'guy','GUY','Guy', 'go', 'wait','pl','pls'}

wc_tmp = time_interval_wc(2000, 1000000, chat,some_stopwords)

show(wc_tmp)
some_stopwords = {'game','haha','mid','lol','ggwp','noob','XD' ,'wp','ez','fuck', 'ok', 'gg',

                  'gl','hf', 'glhf', 'ty', 'guy','GUY','Guy', 'go', 'wait','pl','pls'}

wc_tmp = time_interval_wc(2000, 1000000, chat,some_stopwords)

show(wc_tmp)