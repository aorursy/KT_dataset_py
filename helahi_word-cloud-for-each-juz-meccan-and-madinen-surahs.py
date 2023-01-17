import numpy as np

import pandas as pd

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt



def create_wordcloud(text, custom_stopwords):

    for sw in custom_stopwords:

        STOPWORDS.add(sw);

    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', width=800, height=400).generate(text)

    plt.figure( figsize=(20,10) )

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
df = pd.read_csv('../input/en.yusufali.csv', dtype=object)



for col in ['Surah', 'Ayah']:

    df[col] = pd.to_numeric(df[col])



def idx(i, j):

    df['index'] = df.index

    return int(df.loc[(df['Surah']==i) & (df['Ayah']==j), 'index'])



cut_points = [-1, idx(2,141), idx(2,252), idx(3,92), idx(4,23), idx(4,147), idx(5,81), idx(6,110), idx(7,87), idx(8,40),

             idx(9,92), idx(11,5), idx(12,52), idx(14,52), idx(16,128), idx(18,74), idx(20,135), idx(22,78), idx(25,20),

             idx(27,55), idx(29,45), idx(33,30), idx(36,27), idx(39,31), idx(41,46), idx(45,37), idx(51,30), idx(57,29),

             idx(66,12), idx(77,50), idx(114,6)]

label_names = [str(i) for i in range(1, len(cut_points))]



if 'Juz' not in df.columns:

    df.insert(2, 'Juz', pd.cut(df.index,cut_points,labels=label_names))

df.drop('index', axis=1, inplace=True)

df['Juz'] = pd.to_numeric(df['Juz'])

df.head(10)
meccan_madinen_surahs = {1: 'MC', 2: 'MD', 3: 'MD', 4: 'MD', 5: 'MD', 6: 'MC', 7: 'MC',

                         8: 'MD', 9: 'MD', 10: 'MC', 11: 'MC', 12: 'MC', 13: 'MD', 14: 'MC',

                         15: 'MC', 16: 'MC', 17: 'MC', 18: 'MC', 19: 'MC', 20: 'MC', 21: 'MC',

                         22: 'MD', 23: 'MC', 24: 'MD', 25: 'MC', 26: 'MC', 27: 'MC', 28: 'MC',

                         29: 'MC', 30: 'MC', 31: 'MC', 32: 'MC', 33: 'MD', 34: 'MC', 35: 'MC',

                         36: 'MC', 37: 'MC', 38: 'MC', 39: 'MC', 40: 'MC', 41: 'MC', 42: 'MC',

                         43: 'MC', 44: 'MC', 45: 'MC', 46: 'MC', 47: 'MD', 48: 'MD', 49: 'MD',

                         50: 'MC', 51: 'MC', 52: 'MC', 53: 'MC', 54: 'MC', 55: 'MD', 56: 'MC',

                         57: 'MD', 58: 'MD', 59: 'MD', 60: 'MD', 61: 'MD', 62: 'MD', 63: 'MD',

                         64: 'MD', 65: 'MD', 66: 'MD', 67: 'MC', 68: 'MC', 69: 'MC', 70: 'MC',

                         71: 'MC', 72: 'MC', 73: 'MC', 74: 'MC', 75: 'MC', 76: 'MD', 77: 'MC',

                         78: 'MC', 79: 'MC', 80: 'MC', 81: 'MC', 82: 'MC', 83: 'MC', 84: 'MC',

                         85: 'MC', 86: 'MC', 87: 'MC', 88: 'MC', 89: 'MC', 90: 'MC', 91: 'MC',

                         92: 'MC', 93: 'MC', 94: 'MC', 95: 'MC', 96: 'MC', 97: 'MC', 98: 'MD',

                         99: 'MD', 100: 'MC', 101: 'MC', 102: 'MC', 103: 'MC', 104: 'MC', 105: 'MC',

                         106: 'MC', 107: 'MC', 108: 'MC', 109: 'MC', 110: 'MD', 111: 'MC', 112: 'MC',

                         113: 'MC', 114: 'MC'}

meccan_text = ""

medinan_text = ""
juz_text = []

juz_text.append("")

for i in range(1,31):

    juz_text.append("")

for i, row in df.iterrows():

    try:

        juz_text[row['Juz']] += row['Text']

        if meccan_madinen_surahs[row['Surah']] == 'MC':

            meccan_text += row['Text']

        else:

            medinan_text += row['Text']

    except:

        pass
custom_stopwords = ['ye', 'verily', 'will', 'said', 'say', 'us', 'thy', 'thee', 'hath']

for i, entry in enumerate(juz_text):

    if i != 0:

        print('Juz #' + str(i))

        create_wordcloud(entry, custom_stopwords)



print("Meccan Surahs")

create_wordcloud(meccan_text, custom_stopwords)

print("Medinan Surahs")

create_wordcloud(medinan_text, custom_stopwords)