import pandas as pd



df = pd.read_csv('../input/en.yusufali.csv')

df.info()
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



if 'Para' not in df.columns:

    df.insert(2, 'Para', pd.cut(df.index,cut_points,labels=label_names))

df.drop('index', axis=1, inplace=True)

df['Para'] = pd.to_numeric(df['Para'])

df.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



fig = plt.figure(figsize=(20,5))

ax = fig.add_subplot(1,1,1)

df.plot.scatter('Surah', 'Ayah', ax=ax)    
all_text = []

for text in df['Text']:

    all_text.append(text.split(' '))



punctuation = [",", ":", ";", ".", "'", '"', "’", "?", "/", "-", "+", "&", "(", ")", "!"]

clean_text = []



for item in all_text:

    tokens = []

    for i in item:

        i = i.lower()

        for p in punctuation:

            i = i.replace(p, '')

        tokens.append(i)

    clean_text.append(tokens)



cleaned_rows = []

[cleaned_rows.append(' '.join(c)) for c in clean_text]

df['Clean Text'] = cleaned_rows

df.head()
import numpy as np



unique_tokens = []

single_tokens = []



for tokens in clean_text:

    for token in tokens:

        if token not in single_tokens:

            single_tokens.append(token)

        elif token not in unique_tokens:

            unique_tokens.append(token)

            

counts = pd.DataFrame(0, index=np.arange(len(clean_text)), columns=unique_tokens)

for index, tokens in enumerate(clean_text):

    for token in tokens:

        if token in unique_tokens:

            counts.iloc[index][token] += 1

            

counts.head()
word_counts = counts.sum(axis=0)



from wordcloud import WordCloud



word_list = [word for word in word_counts.index]

wordcloud = WordCloud(max_font_size=40).generate(' '.join(word_list))



fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(1,1,1)

ax.imshow(wordcloud, interpolation="bilinear")

ax.axis("off")

stopwords = ['all', 'just', 'being', 'over', 'both', 'through', 'yourselves', 'its', 'before', 'herself', 'had',

             'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during',

             'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing',

             'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't',

             'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own',

            'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too',

             'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those',

             'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can','theirs', 'my', 'and', 'then', 'is',

             'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same',

             'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so',

             'the', 'having', 'once', 'say', 'thou', 'said', 'shall', 'thee', 'us', 'ye', 'o', 'sent', 'thy',

             'come', 'see', 'made', 'give', 'may', ' ']



clean_word_counts = []

for word in word_counts.index:

    if word in stopwords:

        word_counts.drop(word, inplace=True)

        

word_counts.sort_values(ascending=False).head(20)
words = ['allah', 'truth', 'lie','man', 'men', 'woman', 'women', 'heaven', 'hell', 'paradise', 'hellfire', 'good', 'evil']



word_times = []

for w in words:

    if w not in word_times:

        word_times.append([w, word_counts[w]])

    

times_occurence = pd.DataFrame(data = word_times, columns=['Word', 'Times'])

times_occurence.head(20)