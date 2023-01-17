from PIL import Image

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from tqdm import tqdm

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
tweets_df = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

tweets_df.sample(10)
print(tweets_df.describe())

print('\n')

print(tweets_df.info())
text_count = tweets_df.shape[0]

percentage_selected = []



tweets_df['text_len']         = tweets_df['text'].astype(str).apply(lambda x: len(x.split()))

tweets_df['sel_text_len']     = tweets_df['selected_text'].astype(str).apply(lambda x: len(x.split()))

tweets_df['% words selected'] =  tweets_df['text_len'] / tweets_df['sel_text_len'] 
tweets_df.sample(15)
mean_word_selection_ratio = tweets_df['% words selected'].mean()

mean_word_selection_ratio
plt.figure(figsize=(12, 6))

sns.countplot(x='sentiment', data=tweets_df)

plt.savefig('bar_plot_sns.png')
labels = tweets_df['sentiment'].unique().tolist() # [neutral, negative, positive]

counts = tweets_df['sentiment'].value_counts().tolist() # [neutral, positive, negative]

aux = counts[2]

counts[2] = counts[1]

counts[1] = aux



# show bar plots of the target class distribution

plt.figure(figsize=(12, 6))

plt.bar(labels, counts, edgecolor='black', color='lightblue')

plt.savefig('bar_plot.png')

plt.show()
# join all selected text 

text_list = tweets_df['selected_text'].astype(str).tolist()

all_text = ''

for new_text in tqdm(text_list):

    all_text = all_text + ' ' + new_text
# join positive selected text 

pos_text_list = tweets_df[tweets_df['sentiment'] == 'positive']['selected_text'].astype(str).tolist()

pos_all_text = ''

for new_text in tqdm(pos_text_list):

    pos_all_text = pos_all_text + ' ' + new_text
# join negative selected text 

neg_text_list = tweets_df[tweets_df['sentiment'] == 'negative']['selected_text'].astype(str).tolist()

neg_all_text = ''

for new_text in tqdm(neg_text_list):

    neg_all_text = neg_all_text + ' ' + new_text
# defining stopwords

stopwords = set(STOPWORDS)

stopwords.update(['to', 'im', 'will'])



# generating wordcloud with no parameters

wordcloud = WordCloud().generate(all_text)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
# read ballon mask

ballon_mask = np.array(Image.open('../input/ballon-render/ballon.png'))

print(ballon_mask.shape)

plt.imshow(ballon_mask[:][:][:])
# transforms mask

def transform_format(val):

    if val == 0:

        return 255

    else:

        return 0
# creates a ballon mask

transformed_ballon_mask = np.ndarray((ballon_mask.shape[0], ballon_mask.shape[1], ballon_mask.shape[2]), np.int32) 



for i in tqdm(range(ballon_mask.shape[0])):

    for j in range(ballon_mask.shape[1]):

        if ballon_mask[i][j][3] == 0:

            transformed_ballon_mask[i][j][:3] = list(map(transform_format, ballon_mask[i][j][:3]))

            transformed_ballon_mask[i][j][3] = 255

        else:

            transformed_ballon_mask[i][j][3] = 0
plt.imshow(transformed_ballon_mask)

plt.show()
# create and generate a word cloud image

wordcloud = WordCloud(background_color='black', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)

wordcloud_white = WordCloud(background_color='white', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)



pos_wordcloud = WordCloud(background_color='black', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)

pos_wordcloud_white = WordCloud(background_color='white', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)



neg_wordcloud = WordCloud(background_color='black', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)

neg_wordcloud_white = WordCloud(background_color='white', mask=transformed_ballon_mask, stopwords=stopwords, contour_width=3)



# generate texts

## all

wordcloud.generate(all_text)

wordcloud_white.generate(all_text)



## positive

pos_wordcloud.generate(pos_all_text)

pos_wordcloud_white.generate(pos_all_text)



## negative

neg_wordcloud.generate(neg_all_text)

neg_wordcloud_white.generate(neg_all_text)



# save wordcloud

#wordcloud.to_file('masked_wordcloud.png')

#wordcloud_white.to_file('masked_wordcloud_white.png')



#pos_wordcloud.to_file('pos_masked_wordcloud.png')

#pos_wordcloud_white.to_file('pos_masked_wordcloud_white.png')



#neg_wordcloud.to_file('neg_masked_wordcloud.png')

#neg_wordcloud_white.to_file('neg_masked_wordcloud_white.png')



# show

plt.figure(figsize=[10,20])

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
plt.imshow(wordcloud_white, interpolation='bilinear')

plt.axis('off')

plt.show()
plt.figure(figsize=[10,20])

plt.imshow(pos_wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()
plt.figure(figsize=[10,20])

plt.imshow(neg_wordcloud, interpolation='bilinear')

plt.axis('off')

plt.show()