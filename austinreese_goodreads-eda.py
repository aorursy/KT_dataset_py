import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sn

import nltk

import urllib.request

from PIL import Image

from io import BytesIO

from nltk.corpus import stopwords

from re import sub, match

from wordcloud import WordCloud, ImageColorGenerator

from random import shuffle

from matplotlib import rcParams



rcParams.update({'figure.autolayout': True})



books = pd.read_csv("../input/goodreads-books/goodreads_books.csv")

books.dtypes
corr_matrix = books.corr()

fig, ax = plt.subplots(figsize=(12,10))

sn.heatmap(corr_matrix)
def populate_genre_cols(books, genre):

    book_genres = books["genre_and_votes"]

    if type(book_genres) == type(float('nan')):

        return None

    genre_list = book_genres.split(",")

    for i in genre_list:

        if genre == " ".join(i.split()[:-1]):

            return int(i.split()[-1].replace("user", ""))

    return None

        

genre_count = {}



for i in books["genre_and_votes"]:

    if type(i) != type(float('nan')):

        book_genres = [" ".join(j.split()[:-1]) for j in i.split(",")]

        for j in book_genres:

            if j not in genre_count.keys():

                genre_count[j] = 1

            else:

                genre_count[j] += 1

                

genres_to_use = set()



for k, v in genre_count.items():

    if v >= 1000:

        genres_to_use.add(k)



for i in genres_to_use:

    books[i.lower().replace(" ", "_")] = books.apply(populate_genre_cols, genre=i, axis=1)



print("Percentage of books with specific genre listed")

for i in genres_to_use:

    i = i.lower().replace(" ", "_")

    print(f"{i}: {round(books[i].notna().sum() / books.shape[0] * 100, 2)}%")
base_colors = ["indianred", "red", "orangered", "chocolate", "saddlebrown",

          "orange", "gold", "yellow", "yellowgreen", "greenyellow", "limegreen",

          "mediumseagreen", "mediumaquamarine", "turquoise", "deepskyblue", "dodgerblue",

          "royalblue", "darkblue", "mediumpurple", "darkviolet", "purple", "mediumvioletred",

          "crimson"]



genre_df = books[[i.lower().replace(" ", "_") for i in genres_to_use]]

unique_genres = genre_df.columns

shuffle(base_colors)

fig, ax = plt.subplots(4, figsize=(18,32))



mean_genres = genre_df.mean().tolist()

sorted_mean_genres = mean_genres.copy()

sorted_mean_genres.sort()

sorted_mean_genres.reverse()

sorted_mean_genre_str = [None for i in range(len(sorted_mean_genres))]

for i in range(len(sorted_mean_genres)):

    sorted_mean_genre_str[i] = unique_genres[mean_genres.index(sorted_mean_genres[i])]

ax[0].bar(sorted_mean_genre_str, sorted_mean_genres, color = base_colors)

ax[0].set_title("Mean genre votes")

ax[0].tick_params(labelrotation=90)



shuffle(base_colors)

median_genres = genre_df.median().tolist()

sorted_median_genres = median_genres.copy()

sorted_median_genres.sort()

sorted_median_genres.reverse()

sorted_median_genre_str = [None for i in range(len(sorted_median_genres))]

for i in range(len(sorted_median_genres)):

    sorted_median_genre_str[i] = unique_genres[median_genres.index(sorted_median_genres[i])]

ax[1].bar(sorted_median_genre_str, sorted_median_genres, color = base_colors)

ax[1].set_title("Median genre votes")

ax[1].tick_params(labelrotation=90)



shuffle(base_colors)

sum_genres = genre_df.sum().tolist()

sorted_sum_genres = sum_genres.copy()

sorted_sum_genres.sort()

sorted_sum_genres.reverse()

sorted_sum_genres_str = [None for i in range(len(sorted_sum_genres))]

for i in range(len(sorted_sum_genres)):

    sorted_sum_genres_str[i] = unique_genres[sum_genres.index(sorted_sum_genres[i])]

ax[2].bar(sorted_sum_genres_str, sorted_sum_genres, color = base_colors)

ax[2].set_title("Sum of genre votes")

ax[2].tick_params(labelrotation=90)



shuffle(base_colors)

present_genres = genre_df[genre_df != 0].count().tolist()

sorted_present_genres = present_genres.copy()

sorted_present_genres.sort()

sorted_present_genres.reverse()

sorted_present_genres_str = [None for i in range(len(sorted_present_genres))]

for i in range(len(sorted_present_genres)):

    sorted_present_genres_str[i] = unique_genres[present_genres.index(sorted_present_genres[i])]

ax[3].bar(sorted_present_genres_str, sorted_present_genres, color = base_colors)

ax[3].set_title("Number of books with labeled genre")

ax[3].tick_params(labelrotation=90)

sw = stopwords.words('english')

words = []



for text in books["description"]:

    if type(text) == type(float("nan")):

        continue

    text = text.lower()

    text = sub(r'\[.*?\]', '', text)

    text = sub(r'([.!,?])', r' \1 ', text)

    text = sub(r'[^a-zA-Z.,!?]+', r' ', text)

    text = [i for i in text.split() if i not in sw]

    for word in text:

        words.append(word)



word_freq = nltk.FreqDist([i for i in words if len(i) > 2])

# plt.figure(figsize=(16, 6))

# word_freq.plot(50)



book_img = 'https://www.pinclipart.com/picdir/middle/365-3651885_book-black-and-white-png-peoplesoft-learn-peoplesoft.png'

with urllib.request.urlopen(book_img) as url:

    f = BytesIO(url.read())

img = Image.open(f)



mask = np.array(img)

img_color = ImageColorGenerator(mask)



wc = WordCloud(background_color='white',

              mask=mask,

              max_font_size=2000,

              max_words=2000,

              random_state=42)

wcloud = wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(16, 10))

plt.axis('off')

plt.imshow(wc.recolor(color_func=img_color), interpolation="bilinear")

plt.show()
fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

n, bins, patches = plt.hist(books['average_rating'], bins=100, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer

for i in range(len(patches)):

    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))

plt.title('Average Rating Distribution', fontsize=20)

plt.xlabel('Average Rating', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.show()
def parse_year(date):

    if type(date) == type(float("nan")):

        return

    year_check = match(r'.*([1-3][0-9]{3})', date)

    if year_check != None:

        return int(year_check.group(1))



books['year_published'] = books["date_published"].apply(parse_year)

fig, ax = plt.subplots(1,1)

fig.set_size_inches(18.5, 10.5)

ax.tick_params(labelrotation=90)

n, bins, patches = ax.hist(books['year_published'].dropna(inplace=False), bins=250, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer

for i in range(len(patches)):

    patches[i].set_facecolor(plt.cm.viridis(n[i]/max(n)))



plt.show()