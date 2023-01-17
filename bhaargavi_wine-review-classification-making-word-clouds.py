import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline
import nltk
# Importing Natural language Processing toolkit.
from PIL import Image
# from python imaging library
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wines =  pd.read_csv("../input/wine130k/winemag-data-130k-v2.csv",index_col = 0)
df_wines = wines[['country', 'description', 'points', 'price', 'variety']]
df_wines = df_wines.sample(frac = 0.05)
print(df_wines.shape)
df_wines.head()
# Firstly let us count the number of words in all in the description texts available to us in our sample data
text = " ".join(review for review in df_wines.description)
print ("There are {} words in the combination of all review.".format(len(text)))
# N0w tokenizing the text using TreebankWordTokenizer
tokenizer1 = nltk.tokenize.TreebankWordTokenizer()
tokens = tokenizer1.tokenize(text)
tokens[0:10]
# NOw normalization of the text using lemmatization
stemmer = nltk.stem.WordNetLemmatizer()
words_all = " ".join(stemmer.lemmatize(token) for token in tokens)
words_all[0:100]                     
stopwords = set(STOPWORDS)
stopwords.update(["drink" , 'now', 'wine' ,'flavour'])
wordcloud = WordCloud(stopwords = stopwords, background_color = "white").generate(words_all)
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()
wine_mask = np.array(Image.open('../input/images/wine.jpg'))
def transform_mask(val):
    # For black color inside the wine bottle and glass making it white
    if val == 0:
        return 255
    else:
        return val
transformed_wine_mask = np.ndarray((wine_mask.shape[0], wine_mask.shape[1]))
for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_mask, wine_mask[i]))
wc = WordCloud(background_color="white", max_words=1000, mask=transformed_wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick', max_font_size = 34).generate(words_all)

# show
plt.figure(figsize=[20,10])
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
def create_color_comb(country, img):
    country = " ".join(review for review in  wines[wines['country'] == country].description)
    mask = np.array(Image.open("../input/images/"+img))
    wordcloud_country = WordCloud(stopwords = STOPWORDS, background_color = 'white',contour_color='firebrick',
                                  max_words = 1000, mask = mask).generate(country)
    image_colors = ImageColorGenerator(mask)
    plt.figure(figsize=[5,8])
    plt.imshow(wordcloud_country.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
create_color_comb("India", "india.png")
create_color_comb("US", "usa.jpg")
