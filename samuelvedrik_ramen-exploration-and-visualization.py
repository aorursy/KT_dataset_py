import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress



plt.rcParams["figure.figsize"] = (16, 9)

plt.style.use("ggplot")
FILE_PATH = "/kaggle/input/ramen-ratings-latest-update-jan-25-2020/Ramen_ratings_2020.csv"

df_ramen = pd.read_csv(FILE_PATH)
df_ramen.head(3)
df_ramen.dtypes
def is_float(x):

    """

    Returns true if x is of type float

    """

    # This is probably not the best way to do it. 

    try:

        return type(float(x)) == float

    except:

        return False



        

df_ramen[~df_ramen["Stars"].apply(is_float)]
# All the observations that had dual ratings were in the format of "noodle + broth",

# thus I searched for varieties that matched this format.

filter_ = df_ramen["Variety"].str.match(r".* \+ .*")

df_ramen.loc[filter_, "Stars"] = df_ramen.loc[filter_,"Stars"].apply(lambda x: np.mean(list(map(float, x.split("/")))))

df_ramen["Stars"] = pd.to_numeric(df_ramen["Stars"], errors="coerce")

df_ramen.dropna(inplace=True)
df_ramen.info()
df_ramen["Brand"].value_counts().sort_values().tail(10).plot.barh()

_ = plt.title("Top 10 Brands"), plt.xlabel("Count")
print("There are {} unique brands.".format(len(df_ramen["Brand"].unique())))
brand_country_count = (df_ramen.groupby("Brand")["Country"].nunique().sort_values(ascending=False) > 1).sum()

print(f"There are {brand_country_count} brands that are based in more than 1 country.")
df_ramen["Country"].value_counts().sort_values().plot.barh()

_ = plt.title("Country Distribution"), plt.xlabel("Count")
sns.distplot(df_ramen["Stars"], bins=20)

_ = plt.title("Distribution of ratings")
df_brand_ramen_ratings = df_ramen.groupby("Brand")["Stars"].mean().sort_index()

df_brand_ramen_count = df_ramen["Brand"].value_counts().sort_index()
plt.scatter(np.log10(df_brand_ramen_count), df_brand_ramen_ratings)

# We use the log scale since Nissin has released so many ramen varieties compared to other brands. 



m, b, r, *_ = linregress(np.log10(df_brand_ramen_count), df_brand_ramen_ratings)

X = np.linspace(0, 3)

plt.plot(X, m*X + b)



_ = plt.title("Ramen Release Count Against Average Ramen Rating for each Brand"), \

plt.ylabel("Rating"), plt.xlabel("Count (log scale)")
print(f"The correlation coefficient is {r}")
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import wordnet

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import re



def nltk_tag_to_wordnet_tag(nltk_tag):

    if nltk_tag.startswith('J'):

        return wordnet.ADJ

    elif nltk_tag.startswith('V'):

        return wordnet.VERB

    elif nltk_tag.startswith('N'):

        return wordnet.NOUN

    elif nltk_tag.startswith('R'):

        return wordnet.ADV

    else:          

        return None



lem = WordNetLemmatizer()



def tokenize(x):

    clean = re.sub("[^\w\s]", " ", x) # Get rid of punctuation and newline.

    token = word_tokenize(clean)

    pos = nltk.pos_tag(token)

    res = [(word, nltk_tag_to_wordnet_tag(tag)) for word, tag in pos]

    return [lem.lemmatize(word, pos=tag) for word, tag in res if tag is not None]
# Cleaning the text a little 



df_ramen["Variety"] = df_ramen["Variety"].apply(lambda x : x.lower())

df_ramen["Variety"] = df_ramen["Variety"].str.replace("flavour", "flavor")



# Tokenizing

tokenized_variety = df_ramen["Variety"].apply(tokenize)
words_dict = {}

for token in tokenized_variety:

    for word in token:

        if word not in words_dict:

            words_dict[word] = 0

        words_dict[word] += 1

        

df_words = pd.DataFrame.from_dict(words_dict, orient="index").reset_index().rename(columns={"index": "Word", 0: "Count"})
df_words.sort_values("Count", ascending=False).head(10)
from wordcloud import WordCloud



plt.imshow(WordCloud(width=600, height=600).generate_from_frequencies(words_dict), interpolation='bilinear')

_ = plt.axis("off")