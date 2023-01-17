# Importing the necessary libraries



import pandas as pd

import numpy as np

import re

from wordcloud import WordCloud

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv("/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv")

print(data.shape)

data.head()
# Preprocessing the text data



REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;!]")

BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")

STOPWORDS_nlp = set(stopwords.words('english'))



#Custom Stoplist

stoplist = ["i","me","my","myself","we","our","ours","ourselves","you","you're","you've","you'll","you'd","your",

            "yours","yourself","yourselves","he","him","his","himself","she","she's","her","hers","herself","it",

            "it's","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","that'll",

            "these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did",

            "doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about",

            "against","between","into","through","during","before","after","above","below","to","from","up","down","in","out",

            "on","off","over","under","again","further","then","once","here","there","when","where","why","all","any",

            "both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too",

            "very","s","t","can","will","just","don","don't","should","should've","now","d","ll","m","o","re","ve","y","ain",

            "aren","couldn","didn","doesn","hadn","hasn",

            "haven","isn","ma","mightn","mustn","needn","shan","shan't",

            "shouldn","wasn","weren","won","rt","rt","qt","for",

            "the","with","in","of","and","its","it","this","i","have","has","would","could","you","a","an",

            "be","am","can","edushopper","will","to","on","is","by","ive","im","your","we","are","at","as","any","ebay","thank","hello","know",

            "need","want","look","hi","sorry","http", "https","body","dear","hello","hi","thanks","sir","tomorrow","sent","send","see","there","welcome","what","well","us"]



STOPWORDS_nlp.update(stoplist)



# Function to preprocess the text

def text_prepare(text):

    """

        text: a string

        

        return: modified initial string

    """

    text = text.replace("\d+"," ") # removing digits

    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #removing mentions and urls

    text = text.lower() # lowercase text

    text =  re.sub('[0-9]+', '', text)

    text = REPLACE_BY_SPACE_RE.sub(" ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text

    text = BAD_SYMBOLS_RE.sub(" ", text) # delete symbols which are in BAD_SYMBOLS_RE from text

    text = ' '.join([word for word in text.split() if word not in STOPWORDS_nlp]) # delete stopwors from text

    text = text.strip()

    return text



# Cleaning the "text" column in the data frame using the above defined function

df_text = data["text"].astype(str).apply(text_prepare)

df_text.head()
text = " ".join(sent for sent in df_text)

print("There are {} words in the text.".format(len(text)))
wordcloud = WordCloud(background_color = "white", height=1200, width= 1600, collocations=False , max_words= 100).generate(text)

plt.figure(figsize= (16,12))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# Creating a subset of sentence using the window method



text_tok = text.split()

indices = (i for i, word in enumerate(text_tok) if word == "symptom" or word=="symptoms")

neighbors=[]

for ind in indices:

    neighbors.append(text_tok[ind-6:ind]+ text_tok[ind:ind+6])



neighbors_sent=[]

for i in neighbors:

    sent = " ".join(i)

    neighbors_sent.append(sent)
# Unigram symptom cloud

wordcloud = WordCloud(stopwords = ["symptoms","symptom", "covid", "people", "include","said", "self", "common"], background_color = "white", height=1200, width= 1600, collocations=False , max_words= 100).generate(str(neighbors_sent))

plt.figure(figsize= (16,12))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# Preprocessing for Bigram cloud



bigram_corpus = str(neighbors_sent).split(",")



cv = CountVectorizer(max_features = 1000,ngram_range=(2,2), min_df=3)

X = cv.fit_transform(bigram_corpus)



bigram_df = pd.DataFrame(data = np.column_stack([cv.get_feature_names(),X.toarray().sum(axis=0)]))

bigram_df.tail()
# bigram_text = " ".join(i for i in bigram_df.iloc[:,0])

bigram_df.iloc[:,1] = bigram_df.iloc[:,1].astype(int)

bigram_df = bigram_df[~(bigram_df.iloc[:,0].str.contains("symptoms") | bigram_df.iloc[:,0].str.contains("covid") | bigram_df.iloc[:,0].str.contains("isolate") | bigram_df.iloc[:,0].str.contains("include") | bigram_df.iloc[:,0].str.contains("including"))]
bigram_dict = dict(zip(bigram_df.iloc[:,0], bigram_df.iloc[:,1]))
wordcloud = WordCloud(stopwords = ["symptoms","symptom", "covid", "people", "include","said", "self", "common"], background_color = "white", height=1200, width= 1600, collocations=False , max_words=70).generate_from_frequencies(bigram_dict)

plt.figure(figsize= (16,12))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()