def get_dtype_lists(data,features):

    output = {}

    for f in features:

        dtype = str(data[f].dtype)

        if dtype not in output.keys(): output[dtype] = [f]

        else: output[dtype] += [f]

    return output



def show_uniques(data,features):

    for f in features:

        if len(data[f].unique()) < 30:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()))

        else:

            print("%s: count(%s) %s" % (f,len(data[f].unique()),data[f].unique()[0:10]))



def show_all_uniques(data,features):

    dtypes = get_dtype_lists(data,features)

    for key in dtypes.keys():

        print(key + "\n")

        show_uniques(data,dtypes[key])

        print()
from pandas import read_csv

data = read_csv("../input/survey.csv")
data.head()
data.drop("Timestamp",1, inplace=True)
dtype = get_dtype_lists(data, data.columns)
numerics = dtype["int64"]
categories = dtype["object"]
for category in categories: data[category] = data[category].apply(str)
data[data.comments != "nan"]["comments"].head()
comments = data[data.comments != "nan"]["comments"]
full_text = " ".join(comments.values)
from nltk import word_tokenize, bigrams, trigrams, FreqDist, pos_tag

from nltk.corpus import stopwords
def get_bigram_frequencies(text):

    tokens = word_tokenize(text)

    return FreqDist(bigrams(tokens))



def get_trigram_frequencies(text):

    tokens = word_tokenize(text)

    return FreqDist(trigrams(tokens))



def get_keyword_frequencies(text):

    tokens = [word for word in word_tokenize(text) if word.isalnum() and word not in stopwords.words("english")]

    return FreqDist(tokens)



def get_content_words(text):

    chosen_pos = ["NOUN", "ADJ", "ADV", "VERB", "."]

    tokens = word_tokenize(text)

    tagged = pos_tag(tokens, tagset="universal")

    words = [word for (word,tag) in tagged if tag in chosen_pos]

    return words



def get_words_by_grammar(text,chosen_pos):



    tokens = word_tokenize(text)

    tagged = pos_tag(tokens, tagset="universal")

    words = [word for (word,tag) in tagged if tag in chosen_pos]

    return words



# Write a word list by tag returnable.
len(data)
get_keyword_frequencies(full_text).most_common(60)
get_bigram_frequencies(full_text).most_common(60)
get_trigram_frequencies(full_text).most_common(60)
content_words = " ".join(get_content_words(full_text))

noun_phrases = " ".join(get_words_by_grammar(full_text,["NOUN","ADJ",".","PRONOUN"]))

verb_phrases = " ".join(get_words_by_grammar(full_text,["VERB","ADV",".","ADP"]))
get_keyword_frequencies(content_words).most_common(60)
get_bigram_frequencies(noun_phrases).most_common(60)
get_bigram_frequencies(verb_phrases).most_common(60)