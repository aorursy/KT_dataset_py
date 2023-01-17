%matplotlib inline
%reload_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
!ls ../input/
data_df = pd.read_csv('../input/all_annotated.tsv', sep="\t")
data_df.head()
import re
#https://stackoverflow.com/questions/3868753/find-phone-numbers-in-python-script
phone_re = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
#Match the following patterns
# 000-000-0000
# 000 000 0000
# 000.000.0000

# (000)000-0000
# (000)000 0000
# (000)000.0000
# (000) 000-0000
# (000) 000 0000
# (000) 000.0000

# 000-0000
# 000 0000
# 000.0000

# 0000000
# 0000000000
# (000)0000000
phone_nums = []
for num_list in data_df['Tweet'].apply(lambda x: phone_re.findall(x)):
    if len(num_list) > 0:
        phone_nums += num_list
print("There are {0} phone numbers in the dataset".format(len(phone_nums)))
phone_nums
import langid 
guessed_langs = data_df['Tweet'].apply(langid.classify)
langs = guessed_langs.apply(lambda tuple: tuple[0])
langs[:5]
guess_en_labels = langs.apply(lambda x: 1 if x == 'en' else 0)
guess_en_labels = guess_en_labels.values

en_labels = data_df['Definitely English'].values
from sklearn.metrics import accuracy_score
y_pred = guess_en_labels
y_true = en_labels
print("Accuracy of guessed language using langid is {0}".format(accuracy_score(y_true, y_pred)))
print("Number of unique languages:")
print(len(langs.unique()))
print("")

print("Number of data in English:")
print(sum(langs=="en"))
print("")

print("Percent of data in English:")
print((sum(langs=="en")/len(langs))*100)
print("")

print("Number of data in Thai:")
print((sum(langs=="th")))
print("")

print("Percent of data in Thai:")
print((sum(langs=="th")/len(langs))*100)
langs_df = pd.DataFrame(langs)
langs_count = langs_df.Tweet.value_counts()
langs_count.plot.bar(figsize=(20,10), fontsize=20)
data_eng_df = data_df[data_df["Definitely English"] == 1]
data_eng_df["Definitely English"].unique()
data_eng_df.head(2)
print("There are {0} row left".format(len(data_eng_df)))
import spacy
nlp = spacy.load('en')
def extract_propn_token(doc):
    doc_nlp = nlp(doc)
    propn_tokens = []
    for token in doc_nlp:
        if token.pos_ == "PROPN":
            propn_tokens.append(token.text)
    return propn_tokens
data_eng_df.loc[:,'Tweet_propn'] = data_eng_df['Tweet'].apply(extract_propn_token)
data_eng_df.loc[:,'Tweet_propn'] = data_eng_df['Tweet_propn'].apply(lambda x: " ".join(x) if len(x) > 0 else "")
propn_tokens = []
for item in data_eng_df['Tweet_propn']:
    if item != '':
        propn_tokens += item.split()
print("There are {0} unique words detected as proper nouns".format(len(set(propn_tokens))))
#print("All of proper nouns tokens are: \n{0}".format(propn_tokens))
import re

# text cleaning rules are modified from https://www.kaggle.com/hubert0527/spacy-name-entity-recognition
def clean_string(text):
    SPECIAL_TOKENS = {
        'quoted': 'quoted_item',
        'non-ascii': '',#'non_ascii_word',
        'undefined': 'something'
    }
    def pad_str(s):
        return ' '+s+' '
    
    # Empty text
    
    if type(text) != str or text=='':
        return ''
    
    # preventing first and last word being ignored by regex
    # and convert first word in question to lower case
    text = ' ' + text[0].lower() + text[1:] + ' '
    
    # replace all first char after either [.!?)"'] with lowercase
    # don't mind if we lowered a proper noun, it won't be a big problem
    
    def lower_first_char(pattern):
        matched_string = pattern.group(0)
        return matched_string[:-1] + matched_string[-1].lower()
    
    text = re.sub("(?<=[\.\?\)\!\'\"])[\s]*.",lower_first_char , text)
    
    # Replace weird chars in text
    
    text = re.sub("’", "'", text) # special single quote
    text = re.sub("`", "'", text) # special single quote
    text = re.sub("“", '"', text) # special double quote
    text = re.sub("？", "?", text) 
    text = re.sub("…", " ", text) 
    text = re.sub("é", "e", text) 
    
    # Clean shorthands
    
    text = re.sub("\'s", " ", text) # we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable, I choose to compromise are kill "'s" directly
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("b\.g\.", " bg ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\W|^)([0-9]+)[kK](\W|$)", r"\1\g<2>000\3", text) # better regex provided by @armamut
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    # add padding to punctuations and special chars, we still need them later
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    
    def pad_pattern(pattern):
        matched_string = pattern.group(0)
        return pad_str(matched_string)
    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text) 
        
    text = re.sub('[^\x00-\x7F]+', pad_str(SPECIAL_TOKENS['non-ascii']), text) # replace non-ascii word with special word
    
    # indian dollar
    text = re.sub("(?<=[0-9])rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(" rs(?=[0-9])", " rs ", text, flags=re.IGNORECASE)
    
    # clean text rules get from : https://www.kaggle.com/currie32/the-importance-of-cleaning-text
    
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text) 
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)  
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE) 
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE) 
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)
    
    # typos 
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)  
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" undergraduation ", " undergraduate ", text) # not typo, but GloVe can't find it
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)  
    text = re.sub(r" begineer ", " beginner ", text)  
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)  
    text = re.sub(r" litrate ", " literate ", text)  
    
    #custom
    text = re.sub(r" https ", " http ", text)  
    text = re.sub(r'(.)\1{2,}', r'\1', text) # buuuuuttttt -> but
      
    # for words like A-B-C-D or "A B C D", 
    # if A,B,C,D individuaally has vector in glove:
    #     it can be treat as separate words
    # else:
    #     replace it as a special word, A_B_C_D is enough, we'll deal with that word later
    #
    # Testcase: 'a 3-year-old 4 -tier car'
    
    def dash_dealer(pattern):
        matched_string = pattern.group(0)
        splited = matched_string.split('-')
        splited = [sp.strip() for sp in splited if sp!=' ' and sp!='']
        joined = ' '.join(splited)
        parsed = nlp(joined)
        for token in parsed:
            # if one of the token is not common word, then join the word into one single word
            if not token.has_vector or token.text in SPECIAL_TOKENS.values():
                return '_'.join(splited)
        # if all tokens are common words, then split them
        return joined

    text = re.sub("[a-zA-Z0-9\-]*-[a-zA-Z0-9\-]*", dash_dealer, text)
    
    # try to see if sentence between quotes is meaningful
    # rule:
    #     if exist at least one word is "not number" and "length longer than 2" and "it can be identified by SpaCy":
    #         then consider the string is meaningful
    #     else:
    #         replace the string with a special word, i.e. quoted_item
    # Testcase:
    # i am a good (programmer)      -> i am a good programmer
    # i am a good (programmererer)  -> i am a good quoted_item
    # i am "i am a"                 -> i am quoted_item
    # i am "i am a programmer"      -> i am i am a programmer
    # i am "i am a programmererer"  -> i am quoted_item
    
    def quoted_string_parser(pattern):
        string = pattern.group(0)
        parsed = nlp(string[1:-1])
        is_meaningful = False
        for token in parsed:
            # if one of the token is meaningful, we'll consider the full string is meaningful
            if len(token.text)>2 and not token.text.isdigit() and token.has_vector:
                is_meaningful = True
            elif token.text in SPECIAL_TOKENS.values():
                is_meaningful = True
            
        if is_meaningful:
            return string
        else:
            return pad_str(string[0]) + SPECIAL_TOKENS['quoted'] + pad_str(string[-1])

    text = re.sub('\".*\"', quoted_string_parser, text)
    text = re.sub("\'.*\'", quoted_string_parser, text)
    text = re.sub("\(.*\)", quoted_string_parser, text)
    text = re.sub("\[.*\]", quoted_string_parser, text)
    text = re.sub("\{.*\}", quoted_string_parser, text)
    text = re.sub("\<.*\>", quoted_string_parser, text)

    text = re.sub('[\(\)\[\]\{\}\<\>\'\"]', pad_pattern, text) 
    
    # the single 's' in this stage is 99% of not clean text, just kill it
    text = re.sub(' s ', " ", text)
    
    # reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    text = text.strip()
    
    return text
def filter_stopword(doc):
    doc_nlp = nlp(doc)
    results = []
    for token in doc_nlp:
        if not token.is_stop: # not stop word
            results.append(token.text)
    return " ".join(results)
data_eng_df.loc[:,'Tweet_propn_clean'] = data_eng_df["Tweet_propn"].apply(clean_string)
data_eng_df.loc[:,'Tweet_propn_clean'] = data_eng_df["Tweet_propn_clean"].apply(filter_stopword)
data_eng_df.tail()
def lemmatizer(doc):        
    sent = []
    doc_nlp = nlp(doc)
    for token in doc_nlp:
        sent.append(token.lemma_)
    return " ".join(sent)
data_eng_df.loc[:,'Tweet_propn_clean_lemma'] = data_eng_df["Tweet_propn_clean"].apply(lemmatizer)
cleaned_data_eng_df = data_eng_df[data_eng_df['Tweet_propn_clean_lemma'] != ""]
cleaned_data_eng_df.head()
propn_clean_lemma_tokens = []
for item in cleaned_data_eng_df['Tweet_propn_clean_lemma']:
    if item != '':
        propn_clean_lemma_tokens += item.split()
print("There are {0} remaining unique words after lemmatization process".format(len(set(propn_clean_lemma_tokens))))
from sklearn.feature_extraction.text import TfidfVectorizer
cleaned_texts = cleaned_data_eng_df['Tweet_propn_clean_lemma'].values


tfidf_vectorizer = TfidfVectorizer(
    min_df=1, max_features=None, strip_accents='unicode', lowercase=True,
    analyzer='word', ngram_range=(1, 1), use_idf=True, 
    smooth_idf=True, sublinear_tf=True
)

tfidf_vectorizer.fit(cleaned_texts)
cleaned_texts_tfidf = tfidf_vectorizer.transform(cleaned_texts)
vocab_frame = pd.DataFrame({'words': list(tfidf_vectorizer.vocabulary_.keys())}, index = tfidf_vectorizer.vocabulary_.keys())
vocab_frame.head()
print("Number of vocabulary of tfidf_vectorizer is {0} words".format(len(tfidf_vectorizer.vocabulary_)))
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
# number of clusters
n_clusters = 5

# fit k-mean clustering
kmeans = KMeans(n_clusters=n_clusters, random_state = 0)

kmeans.fit_predict(cleaned_texts_tfidf)
clusters = kmeans.labels_.tolist()
cleaned_data_eng_df.loc[:, 'clusters'] = clusters
cleaned_data_eng_df['clusters'].value_counts() 
cleaned_data_eng_df.head()
cleaned_data_eng_df.index = cleaned_data_eng_df.clusters.values
print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1] 

for i in range(n_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :15]: # n words per cluster
        print(' %s' % vocab_frame.loc[tfidf_vectorizer.get_feature_names()[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print("\n")
    
    print("Cluster %d Country:" % i, end='')
    for country in cleaned_data_eng_df.loc[i]['Country'].unique():
        print(' %s,' % country, end='')
    print("\n")
    
    print("Cluster %d Automatically Generated Tweets:" % i, end='')
    for country in cleaned_data_eng_df.loc[i]['Automatically Generated Tweets'].unique():
        print(' %s,' % country, end='')
    print("\n")
    
    print("\n\n=========")
print("\n\n")
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(cleaned_texts_tfidf)
from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print("\n\n")
#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names using a dict
cluster_names = {0: 'rkiye, stanbul, cafe', 
                 1: 'rt, mastermind, ak', 
                 2: 'co, http, job', 
                 3: 'job, careerarc, sta', 
                 4: 'kuala, lumpur, wp'}
#create data frame that has the result of the MDS plus the cluster numbers and country
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, country=cleaned_data_eng_df['Country'].values)) 

#group by cluster
groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(25, 18)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['country'], size=8)  

plt.show() #show the plot
