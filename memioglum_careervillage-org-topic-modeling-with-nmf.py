!pip install ekphrasis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS 
from wordcloud import WordCloud
from ekphrasis.classes.segmenter import Segmenter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import gensim
from collections import Counter

pd.set_option('display.max_columns', 80)
%matplotlib inline
nltk.download("punkt")
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem.snowball import SnowballStemmer
nltk.download("brown")
nltk.download('averaged_perceptron_tagger')
questions = pd.read_csv("../input/data-science-for-good-careervillage/questions.csv")
# Questions get posted by students. Sometimes they're very advanced. Sometimes they're just getting started.
# It's all fair game, as long as it'"s relevant to the student's future professional success.
tag_questions = pd.read_csv("../input/data-science-for-good-careervillage/tag_questions.csv")
# Every question can be hashtagged. We track the hashtag-to-question pairings, and put them into this file.
tags = pd.read_csv("../input/data-science-for-good-careervillage/tags.csv")
# Each tag gets a name.
question_scores = pd.read_csv("../input/data-science-for-good-careervillage/question_scores.csv")
# "Hearts" scores for each question.

# Data is merged.
questions_all = pd.merge(questions, tag_questions, left_on="questions_id", right_on="tag_questions_question_id").drop("tag_questions_question_id", axis="columns")
questions_all = pd.merge(questions_all, tags, left_on="tag_questions_tag_id", right_on="tags_tag_id").drop(["tag_questions_tag_id", "tags_tag_id"], axis="columns")
questions_all = pd.merge(questions_all, question_scores, left_on="questions_id", right_on="id").drop("questions_id", axis="columns")

question_tags= questions_all.groupby("id")["tags_tag_name"].unique()
questions_all= pd.merge(questions_all, question_tags.to_frame(), left_on="id", right_index=True)
questions_all.drop_duplicates(subset=["id"], inplace=True)
questions_all.drop("tags_tag_name_x", axis="columns", inplace=True)
questions_all.rename(columns={"tags_tag_name_y": "tag_name"}, inplace=True)
print(questions_all.shape)
questions_all.head()
seg = Segmenter(corpus="english")
def tag_extender(tags):
    from_hyphen = [y for x in tags for y in x.split("-")]
    from_underscore = [y for x in from_hyphen for y in x.split("_")]
    from_hashtag = [seg.segment(tag) for tag in from_underscore]
    remove_numbers = re.sub(r'[0-9]+', '', " ".join(from_hashtag))
    create_list = [x for x in remove_numbers.split(" ")]
    final_list = [x for x in create_list if len(x)>0]
    return final_list

questions_all["extended_tags"] = questions_all["tag_name"].apply(tag_extender)
questions_all.tail().iloc[:,[6,7]]
# Unwanted characters are removed from question bodies.
questions_all.reset_index(drop=True, inplace=True)
questions_all["questions_body"] = questions_all["questions_body"].apply(lambda x: re.compile(r"[\n\r\t]").sub(" ", x))
questions_all["questions_body"] = questions_all["questions_body"].apply(lambda x: re.sub(r"(#\S*)", "", x))
# Merging stopwords
nltk_stopwords= set(stopwords.words('english'))
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
total_stopwords=nltk_stopwords|spacy_stopwords
len(total_stopwords)
all_words_body = (' '.join(questions_all["questions_body"]).lower().split())
no_sw_body = [word for word in all_words_body if not word in total_stopwords]
most_common_body_sw = Counter(no_sw_body).most_common(30)

add_stopwords=[]

for i in range(30):
    add_stopwords.append(most_common_body_sw[i][0])
    
add_stopwords
wikipedia_stopwords_nouns = ["time", "person", "year", "way", "day", "thing", "man", "world", "life", "hand", "part", "child",
                             "eye", "woman", "place", "work", "week", "case", "point", "government", "company", "number",
                             "group", "problem", "fact"]

# https://en.wikipedia.org/wiki/Most_common_words_in_English#Nouns
# 25 most common nouns in English language listed on Wikipedia.
wikipedia_stopwords_verbs = ["be", "have", "do", "say", "get", "make", "go", "know", "take", "see", "come", "think", "look",
                             "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave", "call"]

# https://en.wikipedia.org/wiki/Most_common_words_in_English#Verbs
# 25 most common verbs in English language listed on Wikipedia.
manual_stopwords = ["question", "answer", "class", "study", "graduate", "grad", "interest", "university", "undecided", "decide", "decision", "manage"]

# Generic words used frequently in the texts due to the nature of the topic.
# Combining all the stopwords

added_total_stopwords=total_stopwords|set(add_stopwords)|set(wikipedia_stopwords_nouns)|set(wikipedia_stopwords_verbs)|set(manual_stopwords)
len(added_total_stopwords)
# Parameters
min_len = 3
stemmer = SnowballStemmer("english")
not_to_stem = ["animation", "animator"]
###

def process_text(text):
    # Make all the strings lowercase and remove non alphabetic characters
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    # Tokenizing the text
    tokenized_text = word_tokenize(text)
    
    # Taking only nouns, adjectives and verbs
    is_noun_adj_verb = lambda pos: pos[:2] == "NN" or pos[:2] == "JJ" or pos[:2] == "VB"
    noun_adj_verb = [(word, pos) for (word, pos) in nltk.pos_tag(tokenized_text) if is_noun_adj_verb(pos)]
    
    # Lemmatizing the tokens according to their PoS tags
    lemmatized_text = []
    for word, tag in noun_adj_verb:
        if tag.startswith("NN"):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='a'))
        else:
            None
    
    # Removing tokens that are shorter than 3 characters
    longer_words = [word for word in lemmatized_text if len(word)>=min_len]
    
    # Removing stopwords
    remove_stopwords = [word for word in longer_words if not word in added_total_stopwords]
    
    # Stemming with Snowball English
    stemmed_words = [stemmer.stem(word) if word not in not_to_stem else word for word in remove_stopwords]

    return stemmed_words

# Processing question titles and questions

questions_all["title_tokens"]=questions_all["questions_title"].apply(process_text)
questions_all["body_tokens"]=questions_all["questions_body"].apply(process_text)
# Parameters
min_len = 3
stemmer = SnowballStemmer("english")
not_to_stem = ["animation", "animator"]
###

def process_tags(text):
    is_noun_adj_verb = lambda pos: pos[:2] == "NN" or pos[:2] == "JJ" or pos[:2] == "VB"
    noun_adj_verb = [(word, pos) for (word, pos) in nltk.pos_tag(text) if is_noun_adj_verb(pos)]
    
    # Lemmatizing the tokens according to their PoS tags
    lemmatized_text = []
    for word, tag in noun_adj_verb:
        if tag.startswith("NN"):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemmatized_text.append(lemmatizer.lemmatize(word, pos='a'))
        else:
            None
    longer_words = [word for word in lemmatized_text if len(word)>=min_len]
    remove_stopwords = [word for word in longer_words if not word in added_total_stopwords]
    stemmed_words = [stemmer.stem(word) if word not in not_to_stem else word for word in remove_stopwords]
    return stemmed_words

# Processing tags

questions_all["tag_tokens"] = questions_all["extended_tags"].apply(process_tags)
questions_all["tag_len"] = questions_all["tag_tokens"].apply(lambda x: len(x))
questions_all.sort_values("tag_len", ascending=False).head().iloc[:, [2,3, -6, -1]]
questions_all["tag_len"].plot(kind="box", figsize=(10,10), grid=True);
questions_all[questions_all["tag_len"]>30]["tag_tokens"].apply(lambda x: x.clear())
questions_all.drop("tag_len", axis=1, inplace=True)
questions_all["question_tokens"]=questions_all["title_tokens"]+questions_all["body_tokens"]+questions_all["tag_tokens"]
questions_all.head()
def as_it_is(word):
    return word

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, tokenizer=as_it_is, preprocessor=as_it_is)
tfidf = tfidf_vectorizer.fit_transform(questions_all["question_tokens"])
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
len(tfidf_feature_names)
kmin, kmax = 15, 40

topic_models = []

for k in range(kmin,kmax+1):
    model = NMF( init="nndsvd", n_components=k, random_state=10, alpha=0.1, l1_ratio=.5 ).fit(tfidf)
    W = model.fit_transform(tfidf)
    H = model.components_    
    topic_models.append( (k,W,H) )
import gensim
all_tokens = questions_all["question_tokens"].apply(lambda x: " ".join(x))
all_tokens = (("").join(all_tokens).split())
w2v_model = gensim.models.Word2Vec([all_tokens], min_count=2, sg=1)
# https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb

def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            pair_scores.append( w2v_model.similarity(pair[0], pair[1]) )
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

from itertools import combinations
k_values = []
coherences = []
for (k,W,H) in topic_models:
    # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append( get_descriptor( tfidf_feature_names, H, topic_index, 10 ) )
    # Now calculate the coherence based on our Word2vec model
    k_values.append( k )
    coherences.append( calculate_coherence( w2v_model, term_rankings ) )
plt.plot(k_values, coherences)
plt.grid(True)
plt.xlabel("Number of topics", fontsize=14, weight="bold", labelpad = 20)
plt.ylabel("Coherence score", fontsize = 14, weight="bold", labelpad=15)
plt.title("Change in coherence score with number of topics", fontsize = 17, weight="bold", pad=10)
plt.xticks(np.arange(kmin,kmax+1,1))
fig = plt.gcf()
fig.set_size_inches(17,4)
plt.show()
model = NMF(n_components=26, random_state=10, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

W = model.fit_transform(tfidf)
H = model.components_ 
def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)
no_top_words = 15
topic_weight = display_topics(model, tfidf_feature_names, no_top_words)
topic_weight
topic_dict_wc = {
    0: "Engineering",
    1: "Nursing",
    2: "Business adm. and management",
    3: "Computer science",
    4: "Medicine",
    5: "Financing studies",
    6: "Psychology and psychiatry",
    7: "Law",
    8: "Applied arts",
    9: "Teaching",
    10: "Gaining work experience",
    11: "Physical therapy",
    12: "Accounting and finance",
    13: "Biosciences",
    14: "Sports",
    15: "Performing arts",
    16: "Budgetary issues",
    17: "Budgetary issues",
    18: "Medicine",
    19: "Game development",
    20: "Career path",
    21: "Veterinary med. and zoology",
    22: "Arts",
    23: "Science",
    24: "Social work",
    25: "Computer science"
}

fig = plt.figure(figsize=(20,25))
a=1
for col in np.arange(0, 52, 2):
    col_w = col + 1
    words=list()
    weights=list()
    for i in range(15):
        words.append(topic_weight.iloc[i, col])
        weights.append(float(topic_weight.iloc[i, col_w]))
    temp = zip(words, weights)
    dictWords = dict(temp)
    wordcloud = WordCloud(width=500,height=300, background_color="white", min_font_size=8).generate_from_frequencies(dictWords)
    ax = fig.add_subplot(7,4,a)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title("Topic Nr: "+str(int(col/2))+", " + str(topic_dict_wc[col/2]), fontsize=14, weight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor('#666666')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    a=a+1

plt.show();
topic_dict = {
    0: "Engineering",
    1: "Nursing",
    2: "Business administration and management",
    3: "Computer science",
    4: "Medicine",
    5: "Financing studies",
    6: "Psychology and psychiatry",
    7: "Law",
    8: "Applied arts",
    9: "Teaching",
    10: "Gaining work experience",
    11: "Physical therapy",
    12: "Accounting and finance",
    13: "Biosciences",
    14: "Sports",
    15: "Performing arts",
    16: "Budgetary issues",
    17: "Budgetary issues",
    18: "Medicine",
    19: "Game development",
    20: "Career path",
    21: "Veterinary medicine and zoology",
    22: "Arts",
    23: "Science",
    24: "Social work",
    25: "Computer science"
}
#Labeling the questions with their dominant topics

questions_all["topic"] = questions_all.reset_index()["index"].apply(lambda x: topic_dict[W[x,:].argmax()])
questions_all.iloc[:, [2,3,6,-1]].head()
topic_shortened = {
    "Engineering": "Engineering",
    "Nursing": "Nursing",
    "Business administration and management": "Buss. adm. & mng",
    "Computer science": "Comp. science",
    "Medicine": "Medicine",
    "Financing studies": "Financing stud.",
    "Psychology and psychiatry": "Psych. & Psychiatry",
    "Law": "Law",
    "Applied arts": "Appl. arts",
    "Teaching": "Teaching",
    "Gaining work experience": "Gain. work exp.",
    "Physical therapy": "Phys. therapy",
    "Accounting and finance": "Acc. & finance",
    "Biosciences": "Biosciences",
    "Sports": "Sports",
    "Performing arts": "Perf. arts",
    "Budgetary issues": "Budget. iss.",
    "Game development": "Game dev.",
    "Career path": "Career path",
    "Veterinary medicine and zoology": "Vet. med. & zool.",
    "Arts": "Arts",
    "Science": "Science",
    "Social work": "Social work"
}
topic_question_count = questions_all.groupby("topic").size().sort_values(ascending=False).tolist()
topic_question_count_x = questions_all.groupby("topic").size().sort_values(ascending=False).index.tolist()

topic_perc = []
a=0
for count in topic_question_count:
    topic_perc.append(count/sum(topic_question_count)*100 + a)
    a = count/sum(topic_question_count)*100 + a
    
short_topics = [topic_shortened[topic] for topic in topic_question_count_x]

fig, ax1 = plt.subplots()
ax1.bar(short_topics, topic_question_count, color="#7aa428")
ax1.set_xlabel("Topics", labelpad=15, weight="bold", fontsize=19)
ax1.set_ylabel("Number of questions", labelpad=17, weight="bold", fontsize=17)
ax1.tick_params("x", rotation = 50, labelsize=14)
plt.setp(ax1.get_xticklabels(), ha="right")
ax1.tick_params("y", labelsize=15)
ax1.set_yticks(np.arange(0,6000, 1000).tolist())

ax2 = ax1.twinx()
ax2.plot(topic_perc, marker="o")
ax2.set_ylabel("% of questions", labelpad = 25, weight="bold", fontsize=17, rotation=270)
ax2.tick_params("y", labelsize=15)
ax2.set_yticks(np.arange(0,110,10).tolist())

plt.title("Question count per topic", fontsize=20, weight="bold", pad =10)
plt.grid(True, axis="y")
fig=plt.gcf()
fig.set_size_inches(20, 10)
plt.show()
answers = pd.read_csv("../input/data-science-for-good-careervillage/answers.csv")
answers.dropna(axis=0, inplace=True)
pf_answer_topic = answers[["answers_id", "answers_question_id"]].merge(questions_all[["id", "topic"]], left_on="answers_question_id", right_on ="id").drop("answers_question_id", axis=1)
topic_answer_count = pf_answer_topic.groupby("topic").size().sort_values(ascending=False).tolist()
topic_answer_count_x = pf_answer_topic.groupby("topic").size().sort_values(ascending=False).index.tolist()

topic_perc = []
a=0
for count in topic_answer_count:
    topic_perc.append(count/sum(topic_answer_count)*100 + a)
    a = count/sum(topic_answer_count)*100 + a
    
short_topics = [topic_shortened[topic] for topic in topic_answer_count_x]

fig, ax1 = plt.subplots()
ax1.bar(short_topics, topic_answer_count, color="#ffdb0d")
ax1.set_xlabel("Topics", labelpad=15, weight="bold", fontsize=19)
ax1.set_ylabel("Number of answers", labelpad=17, weight="bold", fontsize=17)
ax1.tick_params("x", rotation = 50, labelsize=14)
plt.setp(ax1.get_xticklabels(), ha="right")
ax1.tick_params("y", labelsize=15)
#ax1.set_yticks(np.arange(0,6000, 1000).tolist())

ax2 = ax1.twinx()
ax2.plot(topic_perc, marker="o")
ax2.set_ylabel("% of answers", labelpad = 25, weight="bold", fontsize=17, rotation=270)
ax2.tick_params("y", labelsize=15)
ax2.set_yticks(np.arange(0,110,10).tolist())

plt.title("Answer count per topic", fontsize=20, weight="bold", pad =10)
plt.grid(True, axis="y")
fig=plt.gcf()
fig.set_size_inches(20, 10)
plt.show()
topic_mean_score = questions_all.groupby("topic")["score"].mean().sort_values(ascending=False)
short_topics = [topic_shortened[x] for x in topic_mean_score.index.tolist()]

plt.bar(short_topics, topic_mean_score)
plt.title("Average question scores per topic", pad=15, fontsize=17, weight="bold")
plt.xlabel("Topics", fontsize = 15, weight="bold", labelpad=10)
plt.ylabel("Average score", fontsize=15, weight="bold", labelpad=10)
plt.xticks(short_topics, rotation=50, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, axis="y")
mean_line=questions_all["score"].mean()
plt.axhline(mean_line, label="mean", color="#ff279d")
plt.legend(loc="best")
fig=plt.gcf()
fig.set_size_inches(15,7)
answers = pd.read_csv("../input/data-science-for-good-careervillage/answers.csv")
answers_col = ["answer_id", "author_id", "question_id", "a_date", "answer"]
answers.columns = answers_col
pf = pd.read_csv("../input/data-science-for-good-careervillage/professionals.csv")
pf_col = ["pf_id", "pf_loc", "pf_ind", "pf_hl", "pf_date"]
pf.columns = pf_col

pf_answers = pd.merge(answers, pf, left_on="author_id", right_on="pf_id", how="inner").drop("author_id", axis=1)
pf_answers_questions = pd.merge(pf_answers, questions_all, left_on="question_id", right_on="id").drop("id", axis=1)

pf_answers = pf_answers_questions[[ "topic", "questions_title", "questions_body", "extended_tags", "a_date", "answer", "pf_id", "pf_loc", "pf_ind", "pf_hl", "answer_id", "question_id"]]
pf_answers.columns = ["topic", "title", "question", "tags", "answer_date", "answer", "pf_id", "pf_loc", "pf_ind", "pf_hl", "a_id", "question_id"]
print("Out of %d professionals listed in the dataset, %d of them answered questions." % (pf.shape[0], pf_answers["pf_id"].unique().shape[0]))
print("There are %d answers in total, written by professionals listed in tha dataset." % (pf_answers.shape[0]))
#Most answered topics of professionals are extracted.

pf_id_list = pf_answers["pf_id"].unique().tolist()
pf_answered_topics = pd.DataFrame()
pf_group = pf_answers.groupby(["pf_id", "topic"]).size().sort_values(ascending=False)

for id_nr in pf_id_list:
    topic_count = pf_group.loc[id_nr]
    topic_first = topic_count.index[0]
    pf_answered_topics = pf_answered_topics.append(pd.Series([id_nr, topic_first]), ignore_index=True)
    
pf_answered_topics.columns = ["pf_id", "topic_first"]
pf_topics = pf.merge(pf_answered_topics, left_on="pf_id", right_on="pf_id")
topics_top_ind = pd.DataFrame(pf_topics.groupby("topic_first")["pf_ind"].agg(lambda x:x.value_counts().index[0])).reset_index()
topics_top_ind.columns = ["Most answered topic", "Industry of Professionals"]
topics_top_ind
topics_top_hl = pd.DataFrame(pf_topics.groupby("topic_first")["pf_hl"].agg(lambda x:x.value_counts().index[0])).reset_index()
topics_top_hl.columns = ["Most answered topic", "Headline of Professionals"]
topics_top_hl
pf_topics.groupby("pf_ind").size().sort_values(ascending=False).head()
pf_topics.groupby("pf_hl").size().sort_values(ascending=False).head()