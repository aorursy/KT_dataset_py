! pip install gensim nltk wordcloud twython
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("vader_lexicon")
nltk.download("subjectivity")

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet, subjectivity

from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import gensim
from gensim import corpora, models, similarities
from gensim.utils import ClippedCorpus
df = pd.read_csv("../input/abcnews-date-text.csv", parse_dates=[0])
df.dtypes
df.shape
df.head(10)
df = df.sample(80000).reset_index(drop=True)
df.head(10)
df["day"] = df["publish_date"].dt.day
df["weekday"] = df["publish_date"].dt.weekday_name
df["week"] = df["publish_date"].dt.week
df["month"] = df["publish_date"].dt.month
df["year"] = df["publish_date"].dt.year
df["headline_lower"] = df["headline_text"].str.lower()
df["headline_tokens"] = df["headline_lower"].str.split()
titles = df["headline_tokens"].values
%%time
stopWords = set(stopwords.words('english'))
titles_token = [[token for token in title if token not in stopWords] for title in titles]
%%time
titles_postag = [nltk.pos_tag(title_token) for title_token in titles_token]
def get_wordnet_pos(treebank_tag):
    """
    Convertir le tag treebank au format wordnet
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
%%time
lmtzr = WordNetLemmatizer()
titles_token_lem = [[lmtzr.lemmatize(token, pos=get_wordnet_pos(titles_postag[i][j][1]))
           for j, token in enumerate(title_token)] for i, title_token in enumerate(titles_token)]
frequency = defaultdict(int)
for title in titles_token_lem:
    for token in title:
        frequency[token] += 1

titles_token_lem_filtered = [[token for token in title
                              if frequency[token] > 1 and len(token) > 1]
                             for title in titles_token_lem]
np.save("titles", titles_token_lem_filtered)
titles_token_lem_filtered = np.load("titles.npy")
dictionary = corpora.Dictionary(titles_token_lem_filtered)
dictionary.save('newsheadlines.dict')
print(dictionary)
titles_bow = [dictionary.doc2bow(title) for title in titles_token_lem_filtered]
corpora.MmCorpus.serialize('news-corpus.mm', titles_bow)
dictionary = corpora.Dictionary.load('newsheadlines.dict')
titles_bow = corpora.MmCorpus('news-corpus.mm')
print(titles_bow)
tfidf = models.TfidfModel(titles_bow)
titles_tfidf = tfidf[titles_bow]
n_train = 50000
titles_bow_train = ClippedCorpus(titles_bow, n_train)
titles_tfidf_train = ClippedCorpus(titles_tfidf, n_train)
n_test = 80000
titles_token_test = ClippedCorpus(titles_token_lem_filtered, n_test)
titles_bow_test = ClippedCorpus(titles_bow, n_test)
titles_tfidf_test = ClippedCorpus(titles_tfidf, n_test)
%%time
text = ""
for title in titles_token_test:
    text = text + " " + " ".join(title)
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.close('all')
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
def print_topics(model, n_topics, n_words):
    """
    Afficher les mots affectés pour chaque thème
    """
    topics = model.show_topics(num_topics=n_topics, num_words=n_words, log=False, formatted=False)
    for topic in topics:
        print("Thème %d: " %(topic[0]), *[word[0] for word in topic[1][:n_words]])
def get_top_topics(topic_probs, prob_limit=0.1):
    """
    Obtenir les thèmes avec une probabilité importante
    """
    return list(filter(lambda item: item[1] > prob_limit, topic_probs))
def get_first_topic(topic_probs):
    """
    Obtenir le premier thème d'un titre
    """
    return max(topic_probs, key=lambda item: item[1])[0]
def get_topic_words(model, title, n_words):
    """
    Obtenir les mots affectés à un thème
    """
    topics = model.show_topics(num_topics=-1, num_words=n_words, log=False, formatted=False)
    if len(model[title]) > 0:
        topic = get_first_topic(model[title])
        return " ".join(["Thème %d:" %(topics[topic][0])] + [word[0] for word in topics[topic][1][:n_words]])
    else:
        return "Aucun"
n_topics = 100
%%time
lsi = models.LsiModel(titles_tfidf_train, id2word=dictionary, num_topics=n_topics)
print_topics(lsi, n_topics=5, n_words=4)
lsi.save("lsi-topics")
lsi = models.LsiModel.load("lsi-topics")
%%time
lda = models.LdaModel(titles_bow_train, id2word=dictionary, num_topics=n_topics)
print_topics(lda, n_topics=5, n_words=4)
lda.save("lda-topics")
lda = models.LdaModel.load("lda-topics")
for idx, title in enumerate(titles_tfidf[:5]):
    
    print("Title:", df.loc[idx, "headline_text"])
    print("Title transformed: ", titles_token_lem_filtered[idx])
    print("LDA topic =>", get_topic_words(lda, titles_bow[idx], n_words=4))
    print("LSI topic =>", get_topic_words(lsi, title, n_words=4))
    print()
%%time
number_lsi_topics = [len(title_lsi) for title_lsi in lsi[titles_tfidf_test]]
number_lda_topics = [len(title_lda) for title_lda in lda[titles_bow_test]]
plt.close('all')
fig, axes = plt.subplots(figsize=(16,8))

plt.hist(number_lsi_topics, bins=50, label="LSI")
plt.hist(number_lda_topics, bins=50, label="LDA")

plt.xlabel("Nombre de thèmes")
plt.ylabel("Fréquence")
plt.legend(loc='best')
plt.title('Distribution du nombre de thèmes par titre')
plt.show()
%%time
number_lsi_top_topics = [len(get_top_topics(title_lsi, prob_limit=0.1)) for title_lsi in lsi[titles_tfidf_test]]
number_lda_top_topics = [len(get_top_topics(title_lda, prob_limit=0.1)) for title_lda in lda[titles_bow_test]]
plt.close('all')
fig, axes = plt.subplots(figsize=(16,8))

plt.hist(number_lsi_top_topics, bins=50, label="LSI")
plt.hist(number_lda_top_topics, bins=50, label="LDA")

plt.xlabel("Nombre de thèmes les plus probables")
plt.ylabel("Fréquence")
plt.legend(loc='best')
plt.title('Distribution du nombre de thèmes les plus probables par titre')
plt.show()
%%time 
lsi_first_topic = [get_first_topic(title_lsi) for title_lsi in lsi[titles_tfidf_test] if len(title_lsi) > 0]
%%time 
lda_first_topic = [get_first_topic(title_lda) for title_lda in lda[titles_bow_test] if len(title_lda) > 0]
plt.close('all')
fig, axes = plt.subplots(figsize=(16,8))

plt.hist(lsi_first_topic, bins=200, label="LSI")
plt.hist(lda_first_topic, bins=200, label="LDA")

plt.xlabel("Identifiant du thème")
plt.ylabel("Fréquence")
plt.legend(loc='best')
plt.title('Distribution du premier thème')
plt.show()
%%time
sid = SentimentIntensityAnalyzer()
df["polarity_score"] = [sid.polarity_scores(" ".join(title_token))["compound"] 
                                          for title_token in titles_token_test]

df.loc[df["polarity_score"] > 0, "polarity"] = "pos"
df.loc[df["polarity_score"] == 0, "polarity"] = "neu"
df.loc[df["polarity_score"] < 0, "polarity"] = "neg"
plt.close('all')
fig, axes = plt.subplots(figsize=(8,5))

df["polarity_score"].hist(bins=50)

plt.xlabel("Polarité")
plt.ylabel("Fréquence")
plt.title('Distribution de la polarité des titres')
plt.show()
polarity_score_by_week_and_year = df.groupby(["year", "week"])["polarity_score"].mean()
polarity_score_by_month_and_year = df.groupby(["year", "month"])["polarity_score"].mean()
polarity_score_by_year = df.groupby(["year"])["polarity_score"].mean()
std_polarity_score_by_week_and_year = df.groupby(["year", "week"])["polarity_score"].std()
std_polarity_score_by_month_and_year = df.groupby(["year", "month"])["polarity_score"].std()
std_polarity_score_by_year = df.groupby(["year"])["polarity_score"].std()
polarity_by_year = df.groupby(["year", "polarity"])["polarity"].count().unstack()
plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))

polarity_score_by_year.plot()
plt.fill_between(std_polarity_score_by_year.index, polarity_score_by_year-2*std_polarity_score_by_year,
                 polarity_score_by_year+2*std_polarity_score_by_year, color='b', alpha=0.2)

plt.xlabel("Année")
plt.ylabel("Score de polarité")
plt.title('Score de polarité des titres moyen par année')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))

polarity_by_year.plot.bar(stacked=True, ax=ax)

plt.xlabel("Année")
plt.ylabel("Fréquence")
plt.title('Distribution de la polarité des titres par année')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(16, 5))

df.groupby(['year', 'polarity'])["polarity"].count().unstack().div(
    df.groupby(['year', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0).plot.bar(ax=ax, stacked=True)

plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title('Part des polarités des titres par année')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))

polarity_score_by_month_and_year.plot()

plt.xlabel("(Année, mois)")
plt.ylabel("Score de polarité")
plt.title('Score de polarité des titres médian par mois et année')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(16,5))

polarity_score_by_week_and_year.plot()


plt.xlabel("(Année, semaine)")
plt.ylabel("Score de polarité")
plt.title('Score de polarité des titres moyen par semaine et année')
plt.show()
months = ['January', 'Februrary', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
           'October', 'November', 'December']
polarity_score_by_month = [df[df["month"] == month_number + 1]["polarity_score"].values
                       for month_number, month in enumerate(months)]
plt.close('all')
fig, ax = plt.subplots(figsize=(16, 5))

plt.boxplot(polarity_score_by_month, 0, '')

plt.xticks(range(1, len(months) + 1), months)
plt.xlabel("Mois de l'année")
plt.ylabel("Score de polarité")
plt.title("Box plot du score de polarité des titres moyen par mois de l'année")
plt.show()
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
polarity_score_by_weekday = [df[df["weekday"] == day]["polarity_score"].values
                       for day in days]
plt.close('all')
fig, ax = plt.subplots(figsize=(16, 5))

plt.boxplot(polarity_score_by_weekday, 0, '')

plt.xticks(range(1, len(days) + 1), days)
plt.xlabel("Jour de la semaine")
plt.ylabel("Score de polarité")
plt.title('Box plot du score de polarité des titres moyen par jour de la semaine')
plt.show()
n_instances = 5000 # Nombre de documents labellisés subjectif/objectif dans le jeu de données complet
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
train_subj_docs = subj_docs[:int(n_instances*0.8)]
test_subj_docs = subj_docs[int(n_instances*0.8):n_instances]
train_obj_docs = obj_docs[:int(n_instances*0.8)]
test_obj_docs = obj_docs[int(n_instances*0.8):n_instances]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
# We use simple unigram word features, handling negation:

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
# We apply features to obtain a feature-value representation of our datasets:

training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)
%%time

# We can now train our classifier on the training set, and subsequently output the evaluation results:

trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
%%time
df["subjectivity"] = [sentim_analyzer.classify(title) for title in titles_token_test]
plt.close('all')
fig, axes = plt.subplots(figsize=(10, 5))

df['subjectivity'].value_counts().plot(kind='bar')

plt.xlabel("Niveau de subjectivité")
plt.ylabel("Fréquence")
plt.title('Distribution de la subjectivité des titres')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(15, 5))

df.groupby(["year", 'subjectivity'])["subjectivity"].count().unstack().plot(kind='bar', ax=ax)

plt.xlabel("Année")
plt.ylabel("Fréquence")
plt.title('Distribution de la subjectivité des titres par année')
plt.show()
plt.close('all')
fig, ax = plt.subplots(figsize=(15, 5))

df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().div(
    df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1),
    axis=0).plot.bar(ax=ax, stacked=True)

plt.xlabel("Année")
plt.ylabel("Pourcentage")
plt.title('Part de subjectivité des titres par année')
plt.show()
subjectivity_proportion_by_year = df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().div(
                                    df.groupby(['year', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1),
                                    axis=0)
polarity_proportion_by_year = df.groupby(['year', 'polarity'])["polarity"].count().unstack().div(
    df.groupby(['year', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0)
print("Corrélations (périodicité ANNUELLE de 2003-2017) entre la proportion des titres subjectifs et:")

print("- négatifs: %.2f" %(
    subjectivity_proportion_by_year["subj"].corr(polarity_proportion_by_year["neg"], method='pearson')))
print("- neutre: %.2f" %(
    subjectivity_proportion_by_year["subj"].corr(polarity_proportion_by_year["neu"], method='pearson')))
print("- positifs: %.2f" %(
    subjectivity_proportion_by_year["subj"].corr(polarity_proportion_by_year["pos"], method='pearson')))
subjectivity_proportion_by_month_year = df.groupby(['year', 'month', 'subjectivity'])["subjectivity"].count().unstack().div(
                                    df.groupby(['year', 'month', 'subjectivity'])["subjectivity"].count().unstack().sum(axis=1),
                                    axis=0)
polarity_proportion_by_month_year = df.groupby(['year', 'month', 'polarity'])["polarity"].count().unstack().div(
    df.groupby(['year', 'month', 'polarity'])["polarity"].count().unstack().sum(axis=1),
    axis=0)
print("Corrélations (périodicité MENSUELLE de 2003-2017) entre la proportion des titres subjectifs et:")

print("- négatifs: %.2f" %(
    subjectivity_proportion_by_month_year["subj"].corr(polarity_proportion_by_month_year["neg"], method='pearson')))
print("- neutre: %.2f" %(
    subjectivity_proportion_by_month_year["subj"].corr(polarity_proportion_by_month_year["neu"], method='pearson')))
print("- positifs: %.2f" %(
    subjectivity_proportion_by_month_year["subj"].corr(polarity_proportion_by_month_year["pos"], method='pearson')))