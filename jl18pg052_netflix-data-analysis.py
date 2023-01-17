import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud

import re

import nltk

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

import gensim

from gensim import corpora

import pyLDAvis

import pyLDAvis.gensim

from gensim.models.coherencemodel import CoherenceModel
df=pd.read_csv("../input/netflix-shows/netflix_titles.csv")

df.head()
df.shape
df.isnull().sum()
plt.figure(figsize=(14,7))

df["type"].value_counts().plot(kind="pie",shadow=True,autopct = '%1.1f%%')
count=list(df['country'].dropna().unique())

cloud=WordCloud(colormap="cool",width=800,height=400).generate(" ".join(count))

fig=plt.figure(figsize=(14,10))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["country"].value_counts()[:20].plot(kind="bar",color="lightcoral")

plt.title("Top 20 countries in terms of maximum number of movies on netflix",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["country"].value_counts()[:20].plot(kind="bar",color="mediumslateblue")

plt.title("Top 20 countries in terms of maximum number of TV shows on netflix",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df["release_year"].value_counts()[:20].plot(kind="bar",color="green")

plt.title("Frequency of both TV Shows and movies which are released in different years",size=16)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["release_year"].value_counts()[:20].plot(kind="bar",color="darkorange")

plt.title("Frequency of Movies which are released in different years and are there in netflix",size=16)
df[(df["type"]=="Movie") & (df["release_year"]==2017)]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["release_year"].value_counts()[:20].plot(kind="bar",color="mediumblue")

plt.title("Frequency of TV shows which are released in different years and are there in netflix",size=16)
df[(df["type"]=="TV Show") & (df["release_year"]==2019)]["title"].sample(10)
listed=list(df['listed_in'].unique())

cloud=WordCloud(colormap="Wistia",width=600,height=400).generate(" ".join(listed))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for all category",size=18)
listed2=list(df[df["type"]=="Movie"]['listed_in'].unique())

cloud=WordCloud(colormap="summer",width=600,height=400).generate(" ".join(listed2))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for movie category",size=18)
listed3=list(df[df["type"]=="TV Show"]['listed_in'].unique())

cloud=WordCloud(colormap="YlOrRd",width=600,height=400).generate(" ".join(listed3))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for TV show category",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["listed_in"].value_counts()[:20].plot(kind="barh",color="red")

plt.title("20 most frequent genre for movie type for all the years",size=18)
df[(df["listed_in"]=="Documentaries") & (df["type"]=="Movie")]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(11,6))

df[df["type"]=="TV Show"]["listed_in"].value_counts()[:20].plot(kind="barh",color="darkviolet")

plt.title("20 most frequent genre for TV show type for all the years",size=18)
df[(df["listed_in"]=="Kids' TV") & (df["type"]=="TV Show")]["title"].sample(10)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["type"]=="Movie") & (df["release_year"]==2019)]["listed_in"].value_counts()[:20].plot(kind="barh",color="lime")

plt.title("20 most frequent Genre for movie type for the year 2019",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(11,6))

df[(df["type"]=="TV Show") & (df["release_year"]==2019)]["listed_in"].value_counts()[:20].plot(kind="barh",color="teal")

plt.title("20 most frequent genre for TV show type for the year 2019",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df["rating"].value_counts().plot(kind="bar",color="orange")

plt.title("Frequency of ratings for both TV shows & movies for all the years",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="Movie"]["rating"].value_counts().plot(kind="bar",color="royalblue")

plt.title("Frequency of ratings for movie category for all the years",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[df["type"]=="TV Show"]["rating"].value_counts().plot(kind="bar",color="orangered")

plt.title("Frequency of ratings for TV show category for all the years",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="United States") & (df["type"]=="Movie")]["rating"].value_counts().plot(kind="bar",color="slateblue")

plt.title("Rating for Movies that are released in USA",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="India") & (df["type"]=="Movie")]["rating"].value_counts().plot(kind="bar",color="deeppink")

plt.title("Rating for Movies that are released in India",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="United States") & (df["type"]=="TV Show")]["rating"].value_counts().plot(kind="bar",color="fuchsia")

plt.title("Rating for TV Shows that are released in USA",size=18)
plt.style.use("ggplot")

plt.figure(figsize=(12,6))

df[(df["country"]=="India") & (df["type"]=="TV Show")]["rating"].value_counts().plot(kind="bar",color="gold")

plt.title("Rating for TV Shows that are released in India",size=18)
listed4=list(df[(df["release_year"]==2019) & (df["type"]=="Movie")]['title'])

cloud=WordCloud(colormap="YlOrRd",width=600,height=400).generate(" ".join(listed4))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for movie names which are released in the year 2019",size=18)
listed4=list(df[(df["release_year"]==2019) & (df["type"]=="TV Show")]['title'])

cloud=WordCloud(colormap="winter",width=600,height=400).generate(" ".join(listed4))

fig=plt.figure(figsize=(12,18))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')

plt.title("WordCloud for genre for TV Show category released in 2019",size=18)
data=pd.DataFrame(df["description"])

pd.set_option("display.max_colwidth", 200)

data.head()
data['Number of words'] = data['description'].apply(lambda x:len(str(x).split()))

data.head()
plt.figure(figsize=(12,6))

sns.distplot(data["Number of words"], kde=False, color="red",bins=8)

plt.xlabel("Number of words",size=15)

plt.ylabel("count",size=15)

plt.title("Distribution of number of words in the documents",size=15)
data["Number of words"].describe()
def clean_text(text):

    clean=text.lower()

    clean=re.sub("[^a-z]"," ",clean)

    clean=re.sub(r"\s+[a-z]\s+"," ",clean)

    clean=clean.lstrip()

    clean=re.sub("\s{2,}"," ",clean)

    return clean

data["cleaned_text"]=data["description"].apply(clean_text)
cloud=WordCloud(colormap="PuBuGn",width=800,height=400).generate(str(data["cleaned_text"]))

fig=plt.figure(figsize=(14,10))

plt.axis("off")

plt.imshow(cloud,interpolation='bilinear')
nltk.download("stopwords")

from nltk.corpus import stopwords

stop=stopwords.words('english')

data["stops_removed"]=data["cleaned_text"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data["tokenized"]=data["stops_removed"].apply(lambda x: nltk.word_tokenize(x))
def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i,pos='v') for i in text]

    return lem_text

data["lemmatized"]=data["tokenized"].apply(lambda x: word_lemmatizer(x))

data["joined"]=data["lemmatized"].apply(lambda x: ' '.join(x))
data["stops_removed_2"]=data["joined"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
dictionary = corpora.Dictionary(data["lemmatized"])

doc_term_matrix = [dictionary.doc2bow(rev) for rev in data["lemmatized"]]
LDA = gensim.models.ldamodel.LdaModel



# Build LDA model

lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=15, random_state=100,

                chunksize=1000, passes=50)
lda_model.print_topics()
coherence_model_lda = CoherenceModel(model=lda_model,

texts=data["lemmatized"], dictionary=dictionary, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
pyLDAvis.enable_notebook(sort=True)

vis_2 = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary, sort_topics=False)

pyLDAvis.display(vis_2)
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    """

    Compute c_v coherence for various number of topics



    Parameters:

    ----------

    dictionary : Gensim dictionary

    corpus : Gensim corpus

    texts : List of input texts

    limit : Max num of topics



    Returns:

    -------

    model_list : List of LDA topic models

    coherence_values : Coherence values corresponding to the LDA model with respective number of topics

    """

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=data["lemmatized"], start=2, limit=40, step=6)

# Show graph

import matplotlib.pyplot as plt

limit=40; start=2; step=6;

x = range(start, limit, step)

plt.figure(figsize=(12,6))

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()
lda_model_2 = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=37, random_state=120,

                chunksize=1000, passes=50)
coherence_model_lda_2 = CoherenceModel(model=lda_model_2,

texts=data["lemmatized"], dictionary=dictionary, coherence='c_v')

coherence_lda_2 = coherence_model_lda_2.get_coherence()

print('\nCoherence Score: ', coherence_lda_2)
pyLDAvis.enable_notebook(sort=True)

vis = pyLDAvis.gensim.prepare(lda_model_2, doc_term_matrix, dictionary, sort_topics=False)

pyLDAvis.display(vis)