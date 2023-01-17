import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
bio = pd.read_csv('../input/biorxiv_clean.csv')
"""
comm = pd.read_csv('../input/clean_comm_use.csv')
noncomm = pd.read_csv('../input/clean_noncomm_use.csv')
pmc = pd.read_csv('../input/clean_pmc.csv')
"""
bio.info()
"""
Add other data frames and concat here
"""
frames = [bio]
df = pd.concat(frames)
df.info()
df['length'] = df['text'].apply(len)
df.head()
plt.figure(figsize=(10,10))
df.length.plot(bins=30, kind='hist') 
df.length.describe()
plt.figure(figsize=(10,10))
df[df['length'] < 100000]['length'].plot(bins=50, kind='hist') 
#Looking at text under 100 characters
"""
df[df['length'] < 100]
"""
#Looking at distribution of character length under 10000 words
plt.figure(figsize=(8,8))
df[df['length'] < 10000]['length'].plot(bins=50, kind='hist') 
#Looking at distribution of character length under 5000 words
plt.figure(figsize=(8,8))
df[df['length'] < 5000]['length'].plot(bins=50, kind='hist') 
#Looking at distribution of character length under 1000 words
"""
plt.figure(figsize=(8,8))
df[df['length'] < 1000]['length'].plot(bins=50, kind='hist')
"""
df = df[df['length'] > 500]
import string
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
#download stopwords and wordnet with 'nltk.download_shell()'

def tokenize(text):
    words = ""
    #remove punctuation
    words = [char for char in text if char not in string.punctuation]
    words = ''.join(words)
    #remove /n
    words = words.replace('\n', ' ')
    #remove stopwords
    words = [word for word in words.split() if word.lower() not in stopwords.words('english')]
    #lemmatize
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word, pos = "v") for word in words]
    words = [lemma.lemmatize(word, pos = "n") for word in words]
    return words
    
   

df.info()
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=tokenize, max_features = 2**12)
text_bow = bow_transformer.fit_transform(df['text'])

from sklearn.feature_extraction.text import TfidfTransformer
t_transformer = TfidfTransformer()
tfidf = t_transformer.fit_transform(text_bow)
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
tfidf_pca = pca.fit_transform(tfidf.toarray())
explained_variance = pca.explained_variance_ratio_

#determine the ammount of components to account for 95% of variance
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components = 0

    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components
comp = select_n_components(explained_variance, .95)
pca = PCA(n_components = comp) 
tfidf_pca = pca.fit_transform(tfidf.toarray()) 
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,50))
visualizer.fit(tfidf_pca)      
visualizer.show() 

clusters = 15
kmeans = KMeans(n_clusters = clusters, init = 'k-means++', random_state=42)
kmeans.fit(tfidf_pca)

labels = pd.DataFrame(kmeans.labels_, columns=['cluster'])
tokens = pd.DataFrame(tokens)
abstract = pd.DataFrame(df.abstract)
df_cluster = pd.concat([labels, df.text, abstract], axis=1)
df_cluster

d = {}
for i in range(0,clusters):
    d[i] = df_cluster[df_cluster['cluster'] == i]
df_test = d[6]
df_test
df_test.dropna(inplace=True)
df_test
import re
df_test['paper_text_processed'] = df_test['abstract'].map(lambda x: re.sub('[,\.!?]', '', x))
df_test['paper_text_processed'] = df_test['paper_text_processed'].map(lambda x: x.lower())
df_test['paper_text_processed'] = df_test['paper_text_processed'].map(lambda x: x.replace('\n', ' '))
df_test['paper_text_processed'] = df_test['paper_text_processed'].map(lambda x: x.replace('abstract', ' '))
df_test['paper_text_processed'].head()

from wordcloud import WordCloud
long_string = ','.join(list(df_test['paper_text_processed'].values))
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
    
count_vectorizer = TfidfVectorizer(stop_words='english')
count_data = count_vectorizer.fit_transform(df_test['paper_text_processed'])
plot_10_most_common_words(count_data, count_vectorizer)
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.decomposition import LatentDirichletAllocation as LDA

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        

number_topics = 5
number_words = 10

lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)

print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)
