#imported packages
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd #to read Excel files
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string #to remove punctuation and digits
from glob import glob
from sklearn.cluster import KMeans
#variables
stopwords = list(ENGLISH_STOP_WORDS)
#more stopwords can be added by using .append 
p = string.punctuation
d = string.digits
combined = p + d
#variables
docs1 = []
company_names = [] 
#function
def func(txt):
    txt = txt.lower() #normalizing
    table = str.maketrans(combined, len(combined) * " ") #removing punc
    txt = txt.translate(table)
    words = txt.split()
    cleaned_words = [w for w in words if w not in stopwords] #removing stopwords
    cleaned_text = " ".join(cleaned_words) #cleaned text
    return cleaned_text
corpus = glob("Patents_xls/*xlsx") #reading file into corpus
#combined two columns AB and TI 
for c in corpus:
    df = pd.read_excel(c)
    df['AB'].dropna(inplace = True)
    df['TI'].dropna(inplace = True)
    df['joined_col'] = df.AB.str.cat(df.TI,sep=" ")
    combo = list (df['joined_col'])
    #title=list(df['TI'])
    combo_2 = ','.join([str(i) for i in combo])  
    #" ".join(abstracts)
    combo_f = func(combo_2)
    docs1.append(combo_f)
    name = c.split("/")[-1][:-5]
    company_names.append(name)
#lemmatization
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
lem_doc = []
for sentence in docs1:
    lem_doc.append(" ".join([wnl.lemmatize(i) for i in sentence.split()]))
#To get token's
import nltk
full_str = ' '.join([str(elem) for elem in lem_doc])
words = nltk.word_tokenize(full_str)
words[:10]
#type(words)
#frequent_words plot
word_frequencies = nltk.FreqDist(words)
word_frequencies.most_common(10)
word_frequencies.plot(25)
#word_count
from collections import Counter
c = Counter(words)
c.most_common(10)
#to get POS_Tags
from textblob import TextBlob
txt = TextBlob(full_str)
x=txt.words
y=txt.sentences
type(x)
#x[:10]
#y[:1]
#Run to get pos_tags, takes more time to run so commented
#import nltk
#nltk.download('averaged_perceptron_tagger')
#z=txt.tags
#z[:10]
#after lemmatization
vectorizer = TfidfVectorizer()
sparse_matrix = vectorizer.fit_transform(lem_doc)
print(sparse_matrix.shape)
#converting to dense format
dense_matrix = sparse_matrix.todense()
tdm = dense_matrix.transpose()
print(tdm)
#imported packages and initiated
from sklearn.metrics.pairwise import cosine_similarity
vectorizer_MSD = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm1 = vectorizer.fit_transform(lem_doc)
similarity1 = cosine_similarity(dtm1)
cos_distance1 = 1 - similarity1
#print(cos_distance1)
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
mds = MDS(n_components = 2, dissimilarity='precomputed', random_state=1)
pos1 = mds.fit_transform(cos_distance1)
#print(pos1)
#plot MDS
plt.title("Based on AB & TI", size=15)
xs, ys = pos1[:,0], pos1[:,1]
for x, y, name in zip(xs, ys, company_names):
    plt.scatter(x, y)
    plt.text(x, y, name)
plt.show()
from scipy.cluster.hierarchy import ward, dendrogram
linkage_matrix= ward(cos_distance1)
plt.title("Based on AB & TI", size=15)
dendrogram(linkage_matrix, orientation='right', labels=company_names)
plt.tight_layout()
plt.show()
#results
km = KMeans(n_clusters=4, random_state=999)
km.fit(sparse_matrix) #computes k-means clustering
cluster_membership = km.predict(sparse_matrix) #predicts closest cluster
company_distance_to_center = km.transform(sparse_matrix) #cluster distance
cluster_membership
print(company_distance_to_center)
#cluster for each company
clusters = zip(cluster_membership, company_names)
print("{0:<15s}{1:<9s}".format("Company_Names","Cluster"))
for cluster_number, company_name in clusters:
    print("{0:<15s}{1:2d}".format(company_name,cluster_number))
#fitting in dataframe
companies = {'Company': company_names, 'Cluster_Final': cluster_membership,\
            'Centroid_Dist0':company_distance_to_center[0:,0],\
            'Centroid_Dist1':company_distance_to_center[0:,1],\
            'Centroid_Dist2':company_distance_to_center[0:,2],\
            'Centroid_Dist3':company_distance_to_center[0:,3]
            }
#let us put this in a dataframe
import pandas as pd
df_km = pd.DataFrame(companies)
df_km
#plot K-means cluster
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
#plotted by taking best measures (distances out of 4)
plt.scatter(company_distance_to_center[0:,0], company_distance_to_center[0:,3], c=cluster_membership, s=50, cmap='viridis')
plt.title("Based on AB & TI", size=15)
#imported packages
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import glob
import os
from sklearn import decomposition
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
dtm = vectorizer.fit_transform(lem_doc)
vocab = vectorizer.get_feature_names() 
names = [fn[:-4] for fn in corpus]
num_topics = 5
num_top_words = 20
clf = decomposition.NMF(n_components = num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)
topic_words = []
for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
print("Getting results: 5 topics")
for t in range(len(topic_words)):
    print("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))
#imported packages
from sklearn.decomposition import LatentDirichletAllocation
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')
num_topics = 5
dtm = vectorizer.fit_transform(lem_doc)
lda = LatentDirichletAllocation(n_components=num_topics, learning_method="batch",  max_iter=2000, random_state=0)
#takes time than usual to run
document_topics = lda.fit_transform(dtm)
feature_names = vectorizer.get_feature_names()
def display_topics(model, feature_names, n_top_words):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
print("Getting results: 5 topics")
display_topics(lda, feature_names, 20)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=5, max_df=0.9,stop_words='english', lowercase=True)
data_vectorized = vectorizer.fit_transform(lem_doc)
def get_topics(model, vectorizer, model_name, dff, top_n = 20):
        result = []
        for idx, topic in enumerate(model.components_): 
            print("Topic %d:" % (idx))
            topic_label = model_name + "_topic_" + str(idx)            
            score = "SCORE_" + str(idx)
            aList = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
            l1, l2 = zip(*aList)
            dff[topic_label] = l1
            dff[score] = l2
            print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-top_n - 1:-1]]))
        return dff
print("Getting results: 5 topics and scores") 
lsi_model = TruncatedSVD(n_components=5, n_iter = 5000)
lsi_Z = lsi_model.fit_transform(data_vectorized)
get_topics(lsi_model, vectorizer, "LSI", pd.DataFrame())
