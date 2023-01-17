!pip install -U sentence-transformers
import numpy as np 
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
# true = pd.read_csv('../input/stancedata/real_withstance (1).csv')
# true['label'] = 0

# cleansed_data = []
# for data in true.text:
#     if "@realDonaldTrump : - " in data:
#         cleansed_data.append(data.split("@realDonaldTrump : - ")[1])
#     elif "(Reuters) -" in data:
#         cleansed_data.append(data.split("(Reuters) - ")[1])
#     else:
#         cleansed_data.append(data)

# true["text"] = cleansed_data
# true.head(5)
# fake = pd.read_csv('../input/stancedata/fake_withstance (1).csv')
# fake['label'] = 1

# dataset = pd.concat([true, fake])
# dataset = dataset.sample(frac = 1, random_state = 13).reset_index(drop = True)
# dataset['full_text'] = dataset['title'] + ' ' + dataset['text']

# del true, fake
# gc.collect()
# # dataset = dataset[:15000]
# dataset.head()
dataset = pd.read_csv('../input/real-fake/news_dataset.csv')
dataset = dataset.drop(['Unnamed: 0'], 1)
dataset['label'] = dataset['label'].map({'fake':1,'real':0})
dataset = dataset.dropna(subset=['title','content'])
dataset['full_text'] = dataset['title'] + ' ' + dataset['content']
dataset = dataset.sample(frac = 1, random_state = 13).reset_index(drop = True)
dataset.head()
sentences = dataset.full_text
sentence_embeddings = model.encode(sentences)
stance_embeddings = np.hstack([sentence_embeddings,pd.factorize(dataset.stance)[0].reshape((dataset.shape[0],1))])
from sklearn import cluster

# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2, random_state=20, max_iter = 400)

# Fit predict will return labels
clustered = kmeans.fit_predict(sentence_embeddings)
if clustered[0] == 0:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, dataset['label'].values)))
dbscan = cluster.DBSCAN(eps=0.5,min_samples=5, metric='euclidean', leaf_size=30)

# Fit predict will return labels
clustered = dbscan.fit_predict(sentence_embeddings)
if clustered[0] == 0:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
np.unique(clustered, return_counts=True) 
# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2,random_state=20,max_iter=400)

# Fit predict will return labels
clustered = kmeans.fit_predict(stance_embeddings)
if clustered[0] == 1:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, dataset['label'].values)))
def cosine_similarity(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_similarity(sentence_embeddings[0],sentence_embeddings[1])) ### label 0 and 1
print(cosine_similarity(sentence_embeddings[0],sentence_embeddings[2])) ### label 0 and 0
print(cosine_similarity(sentence_embeddings[1],sentence_embeddings[4])) ### label 1 and 1
from sklearn.decomposition import PCA

X = np.array(sentence_embeddings)

pca = PCA(n_components=3)
result = pca.fit_transform(X)
clustered = kmeans.fit_predict(result)

if clustered[0] == 1:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, dataset['label'].values)))
result_stance = np.hstack([result,pd.factorize(dataset.stance)[0].reshape((dataset.shape[0],1))])

clustered = kmeans.fit_predict(result_stance)

if clustered[0] == 1:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(dataset['label'].values):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, dataset['label'].values)))
df = pd.DataFrame({
    'sent': dataset.full_text.values,
    'cluster': clustered.astype(int),
    'x': result[:, 0],
    'y': result[:, 1],
    'z': result[:, 2]
})
df.head()
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 9))
ax = mplot3d.Axes3D(fig)

for grp_name, grp_idx in df.groupby('cluster').groups.items():
    if grp_name == 0:
        name = 'real news'
    else: name = 'fake news'
    y = df.iloc[grp_idx,3]
    x = df.iloc[grp_idx,2]
    z = df.iloc[grp_idx,4]
    ax.scatter(x,y,z, label=str(name))

ax.legend()
# import nltk
# nltk.download('vader_lexicon')
# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# sid = SentimentIntensityAnalyzer()
# dataset['scores'] = dataset['full_text'].apply(lambda full_text: sid.polarity_scores(full_text))
# def keywithmaxval(d): 
#     v=list(d.values())
#     k=list(d.keys())
#     return k[v.index(max(v))]

# dataset['sentiment'] = dataset['scores'].apply(lambda x: keywithmaxval(x))
# dataset['sentiment'].value_counts()
# result_stancesent = np.hstack([result,pd.factorize(dataset.stance)[0].reshape((dataset.shape[0],1)),
#                               pd.factorize(dataset.sentiment)[0].reshape((dataset.shape[0],1))])

# clustered = kmeans.fit_predict(result_stancesent)

# if clustered[0] == 1:
#     clustered = 1 - clustered  ### invert the first to match the group

# correct = 0
# incorrect = 0
# for index, row in enumerate(dataset['label'].values):
#     if row == clustered[index]:
#         correct += 1
#     else:
#         incorrect += 1
        
# print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
# print("AUC: "+str(roc_auc_score(clustered, dataset['label'].values)))
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short # Preprocesssing
from gensim.models import Word2Vec # Word2vec
import re
def remove_URL(s):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'',s)

# Preprocessing functions to remove lowercase, links, whitespace, tags, numbers, punctuation, strip words
CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, remove_URL, strip_punctuation, strip_multiple_whitespaces, 
                  strip_numeric, remove_stopwords, strip_short]

# Here we store the processed sentences and their label
processed_data = []
processed_labels = []
indices = []

for index, row in dataset.iterrows():
    words_broken_up = preprocess_string(row['full_text'], CUSTOM_FILTERS)
    # This eliminates any fields that may be blank after preprocessing
    if len(words_broken_up) > 0:
        processed_data.append(words_broken_up)
        processed_labels.append(row['label'])
        indices.append(index)
model = Word2Vec(processed_data, min_count=1)

def ReturnVector(x):
    try:
        return model[x]
    except:
        return np.zeros(100)
    
def Sentence_Vector(sentence):
    word_vectors = list(map(lambda x: ReturnVector(x), sentence))
    return np.average(word_vectors, axis=0).tolist()

X_np = []
for data_x in processed_data:
    X_np.append(Sentence_Vector(data_x))
X_np = np.array(X_np)
X_np.shape
# Training for 2 clusters (Fake and Real)
kmeans = cluster.KMeans(n_clusters=2)

# Fit predict will return labels
clustered = kmeans.fit_predict(X_np)

if clustered[0] == 1:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(processed_labels):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, processed_labels)))
shortened_embeddings = np.array([sentence_embeddings[i] for i in indices])
shortened_embeddings = shortened_embeddings*2
clustered = kmeans.fit_predict(np.hstack([X_np, shortened_embeddings]))

if clustered[0] == 0:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(processed_labels):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, processed_labels)))
shortened_embeddings = np.array([stance_embeddings[i] for i in indices])
shortened_embeddings = shortened_embeddings*0.5
clustered = kmeans.fit_predict(np.hstack([X_np, shortened_embeddings]))

if clustered[0] == 1:
    clustered = 1 - clustered  ### invert the first to match the group

correct = 0
incorrect = 0
for index, row in enumerate(processed_labels):
    if row == clustered[index]:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")
print("AUC: "+str(roc_auc_score(clustered, processed_labels)))