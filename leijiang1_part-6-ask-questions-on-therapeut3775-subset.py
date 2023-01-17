from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import matplotlib
from nltk.tokenize import word_tokenize
from nltk import edit_distance
import nltk
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pandas as pd 
df = pd.read_csv('/kaggle/input/therapeut3775/tibbleTherapeut3775.csv') #therapeut3775 subset generated on april 1 in part 3 notebook
df.head()
df.rename(columns = {'abstract':'ABS'}, inplace = True) 
#df.dtypes
df.shape
df.isnull().sum()
outString = ' '.join(df["ABS"])
print (outString)
def stems(words, method) :
    prtr = nltk.stem.PorterStemmer()
    snob = nltk.stem.SnowballStemmer('english')
    lema = nltk.wordnet.WordNetLemmatizer()
    
    word_to_stem = stopwords_removal(words)

    stem = [w for w in word_to_stem]
    stem = []
    
    if method == 'porter' :
        for w in word_to_stem:
            stem.append(prtr.stem(w))
 
    elif method == 'snowball': 
        for w in word_to_stem:
            stem.append(snob.stem(w))

    return (stem)
def stopwords_removal(words) :
    stop_word = set(stopwords.words('english'))
    word_token = word_tokenize(words)
    output_sentence = [words for word in word_token if not word in stop_word]
    output_sentence = []
    for w in word_token:
        if w not in stop_word:
            output_sentence.append(w)
    return(output_sentence)

stopwords_output = stopwords_removal(outString)
for w in stopwords_output:
    print(w+"|",end=' ')
snowball_stems = stems(outString, "snowball")
print("After stemming, there are",len(snowball_stems),"words. And they are as following:")
print()
for s in snowball_stems:
    print(s+"|",end=' ')
    
 #After stemming, there are 290097 words. And they are as following:
import string
x=snowball_stems
x = [''.join(c for c in s if c not in string.punctuation) for s in x]
x
x = [s for s in x if s]
x
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

import matplotlib

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 300
barWidth = 0.25
plt.figure(figsize=(20,15))

counts = Counter(x)
common = counts.most_common(50)
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)

plt.bar(np.arange(nbars), number,width=barWidth, tick_label=labels)
plt.xticks(rotation = 90, fontweight='bold',fontsize=12,)
plt.show()
barWidth = 0.25
plt.figure(figsize=(20,40))

counts = Counter(x)
common = counts.most_common(200)
labels = [item[0] for item in common]
number = [item[1] for item in common]
nbars = len(common)

plt.barh(np.arange(nbars), number,tick_label=labels) #width=barWidth, 
plt.xticks( fontweight='bold',fontsize=12,) #rotation = 90,
plt.title('Top 200 words in titles of ' +str(df.ABS.shape[0])+  ' research papers mentioned Therapeut',fontsize=15)#,fontweight='bold'   #df2.title.shape
plt.show()
customize_stop_words2 = [
    'used', 'using', 'SARS CoV','MERS CoV','Abstract','found','result','method','conclusion','results','case','cases',
    'compared','many','well','including','identified','Although','present','Middle East','infection','patient'
    'infectious','treatment','China','East','Role','COVID','human','model','Chapter','viruses','methods','disease'
]
#capital letter must match

STOPWORDS2 = list(STOPWORDS)  + customize_stop_words2
text = outString

# Create and generate a word cloud image:
wordcloud = WordCloud(stopwords = STOPWORDS2,  background_color="white").generate(text)

#matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['figure.dpi'] = 300

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Top words cloud in abstracts of ' +str(df.ABS.shape[0])+  ' research papers mentioned Therapeut',fontsize=10)#,fontweight='bold'
plt.show()
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',max_features = 2**12)
tf_idf = tf_idf_vectorizor.fit_transform(df.ABS)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()
pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)
kmeans = KMeans(n_clusters=5, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(X)
prediction = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, s=50, cmap='viridis')
centers = fitted.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)

cls = KMeans(n_clusters=6, init='k-means++',random_state=1) # 
cls.fit(X)
newfeature = cls.labels_ # the labels from kmeans clustering

X2 = np.column_stack((X,pd.get_dummies(newfeature)))

plt.figure()
#plt.subplot(1,2,1)
X2=X2
plt.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow, s=20, linewidths=0,alpha=0.5)
plt.xlabel(''), plt.ylabel('')
plt.grid()
#3D plot

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


fig = pyplot.figure()
ax = Axes3D(fig)


ax.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2,X2[:, 2]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow,alpha=0.25)
pyplot.show()
sse = []
list_k = list(range(1, 20))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

data=X

# Silhouette vs Cluster Size
# do it for the k-means
from sklearn import metrics
from sklearn.cluster import KMeans

seuclid = []
scosine = []
k = range(2,11)
for i in k:
    kmeans_model = KMeans(n_clusters=i, init="k-means++").fit(X)
    labels = kmeans_model.labels_
    seuclid.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    scosine.append(metrics.silhouette_score(data, labels, metric='cosine'))
    
plt.figure(figsize=(10,5))
plt.plot(k,seuclid,label='euclidean')
plt.plot(k,scosine,label='cosine')
plt.ylabel("Silhouette")
plt.xlabel("Cluster")
plt.title("Silhouette vs Cluster Size")
plt.legend()
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

print(__doc__)

# y_lower = 10?
X=X2
y=newfeature
range_n_clusters = [ 3, 4, 5, 6,7,8,9,10,11] # [3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 


for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

data1 = np.array(df.ABS.drop_duplicates(keep='last'))
data1
data1.shape

data1list=data1.tolist()
%%capture
# Install the latest Tensorflow version.
#!pip3 install --upgrade tensorflow-gpu
# Install TF-Hub.
!pip3 install tensorflow-hub
#!pip3 install seaborn
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
embeddings = embed(data1list)["outputs"]

#print(embeddings)
embeddings.shape
NParray1941papers512vector=np.array(embeddings)
NParray1941papers512vector
testAbstract1=["background enterovirus 71 ev71 is one of the major causative agents of hand foot and mouth disease hfmd which is sometimes associated with severe central nervous system disease in children there is currently no specific medication for ev71 infection quercetin one of the most widely distributed flavonoids in plants has been demonstrated to inhibit various viral infections however investigation of the antiev71 mechanism has not been reported to date methods the antiev71 activity of quercetin was evaluated by phenotype screening determining the cytopathic effect cpe and ev71induced cells apoptosis the effects on ev71 replication were evaluated further by determining virus yield viral rna synthesis and protein expression respectively the mechanism of action against ev71 was determined from the effective stage and timeofaddition assays the possible inhibitory functions of quercetin via viral 2apro 3cpro or 3dpol were tested the interaction between ev71 3cpro and quercetin was predicted and calculated by molecular docking results quercetin inhibited ev71mediated cytopathogenic effects reduced ev71 progeny yields and prevented ev71induced apoptosis with low cytotoxicity investigation of the underlying mechanism of action revealed that quercetin exhibited a preventive effect against ev71 infection and inhibited viral adsorption moreover quercetin mediated its powerful therapeutic effects primarily by blocking the early postattachment stage of viral infection further experiments demonstrated that quercetin potently inhibited the activity of the ev71 protease 3cpro blocking viral replication but not the activity of the protease 2apro or the rna polymerase 3dpol modeling of the molecular binding of the 3cproquercetin complex revealed that quercetin was predicted to insert into the substratebinding pocket of ev71 3cpro blocking substrate recognition and thereby inhibiting ev71 3cpro activity conclusions quercetin can effectively prevent ev71induced cell injury with low toxicity to host cells quercetin may act in more than one way to deter viral infection exhibiting some preventive and a powerful therapeutic effect against ev71 further quercetin potently inhibits ev71 3cpro activity thereby blocking ev71 replication"]
testAbstract2=["background to investigate the effects and immunological mechanisms of the traditional chinese medicine xinjiaxiangruyin on controlling influenza virus fm1 strain infection in mice housed in a hygrothermal environment methods mice were housed in normal and hygrothermal environments and intranasally infected with influenza virus fm1 a highperformance liquid chromatography fingerprint of xinjiaxiangruyin was used to provide an analytical method for quality control realtime quantitative polymerase chain reaction rtqpcr was used to measure messenger rna expression of tolllike receptor 7 tlr7 myeloid differentiation primary response 88 myd88 and nuclear factorkappa b nfb p65 in the tlr7 signaling pathway and virus replication in the lungs western blotting was used to measure the expression levels of tlr7 myd88 and nfb p65 proteins flow cytometry was used to detect the proportion of th17tregulatory cells results xinjiaxiangruyin effectively alleviated lung inflammation in c57bl6 mice in hot and humid environments guizhimahuanggebantang significantly reduced lung inflammation in c57bl6 mice the expression of tlr7 myd88 and nfb p65 mrna in lung tissue of wt mice in the normal environment gzmhgbt group was significantly lower than that in the model group p  005 in wt mice exposed to the hot and humid environment the expression levels of tlr7 myd88 and nfb p65 mrna in the xjxry group were significantly different from those in the virus group the expression levels of tlr7 myd88 and nfb p65 protein in lung tissue of wt mice exposed to the normal environment gzmhgbt group was significantly lower than those in the model group in wt mice exposed to hot and humid environments the expression levels of tlr7 myd88 and nfb p65 protein in xjxry group were significantly different from those in the virus group conclusion guizhimahuanggebantang demonstrated a satisfactory therapeutic effect on mice infected with the influenza a virus fm1 strain in a normal environment and xinjiaxiangruyin demonstrated a clear therapeutic effect in damp and hot environments and may play a protective role against influenza through downregulation of the tlr7 signal pathway"]
Question1=['What are clinical effective therapeutics or drugs for COVID-19?']
embeddingsT1 = embed(testAbstract1)["outputs"]
embeddingsT2 = embed(testAbstract2)["outputs"]

embeddingsQ1 = embed(Question1)["outputs"]

test1=np.array(embeddingsT1)
test2=np.array(embeddingsT2)

question1=np.array(embeddingsQ1)

import textwrap
result1 = np.sum(NParray1941papers512vector*test1,axis=1)/(np.sqrt(np.sum(NParray1941papers512vector*NParray1941papers512vector,axis=1))*np.sqrt(np.sum(test1*test1)))
maxRows1=result1.argsort()[-10:][::-1]  #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
print("The indexes for most similar papers are:") 
print(maxRows1)
print("\n")
print("The cosine similarity for top 10 papers are:") 
print(result1[result1.argsort()[-10:][::-1]])
print("\n")
print("For Paper Abstract:\n")
print(textwrap.fill(testAbstract1[0],100))
print("\nWe found the top 10 most similar papers as listed below:\n")
print(df.ABS.iloc[maxRows1])
result2 = np.sum(NParray1941papers512vector*test2,axis=1)/(np.sqrt(np.sum(NParray1941papers512vector*NParray1941papers512vector,axis=1))*np.sqrt(np.sum(test2*test2)))
maxRows2=result2.argsort()[-10:][::-1]  #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
print("The indexes for most similar papers are:") 
print(maxRows2)
print("\n")
print("The cosine similarity for top 10 papers are:") 
print(result2[result2.argsort()[-10:][::-1]])
print("\n")
print("For Paper Abstract:\n")
print(textwrap.fill(testAbstract2[0],100))
print("\nWe found the top 10 most similar papers as listed below:\n")
print(df.ABS.iloc[maxRows2])
resultq1 = np.sum(NParray1941papers512vector*question1,axis=1)/(np.sqrt(np.sum(NParray1941papers512vector*NParray1941papers512vector,axis=1))*np.sqrt(np.sum(question1*question1)))
maxRowsq1=resultq1.argsort()[-20:][::-1]  #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
print("The indexes for most related papers are:") 
print(maxRowsq1)
print("\n")
print("The cosine similarity for top 20 papers are:") 
print(resultq1[resultq1.argsort()[-20:][::-1]])
print("\n")
print("For Question:\n")
print(textwrap.fill(Question1[0],100))
print("\nWe found the top 20 most related papers as listed below:\n")
print(df.ABS.iloc[maxRowsq1])
