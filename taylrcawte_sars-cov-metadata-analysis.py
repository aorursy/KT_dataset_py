# standard and visualization libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from mpl_toolkits.mplot3d import Axes3D
import warnings
import numpy as np

# nltk libs
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

# sklearn libs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
# ignore the haters (aka warnings)
warnings.filterwarnings('ignore')

# download stopwords corpus
nltk.download('stopwords') 

# set seaborn grid style 
sns.set(style='darkgrid')

# load the data 
papers = pd.read_csv('../input/dataset/CORD-19-research-challenge/metadata.csv', usecols=['title', 'abstract', 'authors', 'doi', 'publish_time'])

papers.head()
papers.shape

papers.drop_duplicates(subset='abstract', keep='first', inplace=True)
papers.dropna(inplace=True)

# clean text data
stop = stopwords.words('english')
stop.append(['use'])
stemmer = SnowballStemmer('english', ignore_stopwords=True)

papers['cleaned_abs'] = papers['abstract'].str.lower()
papers['cleaned_abs'] = papers['cleaned_abs'].str.replace('U\.S\.A|U\.S\.A\.|U\.S\.|U\.S', 'america')
papers['cleaned_abs'] = papers['cleaned_abs'].str.replace(':|;|,|\.', '')
papers['cleaned_abs'] = papers['cleaned_abs'].str.replace('abstract|background|summary', '')
papers['split_abs'] = papers['cleaned_abs'].str.split()
papers['stemmed_abs'] = papers['split_abs'].apply(lambda x: [stemmer.stem(w) for w in x])
papers['stemmed_abs'] = papers['stemmed_abs'].str.join(' ')
papers['stemmed_abs'] = papers['stemmed_abs'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# view cleaned text
papers.stemmed_abs.head()

def find_parenths(cell):
    words = re.findall('\((.*?)\)', cell)
    return words


papers['brackets'] = papers['cleaned_abs'].apply(find_parenths)
brackCount = papers['brackets'].explode().value_counts()
brackCount.head(20)

pos = papers['cleaned_abs'].str.find('(+)')
pos = pos[pos != -1]
one = papers['cleaned_abs'].str.find('(1)')
one = one[one != -1]
two = papers['cleaned_abs'].str.find('(2)')
two = two[two != -1]
three = papers['cleaned_abs'].str.find('(3)')
three = three[three != -1]
four = papers['cleaned_abs'].str.find('(4)')
four = four[four != -1]
ess = papers['cleaned_abs'].str.find('(s)')
ess = ess[ess != -1]
enn = papers['cleaned_abs'].str.find('(n)')
enn = enn[enn != -1]

print('-'*20+'pos'+'-'*20)
print(papers['cleaned_abs'][list(pos.index[0:3])].head())
print('-'*20+'one'+'-'*20)
print(papers['cleaned_abs'][list(one.index[0:3])].head())
print('-'*20+'two'+'-'*20)
print(papers['cleaned_abs'][list(two.index[0:3])].head())
print('-'*20+'three'+'-'*20)
print(papers['cleaned_abs'][list(three.index[0:3])].head())
print('-'*20+'four'+'-'*20)
print(papers['cleaned_abs'][list(four.index[0:3])].head())
print('-'*20+'s'+'-'*20)
print(papers['cleaned_abs'][list(ess.index[0:3])].head())
print('-'*20+'n'+'-'*20)
print(papers['cleaned_abs'][list(enn.index[0:3])].head())
# drop bracketed items that don't reference disease
brackCount.drop(['+', '2', 's', '1', '3', 'pro', 'n', '4', 'i', 'ii', '50'], inplace=True)
brackCount.drop(brackCount.index[8], inplace=True)
brackCount['sars-cov'] += brackCount['sars']
brackCount.drop('sars', inplace=True)

# plot the number of occurences of disease related words in parentheses 
brackCount[:10].plot(kind='barh')
plt.xlabel('Number of Papers')
plt.title('Most commonly referenced diseases')
# parse date published and plot number of publications as a function of year
papers['date'] = papers['publish_time'].str.split('-')
papers['Year'] = papers['date'].str[0]
papers['month'] = papers['date'].str[1]
papers['day'] = papers['date'].str[2]

pubCount = papers.groupby(['Year']).size().to_frame('count').reset_index()
pubCount.set_index('Year', drop=True, inplace=True)
ax = pubCount.drop('2020').plot(kind='line', legend=False)
ax.axvline(32, color='red')
ax.axvline(43, color='green')
ax.axvline(48, color='blue')
plt.text(19, 400, 'SARS outbreak', color='red')
plt.text(34, 2200, 'MERS \noutbreak 1', color='green')
plt.text(33, 2800, 'MERS outbreak 2', color='blue')
plt.ylabel('Number of Publications')
plt.title('Publications within corpus according to year')
plt.show()
# plot number of papers regarding specific coronaviruses
# human coronaviruses 229E, NL63, OC43, and HKU1 are common, MERS and SARS were transmitted in the early 2000s and are
# more severe


def disease_type(cell):
    if ('(sars)' or '(sars-cov)' or '(sarscov)') in cell:
        type = 'sars'
    elif ('(mers)' or '(mers-cov)' or '(merscov)') in cell:
        type = 'mers'
    else:
        type = 'other'
    return type


papers['disease type'] = papers['cleaned_abs'].apply(disease_type)

sarsPapers = papers[papers['disease type'] == 'sars']
sarsCount = sarsPapers.groupby(['Year']).size().to_frame('count').reset_index()
sarsCount.set_index('Year', drop=True, inplace=True)
ax2 = sarsCount.drop('2020').plot(kind='line', legend=False)
ax2.axvline(0, color='red')
plt.text(0.2, 74, 'SARS outbreak', color='red')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xlim(left=-1)
plt.title('Publications regarding SARS')
plt.show()

mersPapers = papers[papers['disease type'] == 'mers']
mersCount = mersPapers.groupby(['Year']).size().to_frame('count').reset_index()
mersCount.set_index('Year', drop=True, inplace=True)
ax1 = mersCount.drop('2020').plot(kind='line', legend=False)
ax1.axvline(0, color='green')
ax1.axvline(5, color='blue')
plt.text(0.18, 40, 'MERS \noutbreak 1', color='green')
plt.text(3.7, 20, 'MERS \noutbreak 2', color='blue')
plt.xlabel('Year')
plt.ylabel('Number of Publications')
plt.xlim(left=-1)
plt.title('Publications regarding MERS')
plt.show()
therapy = papers['stemmed_abs'].str.count(' therapi ').sum()
treatment = papers['stemmed_abs'].str.count(' treatment ').sum()
vaccine = papers['stemmed_abs'].str.count(' vaccin ').sum()
policy = papers['stemmed_abs'].str.count(' polici ').sum()
strategy = papers['stemmed_abs'].str.count(' strategi ').sum()

height = [therapy, treatment, vaccine, policy, strategy]
bars = ('Therapy', 'Treatment', 'Vaccine', 'Policy', 'Strategy')
ypos = np.arange(len(bars))

plt.figure()
plt.bar(ypos, height)
plt.xticks(ypos, bars)
plt.xlabel('Strategy to deal with disease')
plt.ylabel('Occurences of strategy within corpus')
plt.title('Most common methods referenced in corpus regarding control methods of disease')
plt.show()
# features
vectorizer = TfidfVectorizer(max_features=2**12)
x = vectorizer.fit_transform(papers['stemmed_abs'].values)


# kmeans
ss_distance = []
k_range = range(1, 50, 5)

for k in k_range:
    print('k = {}'.format(k))
    mbk = MiniBatchKMeans(n_clusters=k, init='k-means++', random_state=101)
    mbk = mbk.fit(x.toarray())
    ss_distance.append(mbk.inertia_)

plt.figure()
plt.plot(k_range, ss_distance)
plt.axvline(16, color='k', ls='--')
plt.text(17, 32000, 'Elbow = 16 clusters')
plt.ylabel('Inertia')
plt.xlabel('Number of Clusters')
plt.title('Elbow method for KMeans Clustering')
plt.show()
# ----------------------------- compute with ideal number of clusters and plot with PCA/tSNE------------------------------- #
k = 16  # ideal number of clusters was 13
mbk = MiniBatchKMeans(n_clusters=k, init='k-means++', random_state=101)
pred = mbk.fit_predict(x)

papers['cluster'] = mbk.labels_

palette = sns.color_palette('bright', len(set(pred)))

pca = PCA(n_components=3)
pcaRes = pca.fit_transform(x.toarray())

plt.figure() # 2D plot
sns.scatterplot(pcaRes[:, 0], pcaRes[:, 1], hue=pred, legend='full', palette=palette)
plt.title('2D-PCA for visualizing KMeans clusters')
plt.show()

ax = plt.figure(figsize=(15,12)).gca(projection='3d') # 3D plot
ax.scatter(
    xs=pcaRes[:, 0],
    ys=pcaRes[:, 1],
    zs=pcaRes[:, 2],
    c=pred,
    cmap='tab10'
)

ax.set_xlabel('pca one')
ax.set_ylabel('pca two')
ax.set_zlabel('pca three')
plt.title('3D PCA for visualizing KMeans clusters')
plt.show()
clusters = {}

for i in range(k):
    clusters[i] = papers[papers.cluster == i]

vectorizer = TfidfVectorizer(max_features=15) # use top 15 words to identify themes 

for dct in clusters:
    features = pd.DataFrame(vectorizer.fit_transform(clusters[dct]['stemmed_abs']).toarray(),
                            columns=vectorizer.get_feature_names()).T
    dist = 1-cosine_similarity(features)

    linkage = ward(dist)

    fig, ax = plt.subplots()
    ax = dendrogram(linkage, labels=vectorizer.get_feature_names(), orientation='right')
    plt.title('Most associated words in cluster {}'.format(dct))
papers.head()
