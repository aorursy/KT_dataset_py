import time
import datetime # 時間の変換
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json

import matplotlib.pyplot as plt
plt.style.use('ggplot')
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
meta_df.info()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            try:
                # Abstract
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
                # Body text
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            except Exception as e:
                pass
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)
def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data
# This action takes time
start = time.time() # Start time of loading

dict_ = {'paper_id': [], 'doi':[], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json):
    if idx % (len(all_json) // 10) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    
    try:
        content = FileReader(entry)
    except Exception as e:
        continue  # invalid paper format, skip
        
    
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['abstract'].append(content.abstract)
    dict_['paper_id'].append(content.paper_id)
    dict_['body_text'].append(content.body_text)

    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 100 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # if more than 2 authors, take them all with html tag breaks in between
            dict_['authors'].append(get_breaks('. '.join(authors), 40))
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
    # add doi
    dict_['doi'].append(meta_data['doi'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()
calc_time = time.time() - start # Calculating how long it takes to perform loading above
print("computing time : ", datetime.timedelta(seconds=calc_time))
df_covid.info()
# Partial match
df_keyword_abstract = df_covid[df_covid["abstract"].str.contains("Keyword")]
df_keyword_abstract
df_keyword_body_text = df_covid[df_covid["body_text"].str.contains("Keyword")]
df_keyword_body_text
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))  # word count in abstract
df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))  # word count in body
df_covid['body_unique_words']=df_covid['body_text'].apply(lambda x:len(set(str(x).split())))  # number of unique words in body
df_covid.head()
df_covid.info()
df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
df_covid['body_text'].describe(include='all')
df_covid['abstract'].describe(include='all')
df_covid.head()
df_covid.describe()
df = df_covid.sample(10000, random_state=42)
del df_covid
df.dropna(inplace=True)
df.info()
from tqdm import tqdm
from langdetect import detect
from langdetect import DetectorFactory

# set seed
DetectorFactory.seed = 0

# hold label - language
languages = []

# go through each text
for ii in tqdm(range(0,len(df))):
    # split by space into list, take the first x intex, join with space
    text = df.iloc[ii]['body_text'].split(" ")
    
    lang = "en"
    try:
        if len(text) > 50:
            lang = detect(" ".join(text[:50]))
        elif len(text) > 0:
            lang = detect(" ".join(text[:len(text)]))
    # ught... beginning of the document was not in a good format
    except Exception as e:
        all_words = set(text)
        try:
            lang = detect(" ".join(all_words))
        # what!! :( let's see if we can find any text in abstract...
        except Exception as e:
            
            try:
                # let's try to label it through the abstract then
                lang = detect(df.iloc[ii]['abstract_summary'])
            except Exception as e:
                lang = "unknown"
                pass
    
    # get the language    
    languages.append(lang)
from pprint import pprint

languages_dict = {}
for lang in set(languages):
    languages_dict[lang] = languages.count(lang)
    
print("Total: {}\n".format(len(languages)))
pprint(languages_dict)
df['language'] = languages
plt.bar(range(len(languages_dict)), list(languages_dict.values()), align='center')
plt.xticks(range(len(languages_dict)), list(languages_dict.keys()))
plt.title("Distribution of Languages in Dataset")
plt.show()
df = df[df['language'] == 'en'] 
df.info()
#Importing Spacy STOP_WORDS and English
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
stopwords[:10]
custom_stop_words = [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 'table',
    'rights', 'reserved', 'permission', 'use', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.', 
    'al.', 'Elsevier', 'PMC', 'CZI', 'www','-PRON-', 'usually',
    r'\usepackage{amsbsy', r'\usepackage{amsfonts', r'\usepackage{mathrsfs', r'\usepackage{amssymb', r'\usepackage{wasysym',
    r'\setlength{\oddsidemargin}{-69pt',  r'\usepackage{upgreek', r'\documentclass[12pt]{minimal'
]

for w in custom_stop_words:
    if w not in stopwords:
        stopwords.append(w)
# Download the spacy bio parser

# from IPython.utils import io
# with io.capture_output() as captured:
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_lg-0.2.3.tar.gz
# Parser
import en_core_sci_lg
parser = en_core_sci_lg.load(disable=["tagger", "ner"])
parser.max_length = 7000000

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
tqdm.pandas()
df["processed_text"] = df["body_text"].progress_apply(spacy_tokenizer)
import seaborn as sns
sns.distplot(df['body_word_count'])
df['body_word_count'].describe()
sns.distplot(df['body_unique_words'])
df['body_unique_words'].describe()
from sklearn.feature_extraction.text import TfidfVectorizer
@stopwatch(callback=print_fn)
def vectorize(text, maxx_features):
    
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X
import numpy as np
import re
!pip install lauda
from lauda import stopwatch
def print_fn(watch, function):
    print ('実行時間{0}秒'.format(watch.elapsed_time))
class TF_IDF():

    def __init__(self, corpus):
        self.corpus = corpus

    def tf(self):
        l = []
        c = []

    #corpusの各テキストを単語毎に分割。
    #sklearの結果と合わせるため、分割方法は正規表現でのマッチで行う

        for text in self.corpus:
            c +=  re.findall(r'\b\w+\b', text)

    #抽出した単語の重複を削除
        c = list(set(c))
    #各テキスト毎の単語出現回数をカウントし、該当テキストの総単語数で割る

        for text in self.corpus:
            xxx = re.findall(r'\b\w+\b', text)
            l.append([xxx.count(i)/len(xxx) for i in c])

        return np.array(l)

    def idf(self):

        terms = []

    #corpusの各テキストを単語毎に分割。
    #sklearの結果と合わせるため、分割方法は正規表現でのマッチで行う

        for text in self.corpus:
            terms +=  re.findall(r'\b\w+\b', text)

        terms = list(set(terms))

        l = []

        for term in terms:
            #各単語がそのテキストに含まれているかどうかをカウント
            c = 0

            for text in self.corpus:
                #各テキストを単語単位に分割
                word_list = re.findall(r'\b\w+\b', text)
                #該当テキスト内に含まれている単語であれば、１カウントする
                #重複カウントを防ぐ為に論理演算子は「in」を用いる
                #文章の繋がりで、単語ではないものを単語としてカウントしないように上記でリスト化している
                if term in word_list:
                    c += 1

                #各単語IDFを計算。sklearnの計算と合わせるため、分母分子に１を足し、更にその計算結果にも1を足す。
            l.append(np.log((1 + len(self.corpus))/(c+1)) + 1) 

        return np.array(l)    

    def l2(self, x):
        #l2ノルムで正規化する（単位ベクトル化する）
        l2 = x / np.sqrt(np.sum(x**2))

        return l2

    @stopwatch(callback=print_fn)
    def tf_idf(self):

        xxxx = self.tf()*self.idf()
        #各行にl2ノルムの正規化を適用
        return np.array([self.l2(a) for a in xxxx])
class Okapi_BM25():

    def __init__(self, corpus):
        self.corpus = corpus
        self.k1 = 2
        self.b = 0.75
        # 　k1は主に単語の出現頻度から計算した重要度(TF-IDF値を指す)の影響の大きさを調整するパラメータである．k1=1.2もしくは2.0とし，k1=2.0が一番効果的であることが確認されている．
        # 　bは主に文書の単語数による影響の大きさを調整するパラメータである．0.0から1.0の間で設定し，b=0.75が一番効果的であることが確認されている．なお，b=0.0とした場合，文書の単語数による影響をなくした結果を得ることができる．
   
    def l2(self, x):
        #l2ノルムで正規化する（単位ベクトル化する）
        l2 = x / np.sqrt(np.sum(x**2))

        return l2
    
    
    @stopwatch(callback=print_fn)
    def okapi_bn25(self, max_features):
        terms = []
        dl = np.zeros(len(self.corpus))
        idf_val = 1e-3 # idf値の最小値を定数ϵとし、一般的な用語を完全に無視することを避けつつ、影響を減らす

        #corpusの各テキストを単語毎に分割。
        #sklearの結果と合わせるため、分割方法は正規表現でのマッチで行う
        for text in self.corpus:
            terms +=  re.findall(r'\b\w+\b', text)
        #抽出した単語の重複を削除
        terms = list(set(terms))
        # Only the top n features will be used
        terms = terms[:max_features]
        
        tf = np.zeros((len(self.corpus), len(terms)))
        idf = np.zeros(len(terms))
        
        for i, term in enumerate(terms):
            #各単語がそのテキストに含まれているかどうかをカウント
            c = 0
            for j, text in enumerate(self.corpus):                   
                #各テキストを単語単位に分割
                word_list = re.findall(r'\b\w+\b', text)  
                
                if i==0:
                    # tf値 
                    dl[j] = len(word_list)
                    
                #該当テキストの総単語数で割る
                tf[j, i] = word_list.count(term) / len(word_list)
                
                # idf値
                #該当テキスト内に含まれている単語であれば、１カウントする
                #重複カウントを防ぐ為に論理演算子は「in」を用いる
                #文章の繋がりで、単語ではないものを単語としてカウントしないように上記でリスト化している
                if term in word_list:
                    c += 1
            #各単語IDFを計算。
            if idf_val < np.log((len(self.corpus) - c + 0.5) / (c + 0.5)): 
                idf_val = np.log((len(self.corpus) - c + 0.5) / (c + 0.5)) 
 
            idf[i] = idf_val
              
        avg_dl = dl.sum() / len(self.corpus) # すべての文書の平均単語数: (文書の総数) / (総単語数)
        dl = np.expand_dims(dl, axis=0).T    
             
        xxxx = np.array(tf * idf*(self.k1 + 1.) / (tf + self.k1 * (1. - self.b + self.b * dl / avg_dl)))
        #各行にl2ノルムの正規化を適用
        return np.array([self.l2(a) for a in xxxx])
text = df['processed_text'].values
print(len(text))
x = TF_IDF(text[:20])
tf_idf_a =x.tf_idf()

print(tf_idf_a)
print(tf_idf_a.shape)
print("score: ", np.sum(tf_idf_a[0]))
x = Okapi_BM25(text[:20])
okapi_bn25_a =x.okapi_bn25(max_features=2 ** 12)

print(okapi_bn25_a)
print(okapi_bn25_a.shape)
print("score: ", np.sum(okapi_bn25_a[0]))
text = df['processed_text'].values
X = vectorize(text, 2 ** 12)
np.sum(X.toarray()[0])
print(X.toarray())
print(X.toarray().shape)
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, random_state=42)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape
from sklearn.cluster import KMeans
from PIL import Image
# Image(filename='/kaggle/input/kaggle-resources/kmeans.PNG', width=800, height=800)
from sklearn import metrics
from scipy.spatial.distance import cdist

# run kmeans with many different k
distortions = []
K = range(2, 50)
for k in K:
    k_means = KMeans(n_clusters=k, random_state=42).fit(X_reduced)
    k_means.fit(X_reduced)
    distortions.append(sum(np.min(cdist(X_reduced, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    #print('Found distortion for {} clusters'.format(k))
X_line = [K[0], K[-1]]
Y_line = [distortions[0], distortions[-1]]

# Plot the elbow
plt.plot(K, distortions, 'b-')
plt.plot(X_line, Y_line, 'r')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_reduced)
df['y'] = y_pred
from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=100, random_state=42)
X_embedded = tsne.fit_transform(X.toarray())
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)
plt.title('t-SNE with no Labels')
plt.savefig("t-sne_covid19.png")
plt.show()
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.hls_palette(20, l=.4, s=.9)

# plot
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('t-SNE with Kmeans Labels')
plt.savefig("improved_cluster_tsne.png")
plt.show()
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
Image(filename='/kaggle/input/kaggle-resources/lda.jpg', width=600, height=600)
vectorizers = []
    
for ii in range(0, 20):
    # Creating a vectorizer
    vectorizers.append(CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))
vectorizers[0]
vectorized_data = []

for current_cluster, cvec in enumerate(vectorizers):
    try:
        vectorized_data.append(cvec.fit_transform(df.loc[df['y'] == current_cluster, 'processed_text']))
    except Exception as e:
        print("Not enough instances in cluster: " + str(current_cluster))
        vectorized_data.append(None)
len(vectorized_data)
# number of topics per cluster
NUM_TOPICS_PER_CLUSTER = 20

lda_models = []
for ii in range(0, 20):
    # Latent Dirichlet Allocation Model
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=42)
    lda_models.append(lda)
    
lda_models[0]
clusters_lda_data = []

for current_cluster, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_cluster))
    
    if vectorized_data[current_cluster] != None:
        clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values
all_keywords = []
for current_vectorizer, lda in enumerate(lda_models):
    # print("Current Cluster: " + str(current_vectorizer))

    if vectorized_data[current_vectorizer] != None:
        all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
all_keywords[0][:10]
len(all_keywords)
f=open('topics.txt','w')

count = 0

for ii in all_keywords:

    if vectorized_data[count] != None:
        f.write(', '.join(ii) + "\n")
    else:
        f.write("Not enough instances to be determined. \n")
        f.write(', '.join(ii) + "\n")
    count += 1

f.close()
import pickle

# save the COVID-19 DataFrame, too large for github
pickle.dump(df, open("df_covid.p", "wb" ))

# save the final t-SNE
pickle.dump(X_embedded, open("X_embedded.p", "wb" ))

# save the labels generate with k-means(20)
pickle.dump(y_pred, open("y_pred.p", "wb" ))
# function to print out classification model report
def classification_report(model_name, test, pred):
    from sklearn.metrics import precision_score, recall_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    print(model_name, ":\n")
    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")
    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='macro')) * 100), "%")
    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='macro')) * 100), "%")
    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='macro')) * 100), "%")
from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test, y_train, y_test = train_test_split(X.toarray(),y_pred, test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier

# SGD instance
sgd_clf = SGDClassifier(max_iter=10000, tol=1e-3, random_state=42, n_jobs=4)
# train SGD
sgd_clf.fit(X_train, y_train)

# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3, n_jobs=4)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Training Set)", y_train, sgd_pred)
# cross validation predictions
sgd_pred = cross_val_predict(sgd_clf, X_test, y_test, cv=3, n_jobs=4)

# print out the classification report
classification_report("Stochastic Gradient Descent Report (Training Set)", y_test, sgd_pred)
sgd_cv_score = cross_val_score(sgd_clf, X.toarray(), y_pred, cv=10)
print("Mean cv Score - SGD: {:,.3f}".format(float(sgd_cv_score.mean()) * 100), "%")
import os

# change into lib directory to load plot python scripts
main_path = os.getcwd()
lib_path = '/kaggle/input/kaggle-resources'
os.chdir(lib_path)
# required libraries for plot
from call_backs import input_callback, selected_code  # file with customJS callbacks for bokeh
                                                      # github.com/MaksimEkin/COVID19-Literature-Clustering/blob/master/lib/call_backs.py
import bokeh
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column
# go back
os.chdir(main_path)
import os

topic_path = 'topics.txt'
with open(topic_path) as f:
    topics = f.readlines()
# show on notebook
output_notebook()
# target labels
y_labels = y_pred

# data sources
source = ColumnDataSource(data=dict(
    x= X_embedded[:,0], 
    y= X_embedded[:,1],
    x_backup = X_embedded[:,0],
    y_backup = X_embedded[:,1],
    desc= y_labels, 
    titles= df['title'],
    authors = df['authors'],
    journal = df['journal'],
    abstract = df['abstract_summary'],
    labels = ["C-" + str(x) for x in y_labels],
    links = df['doi']
    ))

# hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles{safe}"),
    ("Author(s)", "@authors{safe}"),
    ("Journal", "@journal"),
    ("Abstract", "@abstract{safe}"),
    ("Link", "@links")
],
point_policy="follow_mouse")

# map colors
mapper = linear_cmap(field_name='desc', 
                     palette=Category20[20],
                     low=min(y_labels) ,high=max(y_labels))

# prepare the figure
plot = figure(plot_width=1200, plot_height=850, 
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'], 
           title="Clustering of the COVID-19 Literature with t-SNE and K-Means", 
           toolbar_location="above")

# plot settings
plot.scatter('x', 'y', size=5, 
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black",
          legend = 'labels')
plot.legend.background_fill_alpha = 0.6
# Keywords
text_banner = Paragraph(text= 'Keywords: Slide to specific cluster to see the keywords.', height=45)
input_callback_1 = input_callback(plot, source, text_banner, topics)

# currently selected article
div_curr = Div(text="""Click on a plot to see the link to the article.""",height=150)
callback_selected = CustomJS(args=dict(source=source, current_selection=div_curr), code=selected_code())
taptool = plot.select(type=TapTool)
taptool.callback = callback_selected

# WIDGETS
slider = Slider(start=0, end=20, value=20, step=1, title="Cluster #", callback=input_callback_1)
keyword = TextInput(title="Search:", callback=input_callback_1)

# pass call back arguments
input_callback_1.args["text"] = keyword
input_callback_1.args["slider"] = slider
# STYLE
slider.sizing_mode = "stretch_width"
slider.margin=15

keyword.sizing_mode = "scale_both"
keyword.margin=15

div_curr.style={'color': '#BF0A30', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
div_curr.sizing_mode = "scale_both"
div_curr.margin = 20

text_banner.style={'color': '#0269A4', 'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 'font-size': '1.1em'}
text_banner.sizing_mode = "scale_both"
text_banner.margin = 20

plot.sizing_mode = "scale_both"
plot.margin = 5

r = row(div_curr,text_banner)
r.sizing_mode = "stretch_width"
# LAYOUT OF THE PAGE
l = layout([
    [slider, keyword],
    [text_banner],
    [div_curr],
    [plot],
])
l.sizing_mode = "scale_both"

# show
output_file('t-sne_covid-19_interactive.html')
show(l)
Image(filename="/kaggle/input/kaggle-resources/cluster_9_keywords.PNG", width=1170, height=60)
Image(filename='/kaggle/input/kaggle-resources/cluster_9.PNG', width=420, height=375)
Image(filename='/kaggle/input/kaggle-resources/cluster_9_cattle.PNG', width=420, height=375)
Image(filename='/kaggle/input/kaggle-resources/selected_paper.PNG', width=600, height=100)