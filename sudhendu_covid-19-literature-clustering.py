import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import json



import matplotlib.pyplot as plt

plt.style.use('ggplot')
root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'

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

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

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
from ast import literal_eval



dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

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

        authors = literal_eval(meta_data['authors'].values[0])

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

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

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])

df_covid.head()
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
df_covid.info()
df_covid['abstract'].describe(include='all')
df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)

df_covid['abstract'].describe(include='all')
df_covid['body_text'].describe(include='all')
df_covid.head()
df_covid.describe()
df_covid.dropna(inplace=True)

df_covid.info()
import re



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
def lower_case(input_str):

    input_str = input_str.lower()

    return input_str



df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))

df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
df_covid.head(4)
text = df_covid.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count", "authors", "title", "journal", "abstract_summary"], axis=1)
text.head(5)
text_arr = text.stack().tolist()

len(text_arr)
words = []

for ii in range(0,len(text)):

    words.append(str(text.iloc[ii]['body_text']).split(" "))
print(words[0][:10])
n_gram_all = []



for word in words:

    # get n-grams for the instance

    n_gram = []

    for i in range(len(word)-2+1):

        n_gram.append("".join(word[i:i+2]))

    n_gram_all.append(n_gram)
n_gram_all[0][:10]
from sklearn.feature_extraction.text import HashingVectorizer



# hash vectorizer instance

hvec = HashingVectorizer(lowercase=False, analyzer=lambda l:l, n_features=2**12)



# features matrix X

X = hvec.fit_transform(n_gram_all)
X.shape
from sklearn.model_selection import train_test_split



# test set size of 20% of the data and the random seed 42 <3

X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)



print("X_train size:", len(X_train))

print("X_test size:", len(X_test), "\n")
from sklearn.manifold import TSNE



tsne = TSNE(verbose=1, perplexity=5)

X_embedded = tsne.fit_transform(X_train)
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", 1)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette=palette)



plt.title("t-SNE Covid-19 Articles")

# plt.savefig("plots/t-sne_covid19.png")

plt.show()
from sklearn.cluster import KMeans



k = 10

kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)

y_pred = kmeans.fit_predict(X_train)
y_train = y_pred
y_test = kmeans.predict(X_test)
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y_pred)))



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_pred, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered")

# plt.savefig("plots/t-sne_covid19_label.png")

plt.show()
# function to print out classification model report

def classification_report(model_name, test, pred):

    from sklearn.metrics import precision_score, recall_score

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import f1_score

    

    print(model_name, ":\n")

    print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(test, pred)) * 100), "%")

    print("     Precision: ", '{:,.3f}'.format(float(precision_score(test, pred, average='micro')) * 100), "%")

    print("        Recall: ", '{:,.3f}'.format(float(recall_score(test, pred, average='micro')) * 100), "%")

    print("      F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



# random forest classifier instance

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)



# cross validation on the training set 

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=3, n_jobs=4)



# print out the mean of the cross validation scores

print("Accuracy: ", '{:,.3f}'.format(float(forest_scores.mean()) * 100), "%")
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_score, recall_score



# cross validate predict on the training set

forest_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3, n_jobs=4)



# print precision and recall scores

print("Precision: ", '{:,.3f}'.format(float(precision_score(y_train, forest_train_pred, average='macro')) * 100), "%")

print("   Recall: ", '{:,.3f}'.format(float(recall_score(y_train, forest_train_pred, average='macro')) * 100), "%")
# first train the model

forest_clf.fit(X_train, y_train)



# make predictions on the test set

forest_pred = forest_clf.predict(X_test)
# print out the classification report

classification_report("Random Forest Classifier Report (Test Set)", y_test, forest_pred)
df_covid = df_covid.head(10000)
from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=2**12)

X = vectorizer.fit_transform(df_covid['body_text'].values)
X.shape
from sklearn.cluster import MiniBatchKMeans



k = 10

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X)
y = y_pred
from sklearn.manifold import TSNE



tsne = TSNE(verbose=1)

X_embedded = tsne.fit_transform(X.toarray())
from matplotlib import pyplot as plt

import seaborn as sns



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y)))



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")

# plt.savefig("plots/t-sne_covid19_label_TFID.png")

plt.show()
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

pca_result = pca.fit_transform(X.toarray())
# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# colors

palette = sns.color_palette("bright", len(set(y)))



# plot

sns.scatterplot(pca_result[:,0], pca_result[:,1], hue=y, legend='full', palette=palette)

plt.title("PCA Covid-19 Articles - Clustered (K-Means) - Tf-idf with Plain Text")

# plt.savefig("plots/pca_covid19_label_TFID.png")

plt.show()
%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ax = plt.figure(figsize=(16,10)).gca(projection='3d')

ax.scatter(

    xs=pca_result[:,0], 

    ys=pca_result[:,1], 

    zs=pca_result[:,2], 

    c=y, 

    cmap='tab10'

)

ax.set_xlabel('pca-one')

ax.set_ylabel('pca-two')

ax.set_zlabel('pca-three')

plt.title("PCA Covid-19 Articles (3D) - Clustered (K-Means) - Tf-idf with Plain Text")

# plt.savefig("plots/pca_covid19_label_TFID_3d.png")

plt.show()
from sklearn.cluster import MiniBatchKMeans



k = 20

kmeans = MiniBatchKMeans(n_clusters=k)

y_pred = kmeans.fit_predict(X)

y = y_pred
from matplotlib import pyplot as plt

import seaborn as sns

import random 



# sns settings

sns.set(rc={'figure.figsize':(15,15)})



# let's shuffle the list so distinct colors stay next to each other

palette = sns.hls_palette(20, l=.4, s=.9)

random.shuffle(palette)



# plot

sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)

plt.title("t-SNE Covid-19 Articles - Clustered(K-Means) - Tf-idf with Plain Text")

# plt.savefig("plots/t-sne_covid19_20label_TFID.png")

plt.show()
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure



output_notebook()

y_labels = y



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded[:,0], 

    y= X_embedded[:,1], 

    desc= y_labels, 

    titles= df_covid['title'],

    authors = df_covid['authors'],

    journal = df_covid['journal'],

    abstract = df_covid['abstract_summary']

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Author(s)", "@authors"),

    ("Journal", "@journal"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Tf-idf with Plain Text", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black")



# output_file('plots/t-sne_covid-19_interactive.html')

show(p)