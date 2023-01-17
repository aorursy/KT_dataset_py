%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train_variants = pd.read_csv("../input/training_variants")

test_variants = pd.read_csv("../input/test_variants")

train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

print('Unique Classes: ', len(train_variants.Class.unique()))

train_text.head()
train_text.loc[:, 'Text Count'] = train_text['Text'].apply(lambda x: len(x.split()))
train_text = train_text[train_text['Text Count'] != 1]

train_text_sorted = train_text.sort_values('Text Count', ascending=0)
train_text_sorted.tail()
train_text_sorted['Text'][0]
from wordcloud import WordCloud

train_full = train_text.merge(train_variants, how="inner", left_on="ID", right_on="ID")

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_full[train_full.Class == 3]['Text']))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

ax = plt.axes()

ax.set_title('Class 3 Text Word Cloud')
data = train_variants[train_variants['Class'] == 3].groupby('Gene')["ID"].count().reset_index()

sns.barplot(x="Gene", y="ID", data=data.sort_values('ID', ascending=False)[:7])

ax = plt.axes()

ax.set_title('Class 3 Top Gene Variants')
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=9, max_iter=5,

                                learning_method='online',

                                learning_offset=50.,

                                random_state=0)
from sklearn.feature_extraction.text import CountVectorizer

n_features = 50

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,

                                max_features=50,

                                stop_words='english')

tf = tf_vectorizer.fit_transform(train_full['Text'])
tf_feature_names = tf_vectorizer.get_feature_names()

tf_feature_names
lda.fit(tf)
def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        print("Topic #%d:" % topic_idx)

        print(" ".join([feature_names[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()
print_top_words(lda, tf_feature_names, 10)