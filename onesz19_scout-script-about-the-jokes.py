import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

from textblob import TextBlob

from wordcloud import WordCloud

import sklearn

# assert sklearn.__version__ == '0.18' # Make sure we are in the modern age

from sklearn.feature_extraction.text import CountVectorizer

from gensim.models import Word2Vec



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/jokes.csv')

df.info()
df.head()
text = ' '.join(df.Question)

cloud = WordCloud(background_color='white', width=1920, height=1080).generate(text)

plt.figure(figsize=(32, 18))

plt.axis('off')

plt.imshow(cloud)

plt.savefig('questions_wordcloud.png')
text = ' '.join(df.Answer)

cloud = WordCloud(background_color='white', width=1920, height=1080).generate(text)

plt.figure(figsize=(32, 18))

plt.axis('off')

plt.imshow(cloud)

plt.savefig('answer_wordcloud.png')
# Some defaults

max_features=1000

max_df=0.95,  

min_df=2,

max_features=1000,

stop_words='english'



from nltk.corpus import stopwords

stop = stopwords.words('english')



# document-term matrix A

vectorized = CountVectorizer(max_features=1000, max_df=0.95, min_df=2, stop_words='english')



a = vectorized.fit_transform(df.Question)

a.shape
from sklearn.decomposition import NMF

model = NMF(init="nndsvd",

            n_components=10,

            max_iter=200)



# Get W and H, the factors

W = model.fit_transform(a)

H = model.components_



print("W:", W.shape)

print("H:", H.shape)
vectorizer = vectorized



terms = [""] * len(vectorizer.vocabulary_)

for term in vectorizer.vocabulary_.keys():

    terms[vectorizer.vocabulary_[term]] = term

    

# Have a look that some of the terms

terms[-5:]
for topic_index in range(H.shape[0]):  # H.shape[0] is k

    top_indices = np.argsort(H[topic_index,:])[::-1][0:10]

    term_ranking = [terms[i] for i in top_indices]

    print("Topic {}: {}".format(topic_index, ", ".join(term_ranking)))
get_polarity = lambda x: TextBlob(x).sentiment.polarity

get_subjectivity = lambda x: TextBlob(x).sentiment.subjectivity



df['q_polarity'] = df.Question.apply(get_polarity)

df['a_polarity'] = df.Answer.apply(get_polarity)

df['q_subjectivity'] = df.Question.apply(get_subjectivity)

df['a_subjectivity'] = df.Answer.apply(get_subjectivity)
plt.figure(figsize=(7, 4))

sns.distplot(df.q_polarity , label='Question Polarity')

sns.distplot(df.q_subjectivity , label='Question Subjectivity')

sns.distplot(df.a_polarity , label='Answer Polarity')

sns.distplot(df.a_subjectivity , label='Answer Subjectivity')

sns.plt.legend()
daf = df.loc[df.Answer.str.len() < 150]  # There appear to be some outliers in the dataset

sns.distplot(daf.Question.str.len(), label='Question Length')

sns.distplot(daf.Answer.str.len(), label='Answer Length')

sns.plt.legend()
# What are the outliers though?

# The threshold has been chosen to keep in the spirit of the dataset

df.loc[df.Answer.str.len() > 400].shape[0]
ql, al = 'Question Length', 'Answer Length'

df[ql] = df.Question.str.len()

df[al] = df.Answer.str.len()

daf = df.loc[df[al] < 250]

sns.jointplot(x=ql, y=al, data=daf, kind='kde', space=0, color='g')