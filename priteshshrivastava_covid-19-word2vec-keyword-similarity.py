from sklearn.decomposition import PCA

import pandas as pd

import numpy as np

from gensim.models import Word2Vec

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import warnings

warnings.filterwarnings("ignore")
model = Word2Vec.load('/kaggle/input/covid19-challenge-trained-w2v-model/covid.w2v')
model.wv.most_similar('coronavirus', topn=10)
keywords = ["infection", "cell", "protein", "virus",\

            "disease", "respiratory", "influenza", "viral",\

            "rna", "patient", "pathogen", "human", "medicine",\

            "cov", "antiviral"]



print("Frequency of keyword & Most similar words")

print("")



#top_words_list = []

for word in keywords:

    top_words = model.wv.most_similar(word, topn=5)

    print(word + " - " + "frequency: ", model.wv.vocab[word].count)

    for idx, top_word in enumerate(top_words):

        print(str(idx+1) + ". " + top_word[0])

        #top_words_list.append(top_word[0])

    print("")
model['coronavirus'].shape
words = [word for word in keywords]

X = model[words]

X.shape
pca = PCA(n_components=2)

result = pca.fit_transform(X)

df = pd.DataFrame(result, columns=["Component 1", "Component 2"])

df
df["Word"] = words

df["Distance"] = np.sqrt(df["Component 1"]**2 + df["Component 2"]**2)

df
freq_list = []

for word in words:

    freq_list.append(model.wv.vocab[word].count)

df['frequency'] = freq_list

df
sns.scatterplot(data=df, x="Component 1", y="Component 2", 

                hue="Distance",size="frequency")
fig = px.scatter(df, x="Component 1", y="Component 2", text="Word", 

                 color="Distance", size="frequency", color_continuous_scale="agsunset")

fig.update_traces(textposition='top center')

fig.layout.xaxis.autorange = True

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.update_layout(height=800, title_text="2D PCA of Word2Vec embeddings", 

                  template="plotly_white", paper_bgcolor="#f0f0f0")

fig.show()
def pca_2d_similar(keyword):

    similar_words = model.wv.most_similar(keyword, topn=20)

    df_similar_words = pd.DataFrame(similar_words, columns = ['word', 'dist'])

    words = [word for word in df_similar_words['word'].tolist()]

    X = model[words]

    result = pca.fit_transform(X)

    df = pd.DataFrame(result, columns=["Component 1", "Component 2"])

    df["Word"] = df_similar_words['word']

    word_emb = df[["Component 1", "Component 2"]].loc[0]

    df["Distance"] = np.sqrt((df["Component 1"] - word_emb[0])**2 + (df["Component 2"] - word_emb[1])**2)

    fig = px.scatter(df[2:], x="Component 1", y="Component 2", text="Word", color="Distance", color_continuous_scale="viridis",size="Distance")

    fig.update_traces(textposition='top center')

    fig.layout.xaxis.autorange = True

    fig.data[0].marker.line.width = 1

    fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

    fig.update_layout(height=800, title_text="2D PCA of words related to {}".format(keyword), template="plotly_white", paper_bgcolor="#f0f0f0")

    fig.show()

    

pca_2d_similar('antiviral')
pca_2d_similar('antiretroviral')
pca_2d_similar('daas')
pca_2d_similar('cyclosporine')
pca_2d_similar('lamivudine')
pca_2d_similar('favipiravir')
pca_2d_similar('rna')
pca_2d_similar('dna')