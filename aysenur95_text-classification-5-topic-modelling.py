import numpy as np

import pandas as pd

from os import path

from PIL import Image

import pickle

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.feature_extraction.text import CountVectorizer
#load datasets



with open("../input/text-classification-2-feature-engineering/df_train.pkl", 'rb') as data:

    df_train = pickle.load(data)



with open("../input/text-classification-2-feature-engineering/df_test.pkl", 'rb') as data:

    df_test = pickle.load(data)

    

with open("../input/text-classification-2-feature-engineering/le.pkl", 'rb') as data:

    le = pickle.load(data)

    
df_all=pd.concat([df_train, df_test])

df_all
#encoder dict

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

le_name_mapping
text = " ".join(review for review in df_all['review_parsed'])



print ("There are {} words in the combination of all review.".format(len(text)))



# Create and generate a word cloud image:

wordcloud = WordCloud(width=800, height=300,background_color="pink").generate(text)



plt.figure(figsize=(20,5))

# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
def create_word_cloud(df, label, key):

    

    text = " ".join(review for review in df[df['condition']==label]['review_parsed'].values)



    print ("There are {} words in the ".format(len(text)), key, "condition")



    wordcloud = WordCloud(width=800, height=300,background_color="white").generate(text)



    plt.figure(figsize=(20,5))

    plt.title(key, fontdict={'fontsize':20})

    

    # Display the generated image:

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
for key, value in le_name_mapping.items():

    create_word_cloud(df_all, value, key)
# Helper function(her bir küme(topic) içn top-5 sözcük bastırıyor)

def print_topics(model, count_vectorizer, n_top_words):

    words = count_vectorizer.get_feature_names()

    for topic_idx, topic in enumerate(model.components_):

        print("\nTopic #%d:" % topic_idx)

        print(" ".join([words[i]

                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
count_vectorizer = CountVectorizer()

count_data = count_vectorizer.fit_transform(df_all['review_parsed'])



number_topics = 10

number_words = 8



lda = LDA(n_components=number_topics, n_jobs=-1, random_state=8)

lda.fit(count_data)



# Print the topics found by the LDA model

print("Topics found via LDA:")

print_topics(lda, count_vectorizer, number_words)
import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

import gzip
#load datasets



with gzip.open("../input/text-classification-3-2-text-representation/x_train_tfidf.pkl", 'rb') as data:

    x_train_tfidf = pickle.load(data)

    



with gzip.open("../input/text-classification-3-2-text-representation/x_test_tfidf.pkl", 'rb') as data:

    x_test_tfidf = pickle.load(data)

    



with gzip.open("../input/text-classification-3-2-text-representation/y_train.pkl", 'rb') as data:

    y_train = pickle.load(data)

    



with gzip.open("../input/text-classification-3-2-text-representation/y_test.pkl", 'rb') as data:

    y_test = pickle.load(data)

import plotly.express as px



pca = PCA(n_components = 3)

pca_2 = PCA(n_components = 2)

title = "PCA decomposition"  





concat_features_arr = np.concatenate([x_train_tfidf, x_test_tfidf], axis=0)

concat_labels_arr = np.concatenate([y_train, y_test], axis=0)



# Fit and transform the features

pc = pca.fit_transform(concat_features_arr)

pc_2 = pca_2.fit_transform(concat_features_arr)



# Put them into a dataframe

df_features = pd.DataFrame(data=pc,

                 columns=['PC1', 'PC2', 'PC3'])



df_features_2 = pd.DataFrame(data=pc_2,

                 columns=['PC1', 'PC2'])



# Now we have to paste each row's label and its meaning

# Convert labels array to df

df_labels = pd.DataFrame(data=concat_labels_arr,

                         columns=['label'])



df_full = pd.concat([df_features, df_labels], axis=1)

df_full_2 = pd.concat([df_features_2, df_labels], axis=1)



new_dict = {value:key for key, value in le_name_mapping.items()}



# And map labels

df_full['label_name'] = df_full['label']

df_full = df_full.replace({'label_name':new_dict})



df_full_2['label_name'] = df_full_2['label']

df_full_2 = df_full_2.replace({'label_name':new_dict})





arr_full=df_full.to_numpy()

arr_full_2=df_full_2.to_numpy()

fig = px.scatter_3d(x=arr_full [:,0], y=arr_full [:,1], z=arr_full [:,2], color=arr_full [:,4])

fig.show()
fig = px.scatter(x=arr_full_2[:,0], y=arr_full_2[:,1], color=arr_full_2[:,3])

fig.show()