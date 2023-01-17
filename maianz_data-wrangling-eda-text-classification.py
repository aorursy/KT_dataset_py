# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
# This diction is used to store date of data source as string

Tweet_Date = []

# this dictionary is used to store all data frames of tweet infomation

Tweet_Dict = {}



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

print("--------------------Start loading data--------------------")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        # extract file path

        filepath = os.path.join(dirname, filename)

        print(filepath, end="")

        # append file name head as tweet date

        head = filename.split()[0]

        head = head.split(".")[0]

        # filter out useless files

        if head not in ["2020-03-00", "Hashtags", "Countries"]:

            print(" -----> loading")

            Tweet_Date.append(head)

            # read csv file and store in dict

            Tweet_Dict[head] = pd.read_csv(filepath)

        else:

            print()

print("--------------------Finish loading data--------------------")

# sort the list

Tweet_Date = sorted(Tweet_Date)
# calculate null value percentage

statistics = pd.concat([Tweet_Dict[date].isnull().mean().to_frame(name=date) for date in Tweet_Date], axis=1)

statistics
# drop columns that have too many missing values and useless columns

for date in Tweet_Date:

    Tweet_Dict[date].drop(labels=["country_code", "place_full_name", "place_type", "account_lang", # too many missing values

                                 "status_id", "user_id", "screen_name",  # useless

                                 "reply_to_status_id", "reply_to_user_id", "reply_to_screen_name", # useless

                                 "account_created_at"], axis=1, inplace=True)
# concatenate all the data to one dataframe

df = pd.concat([Tweet_Dict[date] for date in Tweet_Date], ignore_index=True, sort=False)

print("Dataframe shape: ", df.shape)
# drop duplicate columns

df.drop_duplicates(inplace=True)

print("Dataframe shape: ", df.shape)
# impute null values

df["source"].fillna(df["source"].mode()[0], inplace=True)
# present statistics

df.info()
# present description

df.describe()
# list to store number of Tweets

num_tweet = []



# calculate number of tweets on each day

for date in Tweet_Date:

    num_tweet.append(Tweet_Dict[date].shape[0])



# plot

plt.figure(figsize=(12, 5))

plt.bar(Tweet_Date, num_tweet, color="lightcoral")

plt.xticks(rotation=90)

plt.xlabel('Date')

plt.ylabel("Count")

plt.title('Number of Tweets Trendency')

plt.show()
# configure plot size

plt.figure(figsize=(14, 5))

# subplot for favourites_count

plt.subplot(1,4,1)

df.boxplot(column="favourites_count", rot=0, showfliers=False, figsize=(8,6))

# subplot for retweet_count

plt.subplot(1,4,2)

df.boxplot(column="retweet_count", rot=0, showfliers=False, figsize=(8,6))

# subplot for followers_count

plt.subplot(1,4,3)

df.boxplot(column="followers_count", rot=0, showfliers=False, figsize=(8,6))

# subplot for friends_count

plt.subplot(1,4,4)

df.boxplot(column="friends_count", rot=0, showfliers=False, figsize=(8,6))

plt.show()
# configure plot size

plt.figure(figsize=(10, 6))

# extract numeric columns

df_corr = df[["favourites_count", "retweet_count", "followers_count", "friends_count"]]

# generate correlation matrix

corrMatrix = df_corr.corr()

# plot heatmap

sns.heatmap(corrMatrix, annot=True)

plt.show()
# transform text contents to string variable

text = ""

for date in Tweet_Date:

    text += str(Tweet_Dict[date][Tweet_Dict[date]["lang"]=="en"]["text"].values)

# generate word cloud    

wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="black", collocations=False).generate(str(text))

# plot wordcloud

plt.figure(figsize=(18, 8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
# configure plot size

plt.figure(figsize=(14, 5))

# plot for is_quote

plt.subplot(1,2,1)

plt.pie([df["is_quote"].mean(), 1-df["is_quote"].mean()], labels=['True', 'False'], shadow=False, startangle=140)

plt.title("is_quote")

# plot for is_retweet

plt.subplot(1,2,2)

plt.pie([df["is_retweet"].mean(), 1-df["is_retweet"].mean()], labels=['True', 'False'], shadow=False, startangle=140)

plt.title("is_retweet")

plt.show()
# count top 10 sources

df["source"].value_counts().head(10)
# count top 10 languages

df["lang"].value_counts().head(10)
# filter out all tweets that are not using english

df_en = df[(df["lang"]=="en") & (df["favourites_count"]>=df["favourites_count"].mean()*10)]

print("Number of English Tweets that are above average favourites: ", df_en.shape[0])
# vectorize the text content

vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')

X = vectorizer.fit_transform(df_en["text"])
print("shape of tf-idf weight matrix: ", X.shape)
# range of number of clusters

num_clusters = range(2, 22, 2)



# list to record sum of squared distances

sum_square_error = []



# iterate through different number of clusters and append sse

for k in num_clusters:

        sum_square_error.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=42).fit(X).inertia_)

        print('now fitting {} clusters using  Mini batch K-means algorithm'.format(k))



# plot ssm vs k

plt.figure(figsize=(12, 5))

plt.plot(num_clusters, sum_square_error, "g^-")

plt.xticks(num_clusters)

plt.xlabel('Number of Clusters')

plt.ylabel("Sum of Square Distance")

plt.title('Elbow Method')

plt.show()
# create the models and fit 

cluster_predictions = MiniBatchKMeans(n_clusters=8, init_size=1024, batch_size=2048, random_state=42).fit_predict(X)
def plot_tsne_pca(data, labels):

    '''

    This function plots the PCA and t-SNE on 2D plane.

    args:

        data: tf-idf weight matrix

        labels: predictions from K-means

    '''

    # initial set up and random pick up samples

    max_label = max(labels)

    max_items = np.random.choice(range(data.shape[0]), size=2000, replace=False)

    

    # extract eigenvectors that have the most explained variance and feed the eigenvectors to t-SNE

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())

    tsne = TSNE().fit_transform(PCA(n_components=60).fit_transform(data[max_items,:].todense()))

    

    # random pick centain size of data points for visiualization

    idx = np.random.choice(range(pca.shape[0]), size=400, replace=False)

    label_subset = labels[max_items]

    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]

    

    f, ax = plt.subplots(1, 2, figsize=(14, 6))

    

    # plot PCA

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)

    ax[0].set_title('PCA Cluster')

    

    # plot t-SNE

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)

    ax[1].set_title('TSNE Cluster')



# plot PCA and t-SNE reduced data

plot_tsne_pca(X, cluster_predictions)
def get_top_keywords(data, clusters, labels, n_terms):

    '''

    This function displays the top keywords based on tf-idf score.

    '''

    # group tf-idf array based on predictions

    df = pd.DataFrame(data.todense()).groupby(clusters).mean()

    

    # loop through each clusters and print top 10 score words

    for i,r in df.iterrows():

        print('\nCluster {}'.format(i))

        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))



# run the code

get_top_keywords(X, cluster_predictions, vectorizer.get_feature_names(), 10)