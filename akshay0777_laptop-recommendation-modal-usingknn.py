%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pandas import DataFrame 

import nltk



from sklearn.neighbors import NearestNeighbors

from sklearn.linear_model import LogisticRegression

from sklearn import neighbors

from scipy.spatial.distance import cosine

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



import re

import string

from wordcloud import WordCloud, STOPWORDS

from sklearn.metrics import mean_squared_error
df = pd.read_csv("../input/Final_Dataframe.csv")
print(df.columns)

print(df.shape)
count = df.groupby("brand", as_index=False).count()

mean = df.groupby("brand", as_index=False).mean()



dfMerged = pd.merge(df, count, how='right', on=['brand'])

dfMerged
dfMerged["totalReviewers"] = dfMerged["ratings_5max_y"]

dfMerged["overallScore"] = dfMerged["ratings_5max_x"]

dfMerged["summaryReview"] = dfMerged["graphics_card_y"]



dfNew = dfMerged[['brand','summaryReview','overallScore',"totalReviewers"]]

dfNew.head()
dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)

dfCount = dfMerged[dfMerged.totalReviewers >= 0]

dfCount
dfProductReview = df.groupby("laptop_name", as_index=False).mean()

ProductReviewSummary = dfCount.groupby("laptop_name_x")["summaryReview"].apply(list)

ProductReviewSummary = pd.DataFrame(ProductReviewSummary)

ProductReviewSummary.to_csv("ProductReviewSummary.csv")
dfCount




dfProductReview
dfProductReview = dfProductReview.rename(columns={'laptop_name': 'laptop_name_x'})
dfProductReview
ProductReviewSummary


df3 = pd.read_csv("ProductReviewSummary.csv")

df3 = pd.merge(df3, dfProductReview, on="laptop_name_x", how='inner')
df3 = df3[['laptop_name_x','summaryReview','ratings_5max']]

df3.shape
df3
#reset index and drop duplicate rows

df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)

df3 = df3.reset_index()
reviews = df3["summaryReview"] 

countVector = CountVectorizer(max_features = 300, stop_words='english') 

transformedReviews = countVector.fit_transform(reviews) 



dfReviews = DataFrame(transformedReviews.A, columns=countVector.get_feature_names())

dfReviews = dfReviews.astype(float)
dfReviews.to_csv("dfReviews.csv")

dfReviews



# First let's create a dataset called X

X = np.array(dfReviews)

 # create train and test

tpercent = 0.7

tsize = int(np.floor(tpercent * len(dfReviews)))

dfReviews_train = X[:tsize]

dfReviews_test = X[tsize:]

#len of train and test

lentrain = len(dfReviews_train)

lentest = len(dfReviews_test)
print(lentrain)

print(lentest)
neighbor = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(dfReviews_train)



# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.

distances, indices = neighbor.kneighbors(dfReviews_train)
#find most related products

for i in range(lentest):

    a = neighbor.kneighbors([dfReviews_test[i]])

    related_product_list = a[1]



    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    print ("Based on product reviews, for ", df3["laptop_name_x"][lentrain + i] ," average rating is ",df3["ratings_5max"][lentrain + i])

    print ("The first similar product is ", df3["laptop_name_x"][first_related_product] ," average rating is ",df3["ratings_5max"][first_related_product])

    print ("The second similar product is ", df3["laptop_name_x"][second_related_product] ," average rating is ",df3["ratings_5max"][second_related_product])

    print ("-----------------------------------------------------------")
print (accuracy_score(df5_test_target, knnpreds_test))
print (accuracy_score(df5_test_target, knnpreds_test))
# First let's create a dataset called X

X = np.array(dfReviews)

 # create train and test

tpercent = 0.85

tsize = int(np.floor(tpercent * len(dfReviews)))

dfReviews_train = X[:tsize]

dfReviews_test = X[tsize:]

#len of train and test

lentrain = len(dfReviews_train)

lentest = len(dfReviews_test)
 #Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.

neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)



# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.

distances, indices = neighbor.kneighbors(dfReviews_train)
#find most related products

for i in range(lentest):

    a = neighbor.kneighbors([dfReviews_test[i]])

    related_product_list = a[1]



    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    print ("Based on product reviews, for ", df3["laptop_name_x"][lentrain + i] ," average rating is ",df3["ratings_5max"][lentrain + i])

    print ("The first similar product is ", df3["laptop_name_x"][first_related_product] ," average rating is ",df3["ratings_5max"][first_related_product])

    print ("The second similar product is ", df3["laptop_name_x"][second_related_product] ," average rating is ",df3["ratings_5max"][second_related_product])

    print ("-----------------------------------------------------------")
neighbor = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(dfReviews_train)

distances, indices = neighbor.kneighbors(dfReviews_train)
cluster = df.groupby("ratings_5max")["laptop_name"].apply(list)

cluster = pd.DataFrame(cluster)

cluster.to_csv("cluster.csv")

cluster1 = pd.read_csv("cluster.csv")

cluster1["summaryClean"] = cluster1["laptop_name"].apply(cleanReviews)
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=500,

        max_font_size=30, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(8, 8))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(cluster1["laptop_name"][0], title = "Review Score One")
show_wordcloud(cluster1["laptop_name"][4], title = "Review Score Five")