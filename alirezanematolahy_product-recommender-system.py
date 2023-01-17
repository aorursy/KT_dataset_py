# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

import nltk



from sklearn.neighbors import NearestNeighbors

from sklearn import neighbors

from scipy.spatial.distance import cosine

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score





from sklearn.feature_selection import SelectKBest

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



import re

import string

from wordcloud import WordCloud, STOPWORDS
df = pd.read_csv('../input/amazon-fine-food-reviews/Reviews.csv')
#Basic Information shape and columns

print(df.columns)

print(df.shape)
#compute the count and mean value as group by the products

count = df.groupby("ProductId", as_index=False).count()

mean = df.groupby("ProductId", as_index=False).mean()



#merge two dataset create df1

df1 = pd.merge(df, count, how='right', on=['ProductId'])



#rename column

df1["Count"] = df1["UserId_y"]

df1["Score"] = df1["Score_x"]

df1["Summary"] = df1["Summary_x"]



#Create New datafram with selected variables

df1 = df1[['ProductId','Summary','Score',"Count"]]

df1
#choose only products have over 100 reviews

df1 = df1.sort_values(by=['Count'], ascending=False)

df2 = df1[df1.Count >= 100]

df2
#create new dataframe as combining all summary with same product Id

df4 = df.groupby("ProductId", as_index=False).mean()

combine_summary = df2.groupby("ProductId")["Summary"].apply(list)

combine_summary = pd.DataFrame(combine_summary)

combine_summary.to_csv("combine_summary.csv")

combine_summary
#create with certain columns

df3 = pd.read_csv("combine_summary.csv")

df3 = pd.merge(df3, df4, on="ProductId", how='inner')

df3 = df3[['ProductId','Summary','Score']]

df3
#function for tokenizing summary

cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    sentence = sentence.lower()

    sentence = cleanup_re.sub(' ', sentence).strip()

    sentence = " ".join(nltk.word_tokenize(sentence))

    return sentence
#reset index and drop duplicate rows

df3["Summary_Clean"] = df3["Summary"].apply(cleanup)

df3 = df3.drop_duplicates(['Score'], keep='last')

df3 = df3.reset_index()

df3
docs = df3["Summary_Clean"] 

vect = CountVectorizer(max_features = 100, stop_words='english') 

X = vect.fit_transform(docs) 



df5 = pd.DataFrame(X.A, columns=vect.get_feature_names())

df5 = df5.astype(int)

df5
#save 

df5.to_csv("df5.csv")
# First let's create a dataset called X

X = np.array(df5)

 # create train and test

tpercent = 0.9

tsize = int(np.floor(tpercent * len(df5)))

df5_train = X[:tsize]

df5_test = X[tsize:]

#len of train and test

lentrain = len(df5_train)

lentest = len(df5_test)
# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(df5_train)



# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.

distances, indices = nbrs.kneighbors(df5_train)
#find most related products

for i in range(lentest):

    a = nbrs.kneighbors([df5_test[i]])

    related_product_list = a[1]

    

    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    print ("Based on product reviews, for ", df3["ProductId"][lentrain + i] ," and this average Score is ",df3["Score"][lentrain + i])

    print ("The first similar product is ", df3["ProductId"][first_related_product] ," and this average Score is ",df3["Score"][first_related_product])

    print ("The second similar product is ", df3["ProductId"][second_related_product] ," and this average Score is ",df3["Score"][second_related_product])

    print ("-----------------------------------------------------------")
df5_train_target = df3["Score"][:lentrain]

df5_test_target = df3["Score"][lentrain:lentrain+lentest]

df5_train_target = df5_train_target.astype(int)

df5_test_target = df5_test_target.astype(int)



n_neighbors = 3

knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

knnclf.fit(df5_train, df5_train_target)

knnpreds_test = knnclf.predict(df5_test)

print (knnpreds_test)



print(classification_report(df5_test_target, knnpreds_test))

print (accuracy_score(df5_test_target, knnpreds_test))
count = df.groupby("UserId", as_index=False).count()

mean = df.groupby("UserId", as_index=False).mean()



#merge two dataset create df1

df1 = pd.merge(df, count, how='right', on=["UserId"])

#rename column

df1["Count"] = df1["ProductId_y"]

df1["Score"] = df1["Score_x"]

df1["Summary"] = df1["Summary_x"]



#Create New datafram with selected variables

df1 = df1[["UserId",'Summary','Score',"Count"]]

df1
#choose only products have over 100 reviews

df1 = df1.sort_values(by=['Count'], ascending=False)

df2 = df1[df1.Count >= 100]

df2
df4 = df.groupby("UserId", as_index=False).mean()

combine_summary = df2.groupby("UserId")["Summary"].apply(list)

combine_summary = pd.DataFrame(combine_summary)

combine_summary.to_csv("combine_summary.csv")

combine_summary
df3 = pd.read_csv("combine_summary.csv")

df3 = pd.merge(df3, df4, on="UserId", how='inner')

df3 = df3[['UserId','Summary','Score']]
df3["Summary_Clean"] = df3["Summary"].apply(cleanup)
df3 = df3.drop_duplicates(['Score'], keep='last')

df3 = df3.reset_index()
docs = df3["Summary_Clean"] 

vect = CountVectorizer(max_features = 100, stop_words='english') 

X = vect.fit_transform(docs) 

df5 = pd.DataFrame(X.A, columns=vect.get_feature_names())

df5 = df5.astype(int)

df5
df5.to_csv("df5.csv")

kkk  = df.drop_duplicates(['Summary'], keep='last')

kkk = kkk.reset_index()
# First let's create a dataset called X, with 6 records and 2 features each.

X = np.array(df5)



tpercent = 0.95

tsize = int(np.floor(tpercent * len(df5)))

df5_train = X[:tsize]

df5_test = X[tsize:]



lentrain = len(df5_train)

lentest = len(df5_test)



# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(df5_train)



# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.

distances, indices = nbrs.kneighbors(df5_train)
#finding similar user and intereting products

for i in range(lentest):

    a = nbrs.kneighbors([df5_test[i]])

    related_product_list = a[1]

    

    first_related_product = [item[0] for item in related_product_list]

    first_related_product = str(first_related_product).strip('[]')

    first_related_product = int(first_related_product)

    second_related_product = [item[1] for item in related_product_list]

    second_related_product = str(second_related_product).strip('[]')

    second_related_product = int(second_related_product)

    

    print ("Based on  reviews, for user is ", df3["UserId"][lentrain + i])

    print ("The first similar user is ", df3["UserId"][first_related_product], ".") 

    print ("He/She likes following products")

    for i in range(295743):

        if (kkk["UserId"][i] == df3["UserId"][first_related_product]) & (kkk["Score"][i] == 5):

            aaa= kkk["ProductId"][i]

        

            print (aaa),

    print ("--------------------------------------------------------------------")
df5_train_target = df3["Score"][:lentrain]

df5_test_target = df3["Score"][lentrain:lentrain+lentest]

df5_train_target = df5_train_target.astype(int)

df5_test_target = df5_test_target.astype(int)



n_neighbors = 3

knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

knnclf.fit(df5_train, df5_train_target)

knnpreds_test = knnclf.predict(df5_test)

print ("Predicting review score for testset user are : ", knnpreds_test)



print(classification_report(df5_test_target, knnpreds_test))
cluster = df.groupby("Score")["Summary"].apply(list)
cluster = pd.DataFrame(cluster)

cluster.to_csv("cluster.csv")

cluster1 = pd.read_csv("cluster.csv")
cluster1["Summary_Clean"] = cluster1["Summary"].apply(cleanup)
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
show_wordcloud(cluster1["Summary_Clean"][0], title = "Review Score One")
show_wordcloud(cluster1["Summary_Clean"][1] , title = "Review Score Two")
show_wordcloud(cluster1["Summary_Clean"][2], title = "Review Score Three")
show_wordcloud(cluster1["Summary_Clean"][3], title = "Review Score Four")
show_wordcloud(cluster1["Summary_Clean"][4], title = "Review Score Five")