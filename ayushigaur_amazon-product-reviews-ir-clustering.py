import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import random

import os
os.chdir('/kaggle/input')

amz = pd.read_csv("amazon-reviews.csv.bz2",sep="\t")

amz.sample(5)
amz.review.isna().sum()

#There are 80 missings observations in the data
amz.review[(amz.review.str.len() == 0) | (amz.review == "")].shape[0]

#There are 0 empty reviews in the data
#dropping all NAs

print("Dimensions of dataset before dropping the NAs:",amz.shape)

amz = amz.dropna(subset=['review'])

print("Dimensions of dataset after dropping the NAs:",amz.shape)
np.random.seed(0)

amz_s = amz.sample(5000)

amz_s.head(10)
#Taking a bigger sample of 20k

amz_bs = amz.sample(20000)
vectorizer = CountVectorizer(stop_words='english')

#converting toarrray() to get a dense matrix

V_s = vectorizer.fit_transform(amz_s.review.values).toarray()
#also computing for the bigger subset

V_bs = vectorizer.fit_transform(amz_bs.review.values).toarray()
def idf_transform(word_col):

    #words present in how many documents - df

    w = len(word_col[np.nonzero(word_col)])

    #compute idf for a word

    return np.log(len(word_col)/(w + 1))



def tf_idf(bow):

    #TF matrix

    tf = np.log(bow + 1)

    #1d array

    idf = np.apply_along_axis(idf_transform,0,bow)

    #tf-idf

    return (np.multiply(tf,idf))
#

V_s_tfidf = tf_idf(V_s)
V_bs_tfidf = tf_idf(V_bs)
def kmeans(V,k):

    #select k reviews as centers

    k_center_i = random.sample(range(0,V.shape[0]),k)

    center_v = V[k_center_i, :]

    

    #all reviews

    A_i = np.array([x for x in range(0,V.shape[0])])

    all_v_norm = np.apply_along_axis(np.linalg.norm,1,V)

    

    #clusters - initial

    clusters = [None] * k

    clusters[0] = A_i.tolist()

    for i in range(1,k):

        clusters[i] = []

    

    j=0

    while True:

        #only printing the sizes of first 4 clusters

        print("iteration",j,"cluster0",len(clusters[0]),"cluster1",len(clusters[1]),"cluster2",len(clusters[2]),

              "clusters3",len(clusters[3]))

        #Norm of cluster center vectors

        center_v_norm = np.apply_along_axis(np.linalg.norm,1,center_v)



        #Cosine similarity: 

        #x @ y

        product_v = V @ np.transpose(center_v)

        #divide by norms ||x|| and ||y||

        product_v_n = np.apply_along_axis(np.true_divide,1,product_v,center_v_norm)

        product_v_norm = np.apply_along_axis(np.true_divide,0,product_v_n,all_v_norm)

        #get each review has maximum cosine similarity with which center

        max_center = np.argmax(product_v,axis=1)



        #assign to closest clusters

        clusters_new = [None] * k

        for i in range(k):

            r = np.where(np.array(max_center) == i)

            clusters_new[i] = r[0].tolist()



        if (np.array_equal(clusters,clusters_new)):

            break

        else:

            j = j+1



        #calculate new centers

        for i in range(k):

            reviews = V[clusters_new[i], :]

            center_v[i] = np.mean(reviews,axis=0)

        

        #set old clusters as new clusters

        clusters = clusters_new.copy()

    

    print("Clusters converged after",j+1,"iterations")

    return clusters
clusters = kmeans(V_s_tfidf,6)
amz_s.iloc[clusters[0]][['review']].sample(7)

#This one has kids' products like diapers, pillows, etc
amz_s.iloc[clusters[1]][['review']].sample(7)

#This is one is a mix of musical instrument, baby tubs, strollers, baby gates, car seats
amz_s.iloc[clusters[2]][['review']].sample(7)

#This has baby products - various kinds like teething rings, sheets, pillows, baby books
amz_s.iloc[clusters[3]][['review']].sample(7)

#again some baby products - more related to blankets, feeding (cups, bibs, bowls, spoons)
amz_s.iloc[clusters[4]][['review']].sample(7)

#again some baby products - from bottles, chairs, pacifiers, pumps etc and face creams and some skin products!
amz_s.iloc[clusters[5]][['review']].sample(7)

#beauty products like makeup, nailpolish, etc and some more musical instruments
#using the bigger subset of reviews and k=12

clusters2 = kmeans(V_bs_tfidf,12)
amz_bs.iloc[clusters2[0]][['review']].sample(7)

#These are reviews on mats, high chairs, tray (products related to dining tables, more related to babies)
amz_bs.iloc[clusters2[1]][['review']].sample(7)

#this one has majorly diapers, wipe, faceclothes, bags
amz_bs.iloc[clusters2[2]][['review']].sample(7)

#this has baby feeding products like bottles, pumps, etc
amz_bs.iloc[clusters2[3]][['review']].sample(7)

#this more baby products like crib skirts, bumpers, rail pads, etc
amz_bs.iloc[clusters2[4]][['review']].sample(7)

#this has a lot of different baby toys and books
amz_bs.iloc[clusters2[5]][['review']].sample(7)

#this is mostly beauty products like face masks, face wash, soaps, face wipes, cleansing water
amz_bs.iloc[clusters2[6]][['review']].sample(7)

#this has reviews on baby carriers and gates
amz_bs.iloc[clusters2[7]][['review']].sample(7)

#this is baby products like walkers, car seats, strollers
amz_bs.iloc[clusters2[8]][['review']].sample(7)

#this one is an swful mix of makeup products and toys for kids
amz_bs.iloc[clusters2[9]][['review']].sample(7)

#finally, musical instruments!
amz_bs.iloc[clusters2[10]][['review']].sample(7)

#a lot of nail polishes, some makeup and few coats
amz_bs.iloc[clusters2[11]][['review']].sample(7)

#again baby products - cups, trays, baskets, toys, tubs, baby food