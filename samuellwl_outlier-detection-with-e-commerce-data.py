import pandas as pd

import numpy as np
data = pd.read_csv("../input/data.csv")
data.head(10)
def validateString(s): # Tells us if a string has both letters and numbers

    letter_flag = False

    number_flag = False

    for i in s:

        if i.isalpha():

            letter_flag = True

        if i.isdigit():

            number_flag = True

        # Short circuit

        if number_flag and letter_flag:

            return True

    return False



# Preprocessing

data2 = data.copy()

data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x.lower() for x in x.split()]))

data2['product_name'] = data2['product_name'].str.replace('[^\w\s]','')

data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x for x in x.split() if x.isnumeric()==False]))

data2['product_name'] = data2['product_name'].apply(lambda x: " ".join([x for x in x.split() if validateString(x)==False]))
data2.head(10)
data_processed = data2.loc[data['barcode']==8886012805206,:]



# Keeping another copy consisting of unprocessed test

data_org = data.loc[data['barcode']==8886012805206,:]

data_processed.head()
from sklearn.feature_extraction.text import TfidfVectorizer



#define vectorizer parameters

vectorizer = TfidfVectorizer()



# We only want the product names

tfidf = vectorizer.fit_transform(data_processed.iloc[:,0])
from sklearn.cluster import AgglomerativeClustering

tfidf_dense = tfidf.toarray()

hc = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='cosine').fit(tfidf_dense)

hc_result = hc.labels_
# Reset index because we need to perform a join() on indexes

data_org = data_org.reset_index()



# Create another copy for later use

data_org2 = data_org.copy()



# Convert our result into a DataFrame

hc_result = pd.DataFrame(hc_result)



# Perform the join on index

data_org = data_org.join(hc_result)



# Rename columns

data_org = data_org.rename(index=str, columns={'index':'original_index', 0:'result'})
print(data_org.loc[data_org['result']==0,:])
data_org.head(10)
from sklearn.metrics.pairwise import cosine_similarity

# Similarity

cos_sim = cosine_similarity(tfidf)



# Convert similarity to distance

dist = 1 - cos_sim
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2).fit(dist)

km_result = km.labels_

km_result = pd.DataFrame(km_result)

data_org2 = data_org2.join(km_result)

data_org2 = data_org2.rename(index=str, columns={'index':'original_index', 0:'result'})
data_org2.head(10)