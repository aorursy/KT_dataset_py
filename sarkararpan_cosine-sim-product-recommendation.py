# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # Processing

import time # Time handling



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.feature_extraction.text import TfidfVectorizer # Word vectorizer

os.system("cls") # clear screen

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("STARTING DATA IMPORT AND PRE-PROCESSING")



metadata = pd.read_csv('../input/prod.csv', low_memory=False)# import data from database

metadata = metadata.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  # convert to lower case to avoid discrepancies
metadata.head()
metadata.columns.values
metadata = metadata[['prod_name','prod_cat','prod_cst','prod_weight','prod_features']] # extract feature

metadata.head() #view data
# handle inaccuracy

metadata = metadata.fillna('')

print(metadata['prod_features'].head()) #check feature range
# Call TFIDF 

tfidf = TfidfVectorizer(stop_words='english')

#Fit data to tfidf

tfidf_matrix = tfidf.fit_transform(metadata['prod_features'])

#Check

print(tfidf_matrix.shape)

print(tfidf.get_feature_names())
from sklearn.metrics.pairwise import linear_kernel



#Calculate cosine similarity matrix mat(A) * mat(A) = mat(a)^2

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)



#Check cosine matrix

print(cosine_sim.shape)

print(cosine_sim[1])
indices = pd.Series(metadata.index, index=metadata['prod_name']).drop_duplicates()

#time.sleep(5)

print(indices[:5]) #check
def get_recommendations(name, cosine_sim):



    #time.sleep(6)

    indices = pd.Series(metadata.index, index=metadata['prod_name']).drop_duplicates()

    idx = indices[name]



    # Get the pairwsie similarity scores of all products with that product

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the products based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 5 most similar products

    sim_scores = sim_scores[1:6]



    # Get the product indices

    prod_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar product

    return metadata['prod_name'].iloc[prod_indices]

result = get_recommendations('apple', cosine_sim) #change product here

result #check
from matplotlib import pyplot as plt

#call matplotlib

plt.plot(result,'b-') #plot results

plt.plot(metadata['prod_name'], 'ro') #plot stock

plt.title('Distribution')

plt.ylabel("Product")

plt.xlabel("Cosine Similarity")
def clean_data(x):

        return str.lower(str(x).replace(" ", "")) #remove space



#declare features to be cleaned

features = ['prod_cat','prod_cst','prod_weight','prod_features']



for feature in features:

    metadata[feature] = metadata[feature].apply(clean_data) #apply clean method



metadata.head()

#Join all features with space as delimiter to create stringified description

def create_comb(x):

    return ' '+ x['prod_cat'] + ' ' +x['prod_cst'] + ' ' + x['prod_weight'] + ' '+x['prod_features'].replace(","," ")



metadata['comb'] = metadata.apply(create_comb, axis=1) #apply combine factor

    

metadata[['comb']].head(2) #check
# Import CountVectorizer and create the count matrix

from sklearn.feature_extraction.text import CountVectorizer



count = CountVectorizer(stop_words='english')

#fit stringified description

count_matrix = count.fit_transform(metadata['comb'])



count_matrix.shape

from sklearn.metrics.pairwise import cosine_similarity



#calculate mat(A) * mat(A) = mat(A)^2

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
metadata = metadata.reset_index()

#Map to product names

indices = pd.Series(metadata.index, index=metadata['prod_name'])

result2 = get_recommendations('apple', cosine_sim2) #change product here

result2
plt.plot(result2, 'y-')

plt.plot(metadata['prod_name'], 'ro')

plt.title('Distribution')

plt.ylabel("Product")

plt.xlabel("Cosine Similarity")
plt.plot(cosine_sim2, 'ro')

plt.plot(cosine_sim ,'b.')

plt.title('Distribution MetaData vs Features')
coll_data = ['apple','lemon','potatoes','mangoes','kiwi']

recc_1 = get_recommendations(coll_data[0],cosine_sim2)

recc_2 = get_recommendations(coll_data[1],cosine_sim2)

recc_3 = get_recommendations(coll_data[2],cosine_sim2)

recc_4 = get_recommendations(coll_data[3],cosine_sim2)

recc_5 = get_recommendations(coll_data[4],cosine_sim2)

    #print(recc_1[0])

union = list(set(recc_1) | set(recc_2) | set(recc_3) | set(recc_4) | set(recc_5))

union
print("CALCULATING SIMILARITIES IN SHOPPING PATTERN")

my_dict = {}

for x in union:

    my_dict[x] = 1



for x in union:

    if x in set(recc_1):

        my_dict[x] += 1

    if x in set(recc_2):

        my_dict[x] += 1

    if x in set(recc_3):

        my_dict[x] += 1

    if x in set(recc_4):

        my_dict[x] += 1

    if x in set(recc_5):

        my_dict[x] += 1
my_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1], reverse=True)}

my_dict
top_recc = list(my_dict.keys())

for x in range(5):

    print(top_recc[x])
listed = list(my_dict.items())

listed1 = listed[0:5]

listed2 = listed[5:10]

x,y = zip(*listed1)

x1,y1 = zip(*listed2)

x2,y2 = zip(*listed)

plt.plot(x,y,'r-')

plt.title('Relative Demand Plot')

plt.plot(x1,y1,'b-')

plt.show()
plt.plot(recc_1 , 'r.-')

plt.plot(recc_2 , 'b--')

plt.plot(recc_3 , 'g-')

plt.plot(recc_4 , 'r-')

plt.plot(recc_5 , 'b-')

plt.plot(x2,y2 , 'y-')