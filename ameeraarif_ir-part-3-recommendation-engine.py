import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import nltk
import os
import json
import math
import csv
import re
import time
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from collections import Counter
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

data_read=pd.read_csv("/kaggle/input/amazon-and-best-buy-electronics/DatafinitiElectronicsProductData.csv",delimiter=',', 
                           names = ['id1', 'asins1', 'brand1', 'categories1', 'colors1', 'dateAdded1'
                                    , 'dateUpdated1', 'dimensions1', 'ean1', 'imageURL1', 'keys1', 'manufacturer1'
                                    , 'manufacturerno1', 'name1', 'primarycategory1', 'reviewDate1', 'reviewDateSeen1'
                                    , 'reviewsDoRecom1','reviewsNumHelp1','reviewsRating1','reviewsSourceURL1',
                                   'reviewsText1','reviewsTitle1','reviewsUsername1','sourceURL1','upc1','weight1',])

data_read.head()
print("Aceessing 4 columns of choice \n")
data4read = data_read[['reviewsUsername1', 'id1','reviewsRating1', 'reviewsText1']]
print("Done!")
data4read.head()
print("Shape of chosen columns:\n", data4read.shape)
print("Type of chosen columns:\n",data4read.dtypes)
print('Information about data: \n')
data4read.info()
missing_cols=data4read.isnull()
missing_cols=missing_cols.sum()
print('Number of missing values across columns: \n')
print(missing_cols)
with sns.axes_style('white'):
    g = sns.factorplot("reviewsRating1", data=data4read, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
print("The whole data: ")

un_reviews=np.unique(data4read.reviewsUsername1)
len_un_reviews=len(np.unique(data4read.reviewsUsername1))
print("Total number of Users:\n", len_un_reviews)
un_products=np.unique(data4read.id1)
len_un_products=len(un_products)
print("Total number of products:\n", len_un_products)
un_ratings=data4read.shape[0]
print("Total number of ratings:\n",un_ratings)
reviews_read=data4read[['reviewsText1']]
print(reviews_read)
#remove columns and now we have only one column 'reviews.txt'
import csv

input_file = '/kaggle/input/amazon-and-best-buy-electronics/DatafinitiElectronicsProductData.csv'
output_file = 'output.csv'
cols_to_remove = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26] # Column indexes to be removed (starts at 0)

cols_to_remove = sorted(cols_to_remove, reverse=True) # Reverse so we remove from the end first
row_count = 0 # Current amount of rows processed

with open(input_file, "r") as source:
    reader = csv.reader(source)
    with open(output_file, "w", newline='') as result:
        writer = csv.writer(result)
        for row in reader:
            row_count += 1
            print('\r{0}'.format(row_count), end='') # Print rows processed
            for col_index in cols_to_remove:
                del row[col_index]
            writer.writerow(row)
output_file = 'output.csv'
csv_file = 'output.csv'
txt_file = 'reviews.txt'
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
stop_words = set(stopwords.words('english'))
 
# Open and read in a text file.
with open('output.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader: 
        txt_line = csv_file.read()
        txt_words = txt_line.split()

# stopwords found counter.
sw_found = 0
 
# If each word checked is not in stopwords list,
# then append word to a new text file.
for check_word in txt_words:
    if not check_word.lower() in stop_words:
        # Not found on stopword list, so append.
        appendFile = open('stopwords-removed.csv','a')
        appendFile.write(" "+check_word)
        appendFile.close()
    else:
        # It's on the stopword list
        sw_found +=1
        print(check_word)
 
print(sw_found,"stop words found and removed")
print("Saved as 'stopwords-removed.csv' ")

with open('output.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader: 
        txt_line = csv_file.read()
        #txt_words = txt_line.split()
        print("POS::")
        for x in txt_line:
            print(nltk.pos_tag(x))
d=0
porter = PorterStemmer()
with open('/kaggle/input/amazon-and-best-buy-electronics/DatafinitiElectronicsProductData.csv', "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for lines in csv_reader: 
        txt_line = csv_file.read()
        txt_words = txt_line.split()
print("Porter Stemmer::")
for check_word in txt_words:
    print(check_word,porter.stem(check_word))
data4read.drop(['reviewsText1'], axis=1,inplace=True)
a=data4read.groupby(by='reviewsUsername1')['reviewsRating1'].count()
rated_products_per_user = a.sort_values(ascending=False)
rated_products_per_user.head()
rated_products_per_user.describe()

sum_rated_products_per_user=sum(rated_products_per_user >= 10)
print('No of rated product more than 10 per user : {}\n'.format(sum_rated_products_per_user) )

new_df=data4read.groupby("id1").filter(lambda x:x['reviewsRating1'].count() >=20)

no_of_ratings_per_product = new_df.groupby(by='id1')['reviewsRating1'].count().sort_values(ascending=True)

fig = plt.figure(figsize=plt.figaspect(.25))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.xlabel('Item')
plt.ylabel('No of ratings per item')
plt.title('Ratings per item')
plt.yticks(np.arange( 0.005))
ax.set_xticklabels([])

plt.show()
c=new_df.groupby('id1')['reviewsRating1'].count().sort_values(ascending=False)
c.head()
pop_products = pd.DataFrame(new_df.groupby('id1')['reviewsRating1'].count())
most_popular = pop_products.sort_values('reviewsRating1', ascending=True)

most_popular.head(25).plot(kind = "barh", title="Popular items ratings")
new_df=data4read.groupby("id1").filter(lambda x:x['reviewsRating1'].count() >=10)
reader = Reader(rating_scale=(1, 5))
new_data = Dataset.load_from_df(new_df,reader)
train_set, test_set = train_test_split(new_data, test_size=0.25,random_state=10)
algo4data = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo4data.fit(train_set)
test_prediction1 = algo4data.test(test_set)
test_prediction1
algo4data = KNNWithMeans(k=5, sim_options={'name': 'cosine', 'user_based': False})
algo4data.fit(train_set)
test_prediction2 = algo4data.test(test_set)
test_prediction2
algo4data = KNNWithMeans(k=3, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo4data.fit(train_set)
test_prediction3 = algo4data.test(test_set)
test_prediction3
new_df1=new_df.head(1000)
matrix1 = new_df1.pivot_table(values='reviewsRating1', index='reviewsUsername1', columns='id1', fill_value=0, aggfunc='first')
matrix1.head()
transposed_matrix1 = matrix1.T
transposed_matrix1.head()
SVD = TruncatedSVD(n_components=8)
decomposed_matrix1 = SVD.fit_transform(transposed_matrix1)
correlation_matrix1 = np.corrcoef(decomposed_matrix1)
d=transposed_matrix1.index[5]
iid = "AWIm0C3TYSSHbkXwx3S6"

product_names1 = list(transposed_matrix1.index)
pid1 = product_names1.index(iid)
print("Index no of item id purchased by user: \n")
print(pid1)
correlation_pid = correlation_matrix1[pid1]
correlation_pid
r=transposed_matrix1.index[correlation_pid > -0.20858196]
recommend1 = list(r)

recommend1.remove(iid) #removing itself from recommendations

recommend1[0:3]