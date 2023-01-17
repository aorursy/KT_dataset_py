#!pip install gensim
#!pip install nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

import sqlite3
import string
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import gensim
import string
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import os
from tqdm import tqdm
#!pip install kaggle
#!rm -r  kaggle  # remove directory .kaggle
#!rm -r .kaggle  # remove directory .kaggle
#!mkdir .kaggle  # create directory .kaggle
#!mkdir  kaggle  # create directory kaggle

#from google.colab import files
#files.upload()
#!cp kaggle.json ~/.kaggle/
#import json
#token = {"username":"ashwani0187","key":"f9c7f8263297d23fef670995e74ae664"}
#with open('/content/.kaggle/kaggle.json', 'w') as file:
 # json.dump(token, file)   #save the token file
  
#!chmod 600 /content/.kaggle/kaggle.json #change permissions
#!kaggle config set -n path -v{/content/kaggle} #change directory path
#!kaggle datasets download -d snap/amazon-fine-food-reviews -p /content/kaggle/amazon-fine-food-reviews
#%cd /content/kaggle/amazon-fine-food-reviews
#!ls -a
#import os
#os.getcwd()
#!ls
#!unzip /content/kaggle/amazon-fine-food-reviews/amazon-fine-food-reviews.zip
#!ls
#connecting to the database:
con = sqlite3.connect('../input/database.sqlite')
# Extracting out the positive and negative features 
amazon_featured_reviews = pd.read_sql_query("""SELECT * FROM REVIEWS WHERE SCORE != 3""" , con)

print(amazon_featured_reviews.shape)

# Creating the partition function returning the positive or negative reviews and appending them in the Score column in place 
# of ratings given:

def partition(x):
        if x < 3:
            return 'negative'
        else :
            return 'positive'
        
        
pos_neg_reviews_df = amazon_featured_reviews['Score'].map(partition)
print(type(pos_neg_reviews_df) , 'pos_neg_reviews_df' , pos_neg_reviews_df.shape)
print('type(amazon_featured_reviews):' , type(amazon_featured_reviews))
amazon_featured_reviews['Score'] = pos_neg_reviews_df
amazon_featured_reviews.shape
amazon_featured_reviews.head()

duplicated_data = amazon_featured_reviews.duplicated(subset={'UserId','ProfileName','Time','Summary','Text'} , keep='first')
duplicated_data = pd.DataFrame(duplicated_data , columns=['Boolean'])
print(duplicated_data.head(5))
print(duplicated_data['Boolean'].value_counts(dropna=False)) #gives me the total no of the duplicates

#The total no of duplicates here in the amazon_featured_reviews are:
print("total no of duplicates here in the amazon_featured_reviews are:",duplicated_data[duplicated_data['Boolean']==True].count())

#dropping the duplicates:
final = amazon_featured_reviews.sort_values(by='ProductId',kind='quicksort',ascending=True,inplace=False)
final = final.drop_duplicates(subset={'UserId','ProfileName','Time','Text'} , keep='first', inplace=False)
print('\n','DataFrame final shape before removing helpfullness data :', final.shape)

#Also removing the instances where HelpfulnessNumerator >= HelpfulnessDenominator:
final = final[final['HelpfulnessNumerator'] <= final['HelpfulnessDenominator']]
print('final', final.shape)
#Finding the books data in the amazon_featured_reviews using the regex
import re
print(final.columns)
def analyzing_summary_book(filtered_data , regex):
    
    mask_summary = filtered_data.Summary.str.lower().str.contains(regex) 
    mask_text =    filtered_data.Text.str.lower().str.contains(regex)
    print(len(filtered_data[mask_summary].index) , len(filtered_data[mask_text].index))
    print('initial shape of the filtered_data' , filtered_data.shape)
    filtered_data.drop(filtered_data[mask_summary].index , inplace=True , axis=0)
    filtered_data.drop(filtered_data[mask_text].index , axis=0 , inplace=True)
#Removing the Books reviews we get below final dataframe:
print('final shape before removing books reviews:' , final.shape)
analyzing_summary_book(final , re.compile(r'reading|books|book|read|study|learn|poems|music|story'))

print('final shape after removing the book reviews:' , final.shape)

#Computing the proportion of positive and negative class labels in the DataFrame:
final['Score'].value_counts()
#Text_preprocessing of the Text data, let's see how the text looks like and howw much unwanted things are there in the data:
final['Text'].values[0:2] #return array of all columns values
#  I have the final pandas dataFrame let's print it and analyze the html tags in it:
final.shape
#Let's print out the html tags in the final dataframe:

import re
i = 0
for sentence in final['Text'].values:
    pattern = re.compile('<.*?>')
    if(len(re.findall(pattern , sentence))):
        print(sentence)
        print(i)
        break
        
i+=1    
#nltk.download("popular")
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

stop = set(stopwords.words('english'))
print(stop)
print('\n' , 'length of stopwords set' , len(stop))

print("*" * 30)

sno = SnowballStemmer('english')
print(sno.stem('tasty'))
#Difference between sentence tokenize and word tokenize:

from nltk.tokenize import sent_tokenize , word_tokenize
text ="this's a sent tokenize test. this is sent two. is this sent three? sent 4 is cool! Now it's your turn"
sen = sent_tokenize(text)
word_list = word_tokenize(text)
print(sen)
print(word_list)

def clean_htmlTags(sentence):
    pattern = re.compile('<.*?>')
    cleaned_text = re.sub(pattern , '' , sentence)
    return cleaned_text

def clean_punc(sentence):
    cleaned = re.sub(r'[!|#|,|?|\'|"]' , r' ' , sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]' ,r' ' , cleaned)
    return cleaned

#The below code will remove all the html tags , punctuation marks , uppercase to lowercase conversion only if length of the words
# are greater than 2 and are alphanumeric . Further we perform the Stemming of the each word in the each document.
all_positive_words = []
all_negative_words = []
i = 0
str_temp = ' '
final_string = []
for sent in final['Text'].values:
    filtered_sentence=[]
    sent = clean_htmlTags(sent)
    for w in sent.split():
        for clean_word in clean_punc(w).split():
            if((clean_word.isalpha()) and (len(clean_word) > 2)):
                if(clean_word.lower() not in stop):
                    s = (sno.stem(clean_word.lower())).encode('utf-8')
                    filtered_sentence.append(s)
                    if((final['Score'].values)[i] == 'positive'):
                        all_positive_words.append(s)
                    if((final['Score'].values)[i] == 'negative'):
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue

    str_temp = b" ".join(filtered_sentence)
    final_string.append(str_temp)
    i+=1
#Now I have a final_string of list of each review and append it to the new columns of the final data frame:

final['CleanedText'] = final_string
final['CleanedText'] = final['CleanedText'].str.decode('utf-8')
final.shape
#Storing the data to the database:
conn = sqlite3.connect('final_cleaned.sqlite')
c = conn.cursor()
final.to_sql('Reviews' , conn , if_exists='replace' , schema=None )
conn.close()
if os.path.isfile('final_cleaned.sqlite'):
    conn = sqlite3.connect('final_cleaned.sqlite')
    final_new = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, conn)
    conn.close()
else:
    print("Please the above cell")
#final_cleaned_data = final_new.drop('index',axis=0)
final_cleaned = final_new.drop(columns=['index'],inplace=False,axis=0)

#Now lets take 2500 around data points each of positive and negative review for faster processing the further data:

final_subset = final.groupby('Score').apply(lambda x : x.sample(frac = 0.03))
print(final['Score'].value_counts())
print(final_subset['Score'].value_counts())
print('final_subset shape is :' , final_subset.shape)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
final_counts = cv.fit_transform(final_subset['CleanedText'].values)

print(type(final_counts))
print(final_counts.get_shape())
print(len(cv.get_feature_names()))
print(cv.get_feature_names()[1:100])
# Before plotting the t-SNE plot we will perform TruncatedSVD operation the Bag of words vector so as to perform 
# the dimensionality reduction:

# Create a TSVD with 1000 dimension :
tsvd = TruncatedSVD(n_components=1000)

# Conduct TSVD on sparse matrix final_counts:
final_counts = tsvd.fit(final_counts).transform(final_counts)
#Let's print the properties of the truncated sparse matrix :
print(final_counts.shape)
print(type(final_counts))
print(final_counts)
#Scaling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
final_counts = scaler.fit_transform(final_counts)
from sklearn.manifold import TSNE
model = TSNE(n_components=2 , random_state=None , perplexity=50 , n_iter=750)

tsne_bow_data = model.fit_transform(final_counts)
label = final_subset['Score']
print(tsne_bow_data.shape)
print(label.shape)


tsne_bow_data=np.vstack((tsne_bow_data.T,label)).T
tsne_df=pd.DataFrame(data=tsne_bow_data,columns=("Dimension_1","Dimension_2","label"))

#Plotting the 2D TSNE results:
sns.FacetGrid(tsne_df,hue='label',size=10).map(plt.scatter,'Dimension_1','Dimension_2').add_legend()
plt.title('tSNE for BoW With perplexilty=50')
plt.show()

#Let me calculate the Frequency Distribution of the words:

print('length of the positive words' ,len(all_positive_words))
print('length of the negative' ,len(all_negative_words))

freq_dist_positive = nltk.FreqDist(all_positive_words)
freq_dist_negative = nltk.FreqDist(all_negative_words)

print('Most Common positive words:' , freq_dist_positive.most_common(20))
print('Most Common neagtive words:' , freq_dist_negative.most_common(20))
#Since in the above frequency distribution some words are same in the two freqdist so let's use bi grams and n- grams which 
# preserve the internal sequence information between the words which is detroyed in the case of uni grams/BoW.

#Creating the bi-Grams:
c_vector = CountVectorizer(ngram_range=(1,2) , min_df = 5)
final_counts_bigrams = c_vector.fit_transform(final_subset['CleanedText'].values)
print(final_counts_bigrams.get_shape())
print(type(final_counts_bigrams))
print(final_counts_bigrams.get_shape()[1])
#Uni Gram gram Tf-IDF Vector
tfidf_vector = TfidfVectorizer(ngram_range=(1,1) , min_df=5 )
tfidf_count_values = tfidf_vector.fit_transform(final_subset['CleanedText'].values)
print(type(tfidf_count_values))
print(tfidf_count_values.get_shape())
print(tfidf_count_values.get_shape()[1])
# Before plotting the t-SNE plot we will perform TruncatedSVD operation for the TFIDF vector so as to perform 
# the dimensionality reduction:

# Create a TSVD with 1000 dimension :
tsvd = TruncatedSVD(n_components=2000)

# Conduct TSVD on sparse matrix final_counts:
tfidf_count_values = tsvd.fit(tfidf_count_values).transform(tfidf_count_values)
#Scaling
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
tfidf_standard = scaler.fit_transform(tfidf_count_values)
#tSNE plot:-

model = TSNE(n_components=2 , random_state=None , perplexity=45 , n_iter=750)

tsne_data = model.fit_transform(tfidf_standard)
label = final_subset['Score']
print(tsne_data.shape)
print(label.shape)


tsne_data=np.vstack((tsne_data.T,label)).T
tsne_df=pd.DataFrame(data=tsne_data,columns=("Dimension_1","Dimension_2","Label"))

#Plotting the 2D TSNE results:
sns.FacetGrid(tsne_df,hue='Label',size=7).map(plt.scatter,'Dimension_1','Dimension_2').add_legend()
plt.title('tSNE for TFIDF With perplexilty=45')
plt.show()
#Lets get the vaues of the features of some indexes in the sparse tf_idf vector:

features = tfidf_vector.get_feature_names()
print(features[500:580])

# Now we will Train our own model using Word2vec:
list_of_sentence=[]
for sent in final_subset['CleanedText'].values:
    list_of_sentence.append(sent.split())
print(final_subset['CleanedText'].values[0])
print(list_of_sentence[0])

# Creating the gensim model
model = gensim.models.Word2Vec(list_of_sentence , min_count=5 , size=50 , workers=4)
model.wv.similarity('man' , 'woman')
model.wv.most_similar('tasti')
#Let's get our trained model vocabulary:

vocab_list = list(model.wv.vocab)
print("Words that exist more than 5 times are :" , len(vocab_list))
print(vocab_list[0:60])
#Computing the Average word2vec:
sent_vect= [] #this will hold the all values of the vectors of each words
for sen in tqdm(list_of_sentence):
    sen_vec = np.zeros(50) 
    word_count=0
    for word in sen:
        if word in vocab_list:
            vector_of_current_word = model.wv[word]
            sen_vec+=vector_of_current_word
            word_count+=1
    if word_count != 0:
        sen_vec/=word_count
    sent_vect.append(sen_vec)

print(len(sent_vect))

print(len(sent_vect[0]))
        
#Converting the list type to array type of sent_vect we computed:
sent_vect = np.array(sent_vect)
type(sent_vect)
# Let's plot the t-SNE plot the average word to vector :
# here we have computed all the sentences as the vector using the avgw2v algorithm

model = TSNE(n_components=2 , random_state=None , perplexity = 45 , n_iter =750)

#Let's fit the standardised data into the tsne model:

scaled_vectors = StandardScaler().fit_transform(sent_vect)

#Since all the vectors are densed so there is no need of TruncatedSVD

tsne_data = model.fit_transform(scaled_vectors)
label = final_subset["Score"]
tsne_data = np.vstack((tsne_data.T , label)).T

tsne_df = pd.DataFrame(data=tsne_data , columns=['Dimension_1' , 'Dimension_2', 'Label' ])

#Plotting the tsne data of Avg W2V in 2D:

sns.FacetGrid(tsne_df , hue='Label' , size=7).map(plt.scatter , 'Dimension_1' , 'Dimension_2').add_legend()
plt.title("tSNE for the Avg W2V with perplexity = 45 and n_iter = 750")
plt.show()

# S = ["abc def pqr", "def def def abc", "pqr pqr def"]
tfidf_model = TfidfVectorizer()
tf_idf_matrix = tfidf_model.fit_transform(final_subset['CleanedText'].values)
# we are converting a dictionary with word as a key, and the tfidf as a value
dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
# Creating the gensim model
model = gensim.models.Word2Vec(list_of_sentence , min_count=5 , size=50 , workers=4)
# TF-IDF weighted Word2Vec
tfidf_feat = tfidf_model.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in tqdm(list_of_sentence): # for each review/sentence 
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in vocab_list:
            vec = model.wv[word]
            tf_idf = dictionary[word]*sent.count(word)
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1
    
print('\n' , len(tfidf_sent_vectors))

print(len(tfidf_sent_vectors[0]))
#Converting thr list to array type of the tfidf_sent_vectors:
tfidf_sent_vectors = np.array(tfidf_sent_vectors)
type(tfidf_sent_vectors)
# Let's plot the t-SNE plot the average word to vector :
# here we have computed all the sentences as the vector using the avgw2v algorithm

model = TSNE(n_components=2 , random_state=None , perplexity = 45 , n_iter =750)

#Let's fit the standardised data into the tsne model:

scaled_vectors = StandardScaler().fit_transform(tfidf_sent_vectors)

#Since all the vectors are densed so there is no need of TruncatedSVD

tsne_data = model.fit_transform(scaled_vectors)
label = final_subset["Score"]
tsne_data = np.vstack((tsne_data.T , label)).T

tsne_df = pd.DataFrame(data=tsne_data , columns=['Dimension_1' , 'Dimension_2', 'Label' ])

#Plotting the tsne data of Avg W2V in 2D:

sns.FacetGrid(tsne_df , hue='Label' , size=7).map(plt.scatter , 'Dimension_1' , 'Dimension_2').add_legend()
plt.title("tSNE for the Avg W2V with perplexity = 45 and n_iter = 750")
plt.show()


