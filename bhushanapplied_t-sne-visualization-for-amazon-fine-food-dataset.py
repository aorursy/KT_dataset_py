"""
Importing the Libraries
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3 # SQL queries handling
import string
import matplotlib.pyplot as plt # Graphs plotting

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix # Confusion Matrix
from sklearn.metrics import roc_curve,auc
from sklearn.manifold import TSNE
import nltk # Natural Language processing Toolkit
from nltk.stem.porter import PorterStemmer #for Stemming
from nltk.corpus import stopwords
import seaborn as sns
"""
Importing tqdm Library to check Time Lapse over the Loops
"""
from tqdm import tqdm
"""
Importing the re Library to handle regular expressions
"""
import re
"""
Import warnings Library to ignore the warnings occring in program
"""
import warnings
warnings.filterwarnings('ignore')
"""
Import gensim model and its features Word2Vec,KeyedVectors
"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler
#Created a Connection to Database
connection = sqlite3.connect('../input/database.sqlite')
# Taking only Reviews which have Rating as 1,2,4,5 not 3.
filtered_review=pd.read_sql_query('''SELECT * FROM Reviews WHERE SCORE != 3''',connection)
filtered_review.shape
"""
Assigning the Polarity based on conditions such that Score > 3 = Positive and Score < 3 = Negative
"""
def partition(x):
    if x>3:
        return 'Positive'
    return 'Negative'

ActualScore=filtered_review['Score']
PositiveNegative=ActualScore.map(partition)
filtered_review['Score']=PositiveNegative
"""
Removed some Book Reviews which are irrelavent to Food DataFrame.
Reference: Comment by Mangesh Ingle on https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/exercise-t-sne-visualization-of-amazon-reviews-1/
"""
def apply_mask_summary(filtered_data,regex_string):
    mask = filtered_review.Summary.str.lower().str.contains(regex_string)
    filtered_review.drop(filtered_review[mask].index, inplace=True)

def apply_mask_text(filtered_review,regex_string):
    mask = filtered_review.Text.str.lower().str.contains(regex_string)
    filtered_review.drop(filtered_review[mask].index, inplace=True)

apply_mask_summary(filtered_review,re.compile(r'\bbook\b'))
apply_mask_summary(filtered_review,re.compile(r'\bread\b'))
apply_mask_summary(filtered_review,re.compile(r'\bbooks\b'))
apply_mask_summary(filtered_review,re.compile(r'\breads\b'))
apply_mask_summary(filtered_review,re.compile(r'\breading\b'))

apply_mask_text(filtered_review,re.compile(r'\bbooks\b'))
apply_mask_text(filtered_review,re.compile(r'\breads\b'))
apply_mask_text(filtered_review,re.compile(r'\bbook\b'))
apply_mask_text(filtered_review,re.compile(r'\bread\b'))
apply_mask_text(filtered_review,re.compile(r'\breading\b'))
filtered_review.shape
"""
A single product can have different flavours, 
commenting on 1 Product reflects on Multiple Flavours
"""
display=pd.read_sql_query("""SELECT * FROM REVIEWS WHERE SCORE !=3 
                             AND
                             USERID='AR5J8UI46CURR' ORDER BY PRODUCTID
                          """,connection)
display
"""
First we are sorting data based on Product ID,
Secondly we are removing duplicate entries based on Same User ID, Profile Name at the same Time Stamp
"""
sorted_data=filtered_review.sort_values('ProductId',axis=0,ascending=True,inplace=False,kind='QuickSort',na_position='last')
non_duplicate_data=sorted_data.drop_duplicates(subset={'UserId',"ProfileName","Time","Text"},keep='first',inplace=False)
non_duplicate_data.shape
"""
HelpfulNessNumerator = Number of People found review useful
HelpfulNessDenominator = Number of People found review useful + Number of People found review not useful
"""
display=pd.read_sql_query("""SELECT * FROM REVIEWS WHERE SCORE !=3
                             AND
                             HELPFULNESSNUMERATOR > HELPFULNESSDENOMINATOR
                          """,connection)
display
"""
Since HelpfulnessNumerator can't be greater than HelpfulnessDenominator, we've to remove these reviews
"""
final=non_duplicate_data[non_duplicate_data.HelpfulnessNumerator<=non_duplicate_data.HelpfulnessDenominator]
final.shape
"""
Data has been sampled here
"""
df1=final[final['Score']=='Positive'].sample(n=2500,random_state=0)
df2=final[final['Score']=='Negative'].sample(n=2500,random_state=0)
combine=df1.append(df2)
combine.shape
"""
There are few words which sholud not be in Stopwords List like not,no,doesn't,etc
"""
total_stopwords=set(stopwords.words('english'))
not_stop=set(("weren't","hasn't","aren't","won't","didn't","mightn","doesn","isn't","haven","wouldn","no","not","mustn","didn","wouldn't","shouldn't","dont't","wasn't","shouldn","mightn't","haven't","needn","needn't","don","doesn't","hadn't","wasn","mustn't","couldn","couldn't","hadn"))
final_stopwords=total_stopwords-not_stop
"""
Data Preprocessing:
1) Only Alphabets should be taken from review.
2) Every word should be in lower case from review.
3) Splitting review in words
4) Applying Stemming
5) Checking whether word is in stopwords wor not.
6) If not then add to corpus.
"""
corpus = []
for sent in tqdm(combine['Text'].values):
    review = re.sub('[^a-zA-Z]', ' ', sent)
    review = review.lower()
    review = review.split()
    sno = nltk.stem.SnowballStemmer('english')
    review = [sno.stem(word) for word in review if not word in final_stopwords]
    review = ' '.join(review)
    corpus.append(review)
"""
New Column in Final DataFrame called CleanedText
CleanedText column obtained after all preprocessing of data
"""
combine['CleanedText']=corpus
combine.shape
"""
t-SNE function defined with todense()
"""
def tSNE_with_dense_array(method,title):
    #Data Standardization
    std_data=StandardScaler(with_mean=False).fit_transform(method)
    std_data=std_data.todense()
    
    #Applying TSNE
    model=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
    tsne_data=model.fit_transform(std_data)
    
    tsne_data=np.vstack((tsne_data.T,combine['Score'])).T
    tsne_df=pd.DataFrame(data=tsne_data,columns=["Dimension_1","Dimension_2","Polarity"])
    
    #Visualization fo TSNE
    pal = {"Negative":"red", "Positive":"blue"}
    sns.FacetGrid(tsne_df, hue="Polarity", size=8, palette = pal).map(plt.scatter, 'Dimension_1','Dimension_2').add_legend() 
    plt.title(title)
    plt.xlabel("Dimension_1")
    plt.ylabel("Dimension_2")
    plt.show()
"""
t-SNE function defined without todense()
"""
def tSNE_without_dense_array(method,title):
    #Data Standardization
    std_data=StandardScaler(with_mean=False).fit_transform(method)
    
    #Applying TSNE
    model=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=1000)
    tsne_data=model.fit_transform(std_data)
    
    tsne_data=np.vstack((tsne_data.T,combine['Score'])).T
    tsne_df=pd.DataFrame(data=tsne_data,columns=["Dimension_1","Dimension_2","Polarity"])
    
    #Visualization fo TSNE
    pal = {"Negative":"red", "Positive":"blue"}
    sns.FacetGrid(tsne_df, hue="Polarity", size=8, palette = pal).map(plt.scatter, 'Dimension_1','Dimension_2').add_legend() 
    plt.title(title)
    plt.xlabel("Dimension_1")
    plt.ylabel("Dimension_2")
    plt.show()
"""
t-SNE model over Bag of Words
"""
title="t-SNE model over Bag of Words"
count_vect=CountVectorizer()
final_count=count_vect.fit_transform(combine['CleanedText'].values)
tSNE_with_dense_array(final_count,title)
"""
t-SNE model over Bag of Words with ngrams (Min_grams=1 and Max_grams=2)
"""
title="t-SNE model over Bag of Words with ngrams (Min_grams=1 and Max_grams=2)"
count_vect=CountVectorizer(ngram_range=(1,2))
final_count=count_vect.fit_transform(combine['CleanedText'].values)
tSNE_with_dense_array(final_count,title)
"""
t-SNE model over TF-IDF
"""
title="t-SNE model over TF-IDF"
tfidf_vect=TfidfVectorizer()
final_tf_idf=tfidf_vect.fit_transform(combine['CleanedText'].values)
tSNE_with_dense_array(final_tf_idf,title)
"""
t-SNE model over TF-IDF with ngrams (Min_grams=1 and Max_grams=2)
"""
title="t-SNE model over TF-IDF with ngrams (Min_grams=1 and Max_grams=2)"
tfidf_vect=TfidfVectorizer(ngram_range=(1,2))
final_tf_idf=tfidf_vect.fit_transform(combine['CleanedText'].values)
tSNE_with_dense_array(final_tf_idf,title)
"""
t-SNE model over Average Word2Vec
"""
title="t-SNE model over Average Word2Vec"
sentences=[]
for sentence in corpus:
    words=[]
    for word in sentence.split():
        words.append(word)
    sentences.append(words)
word_to_vec=Word2Vec(sentences,size=50,min_count=1,workers=4)

avg_word2vec_mat=[]
for sen in sentences:
    sentence_vec=np.zeros(50)
    count=0
    for word in sen:
        vec=word_to_vec.wv[word]
        sentence_vec+=vec
        count+=1
    sentence_vec/=count
    avg_word2vec_mat.append(sentence_vec)
print(len(avg_word2vec_mat))
avg_word2vec_df=pd.DataFrame(avg_word2vec_mat)
tSNE_without_dense_array(avg_word2vec_df,title)
"""
t-SNE model over TF-IDF Word2Vec
"""
title="t-SNE model over TF-IDF Word2Vec"
tf_idf_word_to_vec=[]
count=0
tf_idf_features=tfidf_vect.get_feature_names()
for sentence in sentences:
    sentence_vec=np.zeros(50)
    tf_idf_weight=0
    for word in sentence:
        try:
            vec=word_to_vec.wv[word]
            tf_idf_value=final_tf_idf[count,tf_idf_features.index(word)]
            sentence_vec+=(tf_idf_value*vec)
            tf_idf_weight+=tf_idf_value
        except:
            pass
    sentence_vec/=tf_idf_weight
    tf_idf_word_to_vec.append(sentence_vec)
    count+=1
    
print(len(tf_idf_word_to_vec))
tf_idf_df=pd.DataFrame(tf_idf_word_to_vec)
tSNE_without_dense_array(tf_idf_word_to_vec,title)