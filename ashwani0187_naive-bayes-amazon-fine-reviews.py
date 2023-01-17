import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


%matplotlib inline

import sqlite3
import string
import nltk

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import re
import string
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import os
from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

# ============================== data preprocessing ===========================================
# Making the connection to the database.sqlite
con = sqlite3.connect("../input/database.sqlite")




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
amazon_featured_reviews.head(2)

# Data deduplication is used to clean the data having redundancy and many unwanted things which msut be removed to further
# use the data:

duplicate_df = pd.read_sql_query("""SELECT * FROM REVIEWS WHERE SCORE !=3 AND Text IN 
                                    (SELECT Text FROM REVIEWS
                                    GROUP BY Text having count(*) > 1)
                                    """ , con)
duplicate_df.head(4)

#So we can see there are many such duplicated rows having some column values similar to each other
#Doing some other check using the below query to see whether such reduncdancy is over small scale or a large scale of rows:
# From count(*) values we can see that we have so much of redundant data, so it has to be cleaned.
dup_data = pd.read_sql_query("""
select ID,ProductID,USERID , PROFILENAME , Summary ,text ,count(*) AS COUNT
FROM REVIEWS
GROUP BY PRODUCTID,SUMMARY,TEXT  
having count(*) > 1""",con)
dup_data.head(6)
# Let's see another case:

dup_data = pd.read_sql_query("""SELECT * FROM REVIEWS
                                    WHERE SCORE != 3 AND UserId = "AJD41FBJD9010" AND ProductID="7310172001"
                                    Order by ProductID""" , con)
dup_data

#Removing the Duplicate data points:

duplicated_data = amazon_featured_reviews.duplicated(subset={'UserId','ProfileName','Time','Summary','Text'} , keep='first')
duplicated_data = pd.DataFrame(duplicated_data , columns=['Boolean'])
print(duplicated_data.head(5))

#True values in the Boolean Series represents the duplicate data:
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

#Finding the books data in the amazon_featured_reviews using the regex:
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
#On observation of some of the reviews we got certain keywords related to books,reading ,poems , story,learn , study , music 
#So we removed these words as much as possible:


print('final shape before removing books reviews:' , final.shape)
analyzing_summary_book(final , re.compile(r'reading|books|book|read|study|learn|poems|music|story'))

print('final shape after removing the book reviews:' , final.shape)
#Computing the proportion of positive and negative class labels in the DataFrame:
final['Score'].value_counts()
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
# Functions to clean the html tags and punctuation marks using Regular Expression.

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

#Making backup of th pre processed data for the future use:
final_backup = final

final_backup.shape #to use the dataframe in future if required
#Now lets take roughly same proportion of  each of positive and negative review from the data set for faster processing 
#the further data:
#We can process our next tasks with whole amount of the data but we are bounded with time and memory spaces so we have used


# To sample 250K points :

final_clean = final.iloc[:250000,:]
print(final_clean.shape)
print(final_clean['Score'].value_counts())

#Sort the final data frame by timestamp values:
final_clean['Time'] = pd.to_datetime(final['Time'],unit='s')
final_clean = final_clean.sort_values(by='Time')
final_clean.shape
#Now we will compute Naive Bayes
#According to the assignment requirement we have to use cross validation techinque to find the optimal value of alpha.


def Naive_Bayes_GridSearchCV(X_train , y_train):
    
    #creating the odd list of nearest neighbors values:
    alpha_vals = list(np.arange(10e-6 , 10e-2 , 0.005))
    
    cv_scores=[]
    
    # create a parameter grid: map the parameter names to the values that should be searched
    param_grid = dict(alpha=alpha_vals)
    print(param_grid)
    
    # instantiate the grid
    naive = MultinomialNB()
    
    grid = GridSearchCV(naive, param_grid, cv=8, scoring='accuracy', return_train_score=False)
    
    # fit the grid with data
    grid.fit(X_train, y_train)
    
    # array of mean scores 
    cv_scores = [result.mean_validation_score for result in grid.grid_scores_]
    #print(cv_scores)
    
    
    # The below code will give the tuned parameter and the best mean accuracy among K-fold CV:
    optimal_alpha = grid.best_params_['alpha']
    
    Naive_Bayes_plot_CV_error(cv_scores , alpha_vals)
    
    print('*'*80)
    # examine the best model
    print('Best Score among all the alpha values is:' , grid.best_score_)
    print('Grid_best_params:' , grid.best_params_)
    print('Grid_best_estimator:' , grid.best_estimator_)
    
    return optimal_alpha
    
               
    
        
    
        
# Below function plot the Cross validation error versus the K values taken by us and we graphically compute our best K but
# we can have the idea of the rough value of K from the plot.

def Naive_Bayes_plot_CV_error(cv_scores , neighbors):
    
    # Computing the CV_error and plotting them against the K values in neighbors list:
    
    MSE = [1-x for x in cv_scores]
        
    # Plotting the misclassification error and the corresponding K-vaule:
    # We will consider that K value which has low error
    plt.figure(figsize=(8,8))
    plt.plot(neighbors , MSE , linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
    
    for xy in zip(neighbors , np.round(MSE , 3)):
        plt.annotate('(%s,%s)' % xy , xy=xy ,textcoords='data')
    plt.title('CV_error Vs Alpha value')
    plt.xlabel('Alpha values')
    plt.ylabel('CV_error error')
    plt.show()
# ============================== KNN with k = optimal_k_value ===============================================
# instantiate learning model k = optimal_k_value
#Test the optimal value of K to predict the new query point from the X_test:


def Naive_Bayes_optimal_confusion_matrix_plot(X_train , y_train , X_test , optimal_alpha , Vectorizer_type):

  naive_optimal = MultinomialNB(alpha=optimal_alpha)

  # fitting the model
  naive_optimal.fit(X_train, y_train)

  # predict the response
  
  pred_test = naive_optimal.predict(X_test)

  # evaluate accuracy
  
  acc_test = accuracy_score(y_test, pred_test)

  
  print('\nThe accuracy score of the Naive Classifier on Test Data for alpha = %f is %f' % (optimal_alpha, acc_test))
  
  print('*'*100)

  #Now Lets represent the predicted and actual values of class labels by our model:
  #Since our data is imbalanced data set we will use the confusion_matrix to check the accuracy of our model:

  cm = confusion_matrix(y_test , pred_test)
  print("Confusion Matrix for the " , Vectorizer_type , ':\n' , cm)

  print('*'*100)
  # plotting the confusion matrix to describe the performance of our classifier.
  print("Plotting the Confusion Matrix:")
  class_label = ["negative", "positive"]
  df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
  sns.heatmap(df_cm, annot = True, fmt = "d")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()
    
  print('*'*100)
  
  # To show the main classification report:
  from sklearn.metrics import classification_report
  print ('\n The Clasification report for','MultinomialNB',' algorithm with alpha =',optimal_alpha,' is \n ' , classification_report(y_test,pred_test))
# This method will take log probabilites of the each word and frame them in the dataframe for each class in the dataset.

def top_10_features(optimal_alpha , count_vect , X_train , Vectorizer):
    NB_optimal = MultinomialNB()
    NB_optimal.fit(X_train , y_train)
    features = count_vect.get_feature_names()
    df = pd.DataFrame(NB_optimal.feature_log_prob_,columns=features)
    df_new = df.T
    print('Feature Importance for the ' , Vectorizer)
    
    # Computing the negative and positive features log_porbabilities and taking only first 10 high probabilites:
    neg_features = df_new[0].sort_values(ascending = False)[0:10] 
    pos_features = df_new[1].sort_values(ascending = False)[0:10] 
    return pos_features , neg_features
# Creating the training data :
X = final_clean['CleanedText']
y = final_clean['Score']
print(X.shape , y.shape)

#Now we will split our train and test data and convert them to the Bag of words vector:
X_train, X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=40)
print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)

# Computing the Train Vectorizer :
count_vect = CountVectorizer(ngram_range=(1,1) , min_df=5)
X_train_bow = count_vect.fit_transform(X_train)
print(X_train_bow.shape)
type(X_train_bow)
#Computing the Test Vectorizer:
X_test_bow = count_vect.transform(X_test)
print(X_test_bow.shape)
type(X_test_bow)
# Below is the vocabulary dataframe of each word in Bag of words vector with its log probabilities.
NB_optimal = MultinomialNB()
NB_optimal.fit(X_train_bow , y_train)
features = count_vect.get_feature_names()
df = pd.DataFrame(NB_optimal.feature_log_prob_ , columns=features)
df_new = df.T
df_new.head()


#Optimal value for MultinomialNB:
optimal_alpha = Naive_Bayes_GridSearchCV(X_train_bow, y_train)
print('Optimal Value for alpha is:' , optimal_alpha)

print('*'*100)

#Plotting the Confusion matrix for the predicted and actual values to check the accuracy of our model:
#The Naive_Bayes_optimal_confusion_matrix_plot function takes 5 arguments as follows:
#<X_train , y_train , X_test , optimal_value_of_alpha , type_of_vector>
#This method finds the average accuracy score of Naive Bayes model on our test data set.
#This method also computes the final Confusion Matrix which determines the performance of our model.
#We have calculated the classification report for the classifier Naive Bayes.

Naive_Bayes_optimal_confusion_matrix_plot(X_train_bow , y_train ,X_test_bow , optimal_alpha ,'Bag_of_Words')

#Calculating the top_10_features for the positive reviews and negative reviews:
pos_features , neg_features = top_10_features(optimal_alpha , count_vect ,X_train_bow ,  'Bag-of-Words')

print("Top 10 positive words with high probability")
pos_features
print('*'*80)
print("Top 10 negative words with high probability")
neg_features
#Now we will split our train and test data and convert them to the TF - idf  vector:
X_train, X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=55)
print(X_train.shape , y_train.shape , X_test.shape , y_test.shape)
#Uni Gram  Train dataset Tf-IDF Vector:
tfidf_vector = TfidfVectorizer(ngram_range=(1,2) , min_df=5 )
X_train_tfidf= tfidf_vector.fit_transform(X_train)

print('X_train_tfidf.get_shape() : ' , X_train_tfidf.get_shape())


# Uni Gram Test dataset Tf-IDF Vectorizer:
X_test_tfidf= tfidf_vector.transform(X_test)

print('X_test_tfidf.get_shape():' , X_test_tfidf.shape)



# Below is the vocabulary dataframe of each word in Bag of words vector with its log probabilities.
NB_optimal = MultinomialNB()
NB_optimal.fit(X_train_tfidf , y_train)
features = tfidf_vector.get_feature_names()
df = pd.DataFrame(NB_optimal.feature_log_prob_ , columns=features)
df_new = df.T
df_new.head()

#Optimal value for Brute Force K-NN:
optimal_alpha = Naive_Bayes_GridSearchCV(X_train_tfidf, y_train)
print('Optimal Value for alpha is:' , optimal_alpha)

print('*'*100)

#Plotting the Confusion matrix for the predicted and actual values to check the accuracy of our model:
#The Naive_Bayes_optimal_confusion_matrix_plot function takes 5 arguments as follows:
#<X_train , y_train , X_test , optimal_value_of_alpha , type_of_vector>
#This method finds the average accuracy score of Naive Bayes model on our test data set.
#This method also computes the final Confusion Matrix which determines the performance of our model.
#We have calculated the classification report for the classifier Naive Bayes.
Naive_Bayes_optimal_confusion_matrix_plot(X_train_tfidf , y_train ,X_test_tfidf , optimal_alpha ,'Tf_idf')

#Calculating the top_10_features for the positive reviews and negative reviews:
pos_features , neg_features = top_10_features(optimal_alpha , tfidf_vector ,X_train_tfidf,  'Tf-Idf')
print("Top 10 positive words with high probability")
pos_features
print('*'*80)
print("Top 10 negative words with high probability")
neg_features
