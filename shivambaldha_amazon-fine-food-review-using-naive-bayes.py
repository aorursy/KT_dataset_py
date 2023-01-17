#import all we needed module
import numpy as np
import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import model_selection
#from sklearn import cross_validation******** this is not working
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
import os
print(os.listdir("../input/")) 
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')

# we neglect the review having a score = 3

filtered_data = pd.read_sql_query('''select *from reviews where Score !=3''',con)

def partition(x):
    if x<3 :
        return 'negative'
    return 'positive'
actualScore = filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
display = pd.read_sql_query('''select * from reviews where Score != 3 and userId = "AR5J8UI46CURR" order by ProductID''',con)
display.head()
final = sorted_data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'})
final.shape
sorted_data = filtered_data.sort_values('ProductId',axis = 0,ascending=True,inplace = False,kind='quicksort',na_position='last')
#we remove duplication using HelpfulnessDenominator and HelpfulnessNumerator.

final = final[final.HelpfulnessNumerator<= final.HelpfulnessDenominator]
import re 

i = 0
for sent in final['Text'].values:
    if(len(re.findall('<.>*?',sent))):
        print(i)
        print(sent)
        break;
    i+=1
import string
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') # initialising snowball stemmer

def cleanhtml(sentence): #function to clean word of any html tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr , ' ',sentence)
    return cleantext
def cleanpunc(sentence): #function to clean word of any punctuation or special character
    cleaned = re.sub(r'[?|!|\,|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r'',cleaned)
    return cleaned
print(stop)
print('***********************************************')
print(sno.stem('tasty'))
i = 0
strl = ' '
final_string  = []
all_positive_words=[] # store words from +ve reviews here
all_negattive_words=[]#store words from -ve reviews here
s= ''

for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent) #remove html tag
    
    for w in sent.split():
        for cleaned_word in cleanpunc(w).split():
            if((cleaned_word.isalpha()) & (len(cleaned_word)>2)):
                if(cleaned_word.lower() not in stop):
                    s = (sno.stem(cleaned_word.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if(final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s) #list of all words use to store +ve list 
                    if (final['Score'].values)[i] == 'negative':
                        all_negattive_words.append(s) #list of all words use to store -Ve list
                else:
                    continue
            else:
                continue
                
    #print filtered sentens 
    strl = b" ".join(filtered_sentence) #final string of cleaned words
    final_string.append(strl)
    i+=1
final['CleanedText']=final_string
final.head(3) #below the processed review can be seen in the CleanedText Column 


# store final table into an SQlLite table for future.
conn = sqlite3.connect('finalassignment.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)
import sqlite3
con = sqlite3.connect('finalassignment.sqlite')
cleaned_data = pd.read_sql_query('select * from Reviews', con)
cleaned_data.shape
cleaned_data['Score'].value_counts()
# To randomly sample 5k points from both class

data_p = cleaned_data[cleaned_data['Score'] == 'positive'].sample(n = 5000)
data_n = cleaned_data[cleaned_data['Score'] == 'negative'].sample(n = 5000)
final_10k = pd.concat([data_p, data_n])
final_10k.shape
# Sorting data based on time
final_10k['Time'] = pd.to_datetime(final_10k['Time'], unit = 's')
final_10k = final_10k.sort_values(by = 'Time')
# function compute the alpha value 

def naive_bayes(X_train , y_train):
    
    alpha_value = np.arange(1,500,0.5)
    
    # empty list that will hold cv value
    cv_scores = []
    
    #perform 10-fold cross validation
    for alpha in alpha_value:
        mnb = MultinomialNB(alpha = alpha)
        scores = cross_val_score(mnb , X_train , y_train , cv = 10 , scoring = 'accuracy')
        cv_scores.append(scores.mean())
        
    # changing misclassification error
    MSE = [1 - x for x in cv_scores]
    
    #determining best alpha
    optimal_alpha = alpha_value[MSE.index(min(MSE))]
    print('\nThe optimal number of alpha is %d :'% optimal_alpha)
    
    #plot misclassification error vs alpha
    plt.plot(alpha_value ,MSE , marker = '*')
    
   
    #for xy in zip(alpha_values, np.round(MSE,3)):
        #plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
    plt.title("Misclassification Error vs alpha")
    plt.xlabel('value of alpha')
    plt.ylabel('Misclassification Error')
    plt.show()

    #print("the misclassification error for each value of alpha is : ", np.round(MSE,3))
    return optimal_alpha
# 10k data which will use to train model after vectorization
X = final_10k["CleanedText"]
print("shape of X:", X.shape)
# class label
y = final_10k["Score"]
print("shape of y:", y.shape)
# split data into train and test where 70% data used to train model and 30% for test

from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(X_train.shape, y_train.shape, x_test.shape , y_test.shape)
# Train Vectorizor
from sklearn.feature_extraction.text import CountVectorizer 

bow = CountVectorizer()
X_train = bow.fit_transform(X_train)
X_train
# convert test text data to its vectorizor
x_test = bow.transform(x_test)
x_test.shape
# To choose optimal_alpha using cross validation

from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection


optimal_alpha_bow = naive_bayes(X_train, y_train)
optimal_alpha_bow
# instantiate learning model alpha = optimal_alpha
nb_optimal =  MultinomialNB(alpha = optimal_alpha_bow)

# fitting the model
nb_optimal.fit(X_train, y_train)
#knn_optimal.fit(bow_data, y_train)

# predict the response
pred = nb_optimal.predict(x_test)
# to get all feature name

bow_features = bow.get_feature_names()
# To count feature for each class while fitting the model
# Number of samples encountered for each (class, feature) during fitting

feat_count = nb_optimal.feature_count_
feat_count.shape
# Number of samples encountered for each class during fitting

nb_optimal.class_count_
# Empirical log probability  of feature given a class 

log_prob = nb_optimal.feature_log_prob_
log_prob
feature_prob = pd.DataFrame(log_prob , columns = bow_features)
feature_prob_tr = feature_prob.T
feature_prob_tr.shape
# to show top 10 feature from both class
#feature Impportance

print('Top 10 Negative Feature :', feature_prob_tr[0].sort_values(ascending = False)[0:10])
print('------------------------------------------------------------------------------------')
print('Top 10 Postive Feature : ', feature_prob_tr[1].sort_values(ascending = False)[0:10])
# Accuracy on train data
train_acc_bow = nb_optimal.score(X_train, y_train)
print("Train accuracy : %f%%" % (train_acc_bow))
# Error on train data
train_err_bow = 1-train_acc_bow 
print("Train Error %f%%" % (train_err_bow))
# evaluate accuracy on test data
acc_bow = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the naive bayes classifier for alpha = %d is %f%%' % (optimal_alpha_bow, acc_bow))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
models = pd.DataFrame({'Model': ['Naive Bayes with Bow'], 'Hyper Parameter(K)': [optimal_alpha_bow], 'Train Error': [train_err_bow], 'Test Error': [100-acc_bow], 'Accuracy': [acc_bow ], 'Train Accuracy': [train_acc_bow ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# # 10k data which will use to train model after vectorization

X = final_10k['CleanedText']
X.shape
# target / class label

y = final_10k['Score']
y.shape
#split data

X_train , x_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42)
print(X_train.shape , x_test.shape , y_train.shape , y_test.shape)
# Train Vectorizor

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train = tf_idf_vect.fit_transform(X_train)
X_train
# convert test data to its vectorizor

x_test = tf_idf_vect.transform(x_test)
x_test.shape
# to chossing optimal alpha using cv

from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn import model_selection


optimal_alpha_tfidf = naive_bayes(X_train, y_train)
optimal_alpha_tfidf
# instantiate learning model alpha = optimal_alpha
nb_optimal = MultinomialNB(alpha = optimal_alpha_tfidf)

# fitting the model
nb_optimal.fit(X_train, y_train)
#knn_optimal.fit(bow_data, y_train)
    
# predict the response
pred = nb_optimal.predict(x_test)
# To get all the features name 

tfidf_features = tf_idf_vect.get_feature_names()
# To count feature for each class while fitting the model
# Number of samples encountered for each (class, feature) during fitting

feat_count = nb_optimal.feature_count_
feat_count.shape
# Number of samples encountered for each class during fitting

nb_optimal.class_count_
# Empirical log probability of features given a class(i.e. P(x_i|y))

log_prob = nb_optimal.feature_log_prob_
log_prob
feature_prob = pd.DataFrame(log_prob, columns = tfidf_features)
feature_prob_tr = feature_prob.T
feature_prob_tr.shape
# to show top 10 feature from both class
#feature Impportance

print('Top 10 Negative Feature :', feature_prob_tr[0].sort_values(ascending = False)[0:10])
print('------------------------------------------------------------------------------------')
print('Top 10 Postive Feature : ', feature_prob_tr[1].sort_values(ascending = False)[0:10])
# Accuracy on train data
train_acc_tfidf = nb_optimal.score(X_train, y_train)
print("Train accuracy : %f%%" % (train_acc_tfidf))
# Error on train data
train_err_tfidf = 1-train_acc_tfidf
print("Train Error %f%%" % (train_err_tfidf))
# evaluate accuracy
acc_tfidf = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the naive bayes classifier for alpha = %d is %f%%' % (optimal_alpha_tfidf, acc_tfidf))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
models = pd.DataFrame({'Model': ['Naive Bayes with Tf-Idf'], 'Hyper Parameter(K)': [optimal_alpha_tfidf], 'Train Error': [train_err_tfidf], 'Test Error': [100-acc_tfidf], 'Accuracy': [acc_tfidf ], 'Train Accuracy': [train_acc_tfidf ]}, columns = ["Model", "Hyper Parameter(K)", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
# model performence table
models = pd.DataFrame({'Model': ['Naive Bayes with Bow', "Naive Bayes with TFIDF"], 'Hyper Parameter(alpha)': [optimal_alpha_bow, optimal_alpha_tfidf], 'Train Error': [train_err_bow, train_err_tfidf], 'Test Error': [100-acc_bow, 100-acc_tfidf], 'Accuracy': [acc_bow, acc_tfidf]}, columns = ["Model", "Hyper Parameter(alpha)", "Train Error", "Test Error", "Accuracy"])
models.sort_values(by='Accuracy', ascending=False)