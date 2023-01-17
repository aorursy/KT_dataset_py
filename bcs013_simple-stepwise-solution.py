#importing required libraries.....

import os 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression
import matplotlib.pyplot as plt
import seaborn
import nltk
from nltk import PorterStemmer # natural language toolkit
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
os.listdir("../input")
#read the data 
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample=pd.read_csv('../input/sample_submission.csv')
#printing some upper rows of our training data
train.head(10)
#some information about data
print('No. of training examples : ',len(train))
print("No. of test data : ",len(test))
print(train.columns[2:]) #columns_name
row=train.iloc[:,2:].sum(axis=1)
print("No. of examples with no labels : ",(row==0).sum())
#fill blank values with unknown otherwise model gives problem
print("Check for missing values in Train dataset")
null_check=train.isnull().sum()
print(null_check)
print("Check for missing values in Test dataset")
null_check=test.isnull().sum()
print(null_check)
print("filling NA with \"unknown\"")
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)
#plot
x=train.iloc[:,2:].sum()
plt.figure(figsize=(8,4))
ax= seaborn.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
# creating train-validation split
x_train, x_val, y_train, y_val = train_test_split(train.comment_text, train.iloc[:,2:8], test_size=0.3, random_state=19)
x_test = test.comment_text
def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    
    words = [w for w in words if not w in stopwords.words('english')]
    words=[lem.lemmatize(word, "v") for word in words]
    
    clean_sent=" ".join(words)
    return(clean_sent)
# preparing training text to pass in count vectorizer
corpus=[]
for text in x_train:
    text = clean(text)
    corpus.append(text)
# build Count Vectorizer, to convert a collection of text documents to a matrix of token counts
count_vect = CountVectorizer(ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(corpus)

# build TFIDF Transformer, to transform a count matrix to a normalized tf or tf-idf representation
# tfidf - term frequency inverse document frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# preparing validation text to pass in count vectorizer
X_val_set = []
for text in x_val:
    text = clean(text)
    X_val_set.append(text)
# tranforming validation data using count vectorizer followed by tfidf transformer
X_val_counts = count_vect.transform(X_val_set)
X_val_tfidf = tfidf_transformer.transform(X_val_counts)
# preparing test text to pass in count vectorizer
X_test_set = []
for text in x_test:
    text=clean(text)
    X_test_set.append(text)
# tranforming validation data using count vectorizer followed by tfidf transformer
X_test_counts = count_vect.transform(X_test_set)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
# creating dictionary to store prediction results
result_test = dict()
result_val =  dict()
result_train =  dict()
#Applying Model
# Multinomial Naive Bayes Model
MNB_classifier = OneVsRestClassifier(MultinomialNB())
MNB_classifier.fit(X_train_tfidf, y_train)
y_pred_train=MNB_classifier.predict(X_train_tfidf)
result_train['Multinomial_NB'] = y_pred_train
print ("Accurary of Multinomial Naive Bayes Classifier on Training Data:",accuracy_score(y_pred_train,y_train))
# Bernoulli Naive Bayes Model
BNB_model = OneVsRestClassifier(BernoulliNB())
BNB_model.fit(X_train_tfidf, y_train)
y_pred_train=BNB_model.predict(X_train_tfidf)
result_train['Bernoulli_NB'] = y_pred_train
print('Accurary of Bernoulli Naive Bayes Classifier on Training Data:',accuracy_score(y_pred_train,y_train))
#Ridge Classifier Model
ridge_model = OneVsRestClassifier(RidgeClassifier(normalize=True))
ridge_model.fit(X_train_tfidf, y_train)
y_pred_train=ridge_model.predict(X_train_tfidf)
result_train['Ridge_Classifier'] = y_pred_train
print('Accurary of Ridge Classifier on Training Data:',accuracy_score(y_pred_train,y_train))                          
# Logistic Regression Model
log_model = OneVsRestClassifier(LogisticRegression(multi_class='ovr'))
log_model.fit(X_train_tfidf, y_train)
y_pred_train=log_model.predict(X_train_tfidf)
result_train['Logistic_Regression'] = y_pred_train
print('Accurary of Logistic Regression on Training Data:',accuracy_score(y_pred_train,y_train))
# SVM Classifier Model
svm_model = OneVsRestClassifier(LinearSVC(multi_class='ovr'))
svm_model.fit(X_train_tfidf, y_train)
y_pred_train=svm_model.predict(X_train_tfidf)
result_train['SVM'] = y_pred_train
print('Accurary of SVM Classifier on Training Data:',accuracy_score(y_pred_train,y_train))
#visualizations

D=result_train[:]
plt.figure(figsize=(20, 7))
plt.yticks( fontsize=20)
plt.xticks(range(len(D)), list(D.keys()), fontsize=20)
ax=plt.bar(range(len(D)), list(D.values()), align='center',width=0.8)
plt.title("# Accuracy Score on Training Set by different Models\n", fontsize=40)
plt.ylabel('# Accuracy Range', fontsize=30)
plt.xlabel('\n#Model type ', fontsize=30)
#adding the text labels
for rect in ax:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % float(height), ha='center', va='bottom',fontsize=20)
plt.show()
#Picking the model with highest accuracy rate on train data and now do hyperparameter tuning on cross-validation set
#Hyperparameter Tuning
grid_values = {'estimator__C': [0.3, 1.0, 30.0]}
svm_grid = GridSearchCV(svm_model, param_grid = grid_values, scoring = 'roc_auc')
svm_grid.fit(X_train_tfidf, y_train)
print('Accurary of SVM Classifier on Training Data: {:.3f}' .format(svm_grid.score(X_train_tfidf, y_train)))
print('Accurary of SVM Classifier on Validation Data: {:.3f}' .format(svm_grid.score(X_val_tfidf, y_val)))
print('Grid best parameter (max. accuracy): ', svm_grid.best_params_)
print('Grid best score (accuracy): ', svm_grid.best_score_)

#predict for the test data
result_test['SVM']=svm_grid.predict(X_test_tfidf)
# storing results of SVM Classifier as our result
y_test = result_test['SVM']
# combining final results with the original test data set
output = pd.DataFrame(y_test, columns = train.columns[2:8], index = test.index)
output = pd.concat([test, output], axis=1)
#Sample Submission
sample.head()
# verifing data
output.head()
# verifing select random case, as per index from above code chunk
output.comment_text[5902]
output.iloc[5902,:]
# quick summary for training, validation and test set respectively
y_train.sum(axis=0)
y_val.sum(axis=0)
output.iloc[:,2:8].sum(axis=0)
#Final Submission
my_submission = output.drop(['comment_text'], axis = 1, inplace = False)
my_submission.to_csv('submission.csv', index=False)
