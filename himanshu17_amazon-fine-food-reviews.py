# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sqlite3
import seaborn as sns
import re
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#sqlobj = sqlite3.connect('./database.sqlite')
#reviews = pd.read_sql_query("""select * from Reviews""", sqlobj)
review=pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
review.shape
#Printing dataset information
review.info()
#Printing few rows
review.head(5)
# Removing duplicate entries
reviews=review.drop_duplicates(subset=["UserId","ProfileName","Time","Text"], keep='first', inplace=False)
print("The shape of the data set after removing duplicate reviews : {}".format(reviews.shape))
# Helpfulness Numerator - Number of users who found the review helpful
# Helpfulness Denominator - Number of users who indicated whether they found the review helpful or not
# Score - Rating between 1 and 5

reviews[["HelpfulnessNumerator", "HelpfulnessDenominator", "Score"]].describe()
reviews['Score'].value_counts()
reviews['Score'].value_counts().plot(kind='bar', 
                                     color=['r', 'b', 'g', 'y', 'm'], 
                                     title='Ratings distribution', 
                                     xlabel='Score', ylabel='Number of Users')
#Method to calculate Helpfulness

# (D) == 0        : No Indication
# (N/D) > 75 %    : Helpful
# (N/D) 75 - 25 % : Intermediate
# (N/D) < 25 %    : Not Helpful


def helpCalc(n, d):
    if d==0:
        return 'No Indication'
    elif n > (75.00/100*d):
        return 'Helpful'
    elif n < (25.00/100*d):
        return 'Not Helpful'
    else:
        return 'Intermediate'

reviews['Helpfulness'] = reviews.apply(lambda x : helpCalc(x['HelpfulnessNumerator'], x['HelpfulnessDenominator']), axis=1)
reviews.head(5)
reviews['Helpfulness'].value_counts()
reviews['Helpfulness'].value_counts().plot(kind='bar', 
                                           color=['r', 'b', 'g', 'y'], 
                                           title='Distribution of Helpfulness', 
                                           xlabel='Helpfulness', ylabel='Number of Users')
print('Percentage of No Indication reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['No Indication'])*100.0/len(reviews)))
print('Percentage of Helpful reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Helpful'])*100.0/len(reviews)))
print('Percentage of Intermediate reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Intermediate'])*100.0/len(reviews)))
print('Percentage of Not Helpful reviews %.2f %%' % ((reviews['Helpfulness'].value_counts()['Not Helpful'])*100.0/len(reviews)))
helpfulness_score = pd.crosstab(reviews['Score'], reviews['Helpfulness'])
helpfulness_score
helpfulness_score.plot(kind='bar', figsize=(10,6), title='Helpfulness Score')
reviews['DateTime'] = pd.to_datetime(reviews['Time'], unit='s')
monthly_review = reviews.groupby([reviews['DateTime'].dt.year, reviews['DateTime'].dt.month, reviews['Score']]).count()['ProductId'].unstack().fillna(0)
monthly_review.head(30)
monthly_review.plot(figsize=(25,8), xlabel='Year,Month', ylabel='Review Counts', title='Monthly Review Counts')
def WordLength(text):
    words = str(text).split(" ")
    return len(words)


reviews['TextLength'] = reviews['Text'].apply(lambda x : WordLength(x))

print('Maximum length of Text Words:', reviews['TextLength'].max())
print('Mean length of Text Words:', reviews['TextLength'].mean())
print('Minimum length of Text Words:', reviews['TextLength'].min())

plt.figure(figsize=(12,10))
ax = sns.boxplot(x='Score',y='TextLength', data=reviews)
plt.figure(figsize=(12,10))
ax = sns.violinplot(x='Helpfulness', y='TextLength', data=reviews)
reviews['SummaryLength'] = reviews['Summary'].apply(lambda x : WordLength(x))

print('Maximum length of Summary Words:', reviews['SummaryLength'].max())
print('Mean length of Summary Words:', reviews['SummaryLength'].mean())
print('Minimum length of Summary Words:', reviews['SummaryLength'].min())

plt.figure(figsize=(12,10))
ax = sns.boxplot(x='Score',y='SummaryLength', data=reviews)
plt.figure(figsize=(12,10))
ax = sns.violinplot(x='Helpfulness', y='SummaryLength', data=reviews)
rf = reviews.groupby(['UserId', 'ProfileName']).count()['ProductId']
y = rf.to_frame()
x = y.sort_values('ProductId', ascending=False)
x.head(20).plot(kind='bar', figsize=(20,5), title='Frequency of Top 20 Reviewers', xlabel='(UserId, Profile Name)', ylabel='Number of Reviews')
reviews['ScoreClass'] = reviews['Score'].apply(lambda x : 'Positive' if x > 3 else 'Negative')
reviews['ScoreClass'].value_counts()
reviews['ScoreClass'].value_counts().plot(kind='bar', color=['g','r'], title='Score Class Distribution', xlabel='Score Class', ylabel='Score Count')
print('Reviews with Positive Score Class is %.2f %%' % ((reviews['ScoreClass'].value_counts()['Positive'])*100.0/len(reviews)))
print('Reviews with Negative Score Class is %.2f %%' % ((reviews['ScoreClass'].value_counts()['Negative'])*100.0/len(reviews)))
reviews[['Text', 'ScoreClass']].head(5)
def splitPosNeg(reviews):
    neg = reviews.loc[reviews['ScoreClass']=='Negative']
    pos = reviews.loc[reviews['ScoreClass']=='Positive']
    return [pos,neg]

[pos,neg] = splitPosNeg(reviews)

print("Number of Total Reviews : ", len(pos)+len(neg))
print("Number of Positive Reviews : ", len(pos))
print("Number of Negative Reviews : ", len(neg))
# Printing a positive review and polarity
print('Positive Polarity Review :', pos['Text'].values[0])
print('Polarity Sentiment :', pos['ScoreClass'].values[0])
# Printing a negative review and polarity
print('Negative Polarity Review :', neg['Text'].values[0])
print('Negative Sentiment :', neg['ScoreClass'].values[0])
# To-do : Lemmatization

# Expand the reviews x is an input string of any length. Convert all the words to lower case
def decontracted(x):
    x = str(x).lower()
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("won't", " will not").replace("cannot", " can not").replace("can't", " can not")\
                           .replace("n't", " not").replace("what's", " what is").replace("it's", " it is")\
                           .replace("'ve", " have").replace("'m", " am").replace("'re", " are")\
                           .replace("he's", " he is").replace("she's", " she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("how's"," how has").replace("y'all"," you all")\
                           .replace("o'clock"," of the clock").replace("ne'er"," never").replace("let's"," let us")\
                           .replace("finna"," fixing to").replace("gonna"," going to").replace("gimme"," give me").replace("gotta"," got to").replace("'d"," would")\
                           .replace("daresn't"," dare not").replace("dasn't"," dare not").replace("e'er"," ever").replace("everyone's"," everyone is")\
                           .replace("'cause'"," because")
    
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    return x

# Removing the html tags
def removeHTML(text):
    pattern = re.compile('<.*?>')
    cleanText = re.sub(pattern,' ',text)
    return cleanText

# Remove any punctuations or limited set of special characters like , or . or # etc.
def removePunctuations(text):
    cleanText  = re.sub('[^a-zA-Z]',' ',text)
    return (cleanText)

# Remove words with numbers
def removeNumbers(text):
    cleanText = re.sub("\S*\d\S*", " ", text).strip()
    return (cleanText)

#Remove URL from sentences.
def removeURL(text):
    textModified = re.sub(r"http\S+", " ", text)
    cleanText = re.sub(r"www.\S+", " ", textModified)
    return (cleanText)

#Remove words like 'zzzzzzzzzzzzzzzzzzzzzzz', 'testtting', 'grrrrrrreeeettttt' etc. Preserves words like 'looks', 'goods', 'soon' etc. 
#We will remove all such words which has three consecutive repeating characters.
def removePatterns(text): 
    cleanText  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',text)
    return (cleanText)

# Remove Stopwords
defaultStopwordList = set(stopwords.words('english'))
remove_not = set(['not','no','nor'])
stopwordList = defaultStopwordList - remove_not

# Snowball Stemming
stemmer = SnowballStemmer(language='english')
# Data preprocessing considering all words across whole reviews
def preprocessing(text):
    total_words = []
    text = decontracted(text)
    text = removeHTML(text)
    text = removePunctuations(text)
    text = removeNumbers(text)
    text = removeURL(text)
    text = removePatterns(text)
    
    line = nltk.word_tokenize(text)
    for word in line:
        if (word not in stopwordList):
            stemmed_word = stemmer.stem(word)
            total_words.append(stemmed_word)
    return ' '.join(total_words)

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Preprocessing positive review and negative review separately.
pos_data = [] # A list of preprocessed positive reviews
neg_data = [] # A list of preprocessed negative reviews

for p in tqdm(pos['Text']):
    pos_data.append(preprocessing(p))
    
for n in tqdm(neg['Text']):
    neg_data.append(preprocessing(p))
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Combining preprocessed positive review and negative review
data = pos_data + neg_data # A list of combined preprocessed positive and negative reviews
labels = np.concatenate((pos['ScoreClass'].values,neg['ScoreClass'].values)) # An array of combined positive score class and negative score class

#------------------------------------------------------------------------------------------------------------------------------------------------
# Tokenizing the data and creating a token list
token_list = []
for line in tqdm(data):
    l = nltk.word_tokenize(line)
    for w in l:
        token_list.append(w)
        
#------------------------------------------------------------------------------------------------------------------------------------------------
# Get list of unique words from whole reviews
total_words = list(set(token_list))
print("Total unique words in whole reviews : ", len(total_words))

#------------------------------------------------------------------------------------------------------------------------------------------------
# Save Total unique words in whole reviews
with open('unique_words_in_whole_reviews.pkl', 'wb') as file:
    pickle.dump(total_words, file)
# Load the unique word from whole reviews
with open('unique_words_in_whole_reviews.pkl', 'rb') as file:
    total_words = pickle.load(file)
    
#----------------------------------------------------------------------------------------------------------------------------------    
# Finding the distribution of length of all unique words across whole reviews
word_length_dist = []

for word in tqdm(total_words):
    length = len(word)
    word_length_dist.append(length)

plt.figure(figsize=(20,10))
plt.hist(word_length_dist, color='green', bins =90)
plt.title('Distribution of the length of all unique words across whole reviews')
plt.xlabel('Word Lengths')
plt.ylabel('Number of Words')
# Data preprocessing considering only words whose length is greater tha 2 and less than 16 across whole reviews
def _preprocessing(text):
    total_words_reduced = []
    text = decontracted(text)
    text = removeHTML(text)
    text = removePunctuations(text)
    text = removeNumbers(text)
    text = removeURL(text)
    text = removePatterns(text)
    
    line = nltk.word_tokenize(text)
    for word in line:
        if (word not in stopwordList) and (2 < len(word) < 16):
            stemmed_word = stemmer.stem(word)
            total_words_reduced.append(stemmed_word)
    return ' '.join(total_words_reduced)

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Preprocessing positive review and negative review separately.
pos_data_reduced = [] # A list of preprocessed positive reviews
neg_data_reduced = [] # A list of preprocessed negative reviews

for p in tqdm(pos['Text']):
    pos_data_reduced.append(_preprocessing(p))
    
for n in tqdm(neg['Text']):
    neg_data_reduced.append(_preprocessing(p))
    
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Combining preprocessed positive review and negative review
data_final = pos_data_reduced + neg_data_reduced # A list of combined preprocessed positive and negative reviews
# An array of combined positive score class and negative score class
labels_final = np.concatenate((pos['ScoreClass'].values,neg['ScoreClass'].values)) 

#------------------------------------------------------------------------------------------------------------------------------------------------
# Tokenizing the data and creating a token list
token_list_reduced = []
for line in tqdm(data_final):
    l = nltk.word_tokenize(line)
    for w in l:
        token_list_reduced.append(w)
        
#------------------------------------------------------------------------------------------------------------------------------------------------
# Get list of unique words from train token list
total_words_reduced = list(set(token_list_reduced))
print("Total unique words in whole reviews of length > 2 and < 16 : ", len(total_words_reduced))

#------------------------------------------------------------------------------------------------------------------------------------------------
# Save Total unique words reduced in whole reviews
with open('unique_words_reduced_in_whole_reviews.pkl', 'wb') as file:
    pickle.dump(total_words_reduced, file)

# Save final data of whole reviews
with open('data_final.pkl', 'wb') as file:
    pickle.dump(data_final, file)
    
# Save final labels of whole reviews
with open('labels_final.pkl', 'wb') as file:
    pickle.dump(labels_final, file)
# Load the final data and final labels
with open('data_final.pkl', 'rb') as file:
    data_final = pickle.load(file)
    
with open('labels_final.pkl', 'rb') as file:
    labels_final = pickle.load(file)
    
#------------------------------------------------------------------------------------------------------------------------------------------------    
# Splitting the datasets into train-test in 80:20 ratio
[train_data,test_data,train_label,test_label] = train_test_split(data_final, labels_final, test_size=0.20, random_state=20160121, stratify=labels)
# Get list of unique tokens in train data
train_token = []

for line in tqdm(train_data):
    l = nltk.word_tokenize(line)
    for w in l:
        train_token.append(w)
        
x = len(list(set(train_token)))        
print("Unique token in train data : ", x)
cv_object = CountVectorizer(min_df=10, max_features=50000, dtype='float')
cv_object.fit(train_data)

print("Some BOW Unigram features are : ", cv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating BOW Unigram vectors...")
train_data_cv = cv_object.transform(train_data)
test_data_cv = cv_object.transform(test_data)

print("\nThe type of BOW Unigram Vectorizer ", type(train_data_cv))
print("Shape of train BOW Unigram Vectorizer ", train_data_cv.get_shape())
print("Shape of test BOW Unigram Vectorizer ", test_data_cv.get_shape())


with open('train_data_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(train_data_cv, file)

with open('test_data_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(test_data_cv, file)
    
with open('train_label_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_BOW_Uni.pkl', 'wb') as file:
    pickle.dump(test_label, file)
cv_object = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=50000, dtype='float')
cv_object.fit(train_data)

print("Some BOW Bigram features are : ", cv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating BOW Bigram vectors...")
train_data_cv = cv_object.transform(train_data)
test_data_cv = cv_object.transform(test_data)

print("\nThe type of BOW Bigram Vectorizer ", type(train_data_cv))
print("Shape of train BOW Bigram Vectorizer ", train_data_cv.get_shape())
print("Shape of test BOW Bigram Vectorizer ", test_data_cv.get_shape())

with open('train_data_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(train_data_cv, file)

with open('test_data_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(test_data_cv, file)
    
with open('train_label_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_BOW_Bi.pkl', 'wb') as file:
    pickle.dump(test_label, file)
tv_object = TfidfVectorizer(min_df=10, max_features=50000, dtype='float')
tv_object.fit(train_data)

print("Some Tf-idf Unigram features are : ", tv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating Tf-idf Unigram vectors...")
train_data_tv = tv_object.transform(train_data)
test_data_tv = tv_object.transform(test_data)

print("\nThe type of Tf-idf Unigram Vectorizer ", type(train_data_tv))
print("Shape of train Tf-idf Unigram Vectorizer ", train_data_tv.get_shape())
print("Shape of test Tf-idf Unigram Vectorizer ", test_data_tv.get_shape())

with open('train_data_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(train_data_tv, file)

with open('test_data_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(test_data_tv, file)
    
with open('train_label_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_TF-IDF_Uni.pkl', 'wb') as file:
    pickle.dump(test_label, file)
tv_object = TfidfVectorizer(ngram_range=(1,2), min_df=10, max_features=50000, dtype='float')
tv_object.fit(train_data)

print("Some Tf-idf Bigram features are : ", tv_object.get_feature_names()[100:110])
print("="*145)

print("\nCreating Tf-idf Bigram vectors...")
train_data_tv = tv_object.transform(train_data)
test_data_tv = tv_object.transform(test_data)

print("\nThe type of Tf-idf Bigram Vectorizer ", type(train_data_tv))
print("Shape of train Tf-idf Bigram Vectorizer ", train_data_tv.get_shape())
print("Shape of test Tf-idf Bigram Vectorizer ", test_data_tv.get_shape())

with open('train_data_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(train_data_tv, file)

with open('test_data_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(test_data_tv, file)
    
with open('train_label_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(train_label, file)

with open('test_label_TF-IDF_Bi.pkl', 'wb') as file:
    pickle.dump(test_label, file)
# Utility functions
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
# def top_features()

#This function is used to standardize a data matrix.
def standardize(data, with_mean):
    scalar = StandardScaler(with_mean=with_mean)
    std=scalar.fit_transform(data)
    return (std)

#------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting ROC Curve
def plot_roc_curve(clf, train_data, train_label, test_data, test_label):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    train_prob = clf.predict_proba(train_data)
    train_label_prob = train_prob[:,1]
    fpr["Train"], tpr["Train"], threshold = roc_curve(train_label, train_label_prob, pos_label='Positive')
    roc_auc["Train"] = auc(fpr["Train"], tpr["Train"])
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
    fpr["Test"], tpr["Test"], threshold = roc_curve(test_label, test_label_prob, pos_label='Positive')
    roc_auc["Test"] = auc(fpr["Test"], tpr["Test"])
    
    plt.figure(figsize=(5,5))
    linewidth = 2
    plt.plot(fpr["Test"], tpr["Test"], color='green', lw=linewidth, label='ROC curve Test Data (area = %0.2f)' % roc_auc["Test"])
    plt.plot(fpr["Train"], tpr["Train"], color='red', lw=linewidth, label='ROC curve Train Data (area = %0.2f)' % roc_auc["Train"])
    plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--', label='Baseline ROC curve (area = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
#------------------------------------------------------------------------------------------------------------------------------------------------
# Plotting Precision-Recall Curve
def plot_pr_curve(clf, train_data, train_label, test_data, test_label):
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    train_prob = clf.predict_proba(train_data)
    train_label_prob = train_prob[:,1]
    precision["Train"], recall["Train"], threshold = precision_recall_curve(train_label, train_label_prob, pos_label='Positive')
    pr_auc["Train"] = auc(recall["Train"], precision["Train"])
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
    precision["Test"], recall["Test"], threshold = precision_recall_curve(test_label, test_label_prob, pos_label='Positive')
    pr_auc["Test"] = auc(recall["Test"], precision["Test"])
    
    plt.figure(figsize=(5,5))
    linewidth = 2
    plt.plot(recall["Test"], precision["Test"], color='green', lw=linewidth, label='Precision-Recall Curve Test Data (area = %0.2f)' % pr_auc["Test"])
    plt.plot(recall["Train"], precision["Train"], color='red', lw=linewidth, label='Precision-Recall curve Train Data (area = %0.2f)' % pr_auc["Train"])
    #plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--', label='Baseline Precision-Recall curve (area = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Plot')
    plt.legend(loc="lower right")
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate Metrics Performance 
def metrics_performance(grid, train_data, train_label, test_data, test_label):
    clf = grid.best_estimator_
    clf.fit(train_data, train_label)
    test_label_pred = clf.predict(test_data)
    
    test_prob = clf.predict_proba(test_data)
    test_label_prob = test_prob[:,1]
   
    print("Accuracy : ", accuracy_score(test_label, test_label_pred, normalize=True) * 100)
    print("Points : ", accuracy_score(test_label, test_label_pred, normalize=False))
    print("Precision : ", np.round(precision_score(test_label ,test_label_pred, pos_label='Positive'),4))
    print("Recall : ", recall_score(test_label, test_label_pred, pos_label='Positive'))
    print("F1-score : ", f1_score(test_label,test_label_pred, pos_label='Positive'))
    print("AUC : ", np.round(roc_auc_score(test_label, test_label_prob),4))
    print ('\nClasification Report :')
    print(classification_report(test_label,test_label_pred))
    
    print('\nConfusion Matrix :')
    cm = confusion_matrix(test_label ,test_label_pred)
       
    df_cm = pd.DataFrame(cm, index = [' (0)',' (1)'], columns = [' (0)',' (1)'])
    plt.figure(figsize = (5,5))
    ax = sns.heatmap(df_cm, annot=True, fmt='d')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title('Confusion Matrix')
    
    plot_roc_curve(clf, train_data, train_label, test_data, test_label)
    plot_pr_curve(clf, train_data, train_label, test_data, test_label)
#------------------------------------------------------------------------------------------------------------------------------------------------    
# Plot GridSearch Results   
def plot_gridsearch_result(grid):
    cv_result = grid.cv_results_
    auc_train = list(cv_result['mean_train_score'])
    auc_test = list(cv_result['mean_test_score'])
    params = cv_result['params']
    
    hp_values = [p['C'] for p in params] #Get list of C values
    
    # Plot Heapmap of GridSearchCV Result
    cv_result = {'C':hp_values, 'Mean Train Score':auc_train, 'Mean Test Score':auc_test} # dataframe of cv_result
    cv = pd.DataFrame(cv_result)
    sns.heatmap(cv, annot=True)
    
    # Plot error vs C values
    plt.figure(figsize=(15,6))
    plt.plot(hp_values , auc_train, color='red', label='Train AUC')
    plt.plot(hp_values , auc_test, color='blue', label='Validation AUC')
    plt.title('Area under the ROC Curve vs C Values ')
    plt.xlabel('Hyperparameter: Values of C')
    plt.ylabel('Area under the ROC Curve (AUC Scores)')
    plt.legend()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------    
# Function to apply GridSearchCV
def gridSearchCV(train_data, train_label, regularization):
    param = {'C' : [0.001, 0.01, 0.1, 1, 10]}
    model = LogisticRegression(penalty=regularization, solver='liblinear', random_state=0)
    cv = TimeSeriesSplit(n_splits=10).split(train_data)
    grid = GridSearchCV(estimator=model, param_grid=param, cv=cv, n_jobs=-1, scoring='roc_auc', verbose=40, return_train_score=True)
    grid.fit(train_data, train_label)
    
    print("Best Parameters : ", grid.best_params_)
    print("Best Score : ", grid.best_score_)
    print("Best Estimator : ", grid.best_estimator_)
    
    return grid
def logisticRegression(train_data, train_label, test_data, test_label, regularization):
    grid = gridSearchCV(train_data, train_label, regularization)
    plot_gridsearch_result(grid)
    metrics_performance(grid, train_data, train_label, test_data, test_label)
# Load BOW Unigram Datasets
with open('train_data_BOW_Uni.pkl', 'rb') as file:
    train_data_uni = pickle.load(file)
    
with open('test_data_BOW_Uni.pkl', 'rb') as file:
    test_data_uni = pickle.load(file)

with open('train_label_BOW_Uni.pkl', 'rb') as file:
    train_label_uni = pickle.load(file)

with open('test_label_BOW_Uni.pkl', 'rb') as file:
    test_label_uni = pickle.load(file)
    
# Load BOW Bigram Datasets
with open('train_data_BOW_Bi.pkl', 'rb') as file:
    train_data_bi = pickle.load(file)
    
with open('test_data_BOW_Bi.pkl', 'rb') as file:
    test_data_bi = pickle.load(file)

with open('train_label_BOW_Bi.pkl', 'rb') as file:
    train_label_bi = pickle.load(file)

with open('test_label_BOW_Bi.pkl', 'rb') as file:
    test_label_bi = pickle.load(file)
    
# Standardize the data
train_data_uni=standardize(train_data_uni, False)
test_data_uni=standardize(test_data_uni, False)
train_data_bi=standardize(train_data_bi, False)
test_data_bi=standardize(test_data_bi, False)
logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l1')
logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l2')
logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l1')
logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l2')
# Load TF-IDF Unigram Datasets
with open('train_data_TF-IDF_Uni.pkl', 'rb') as file:
    train_data_uni = pickle.load(file)
    
with open('test_data_TF-IDF_Uni.pkl', 'rb') as file:
    test_data_uni = pickle.load(file)

with open('train_label_TF-IDF_Uni.pkl', 'rb') as file:
    train_label_uni = pickle.load(file)

with open('test_label_TF-IDF_Uni.pkl', 'rb') as file:
    test_label_uni = pickle.load(file)
   
#--------------------------------------------------------------------------------------------------------------------------------------
# Load TF-IDF Bigram Datasets
with open('train_data_TF-IDF_Bi.pkl', 'rb') as file:
    train_data_bi = pickle.load(file)
    
with open('test_data_TF-IDF_Bi.pkl', 'rb') as file:
    test_data_bi = pickle.load(file)

with open('train_label_TF-IDF_Bi.pkl', 'rb') as file:
    train_label_bi = pickle.load(file)

with open('test_label_TF-IDF_Bi.pkl', 'rb') as file:
    test_label_bi = pickle.load(file)
logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l1')
logisticRegression(train_data_uni, train_label_uni, test_data_uni, test_label_uni, 'l2')
logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l1')
logisticRegression(train_data_bi, train_label_bi, test_data_bi, test_label_bi, 'l2')
# Feature Importance - Top 100 Features of Positive Class
# Feature Importance - Top 100 Features of Negative Class