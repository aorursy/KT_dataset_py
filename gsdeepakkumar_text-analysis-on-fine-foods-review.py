import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import warnings 
import string
import re
import itertools
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud
warnings.filterwarnings('ignore')
%matplotlib inline

## Modelling :
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix,make_scorer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

### Read the dataset:
Kaggle=1
if Kaggle==0:
    review=pd.read_csv("Reviews.csv",parse_dates=["Time"])
else:
    review=pd.read_csv("../input/Reviews.csv",parse_dates=["Time"])
review.head()
review.describe()
print("There are {} unique product IDs and there are {} uniques users who have submitted their reviews.".format(review['ProductId'].nunique(),review['UserId'].nunique()))
plt.figure(figsize=(8,8))
ax=sns.countplot(review['Score'],color='skyblue')
ax.set_xlabel("Score")
ax.set_ylabel('Count')
ax.set_title("Distribution of Review Score")
## Borrowed from https://www.kaggle.com/neilash/team-ndl-algorithms-and-illnesses

plt.scatter(review.Score, review.HelpfulnessDenominator, c=review.Score.values, cmap='tab10')
plt.title('Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Useful Count')
plt.xticks([i for i in range(1,6)]);
### Borrowed from https://www.kaggle.com/neilash/team-ndl-algorithms-and-illnesses

# Create a list (cast into an array) containing the average usefulness for given ratings
use_ls = []

for i in range(1, 6):
    use_ls.append([i, np.sum(review[review.Score == i].HelpfulnessDenominator) / np.sum([review.Score == i])])
    
use_arr = np.asarray(use_ls)
plt.scatter(use_arr[:, 0], use_arr[:, 1], c=use_arr[:, 0], cmap='tab10', s=200)
plt.title('Average Useful Count vs Rating')
plt.xlabel('Rating')
plt.ylabel('Average Useful Count')
plt.xticks([i for i in range(1, 6)]);
useful_rating=review.sort_values('HelpfulnessDenominator',ascending=False)

# Print most helpful reviews:
for i in useful_rating.Text.iloc[:3]:
    print(i,'\n')
## Print least helpful reviews :
for i in useful_rating.Text.iloc[-3:]:
    print(i,'\n')
useful=review.groupby('ProfileName')['HelpfulnessDenominator'].mean().reset_index().sort_values('HelpfulnessDenominator',ascending=False)
useful.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='HelpfulnessDenominator',data=useful[:5],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Usefulness Rating by Profile-Top 5 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average People")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
scores=review.groupby('ProfileName')['Score'].mean().reset_index().sort_values(by='Score',ascending=False)
scores.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='Score',data=scores[:10],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Users with most positive scores-Top 10 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average Score")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
review['review_length']=review['Text'].str.len()
length=review.groupby('ProfileName')['review_length'].mean().reset_index().sort_values(by='review_length',ascending=False)
length.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x='ProfileName',y='review_length',data=length[:10],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Length of the review-Top 10 Users")
ax.set_xlabel("Profile Name")
ax.set_ylabel("Average Length")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
review_sample=review.sample(10000)
y=review_sample['Score']
x=review_sample['Text']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100,stratify=y)
print('Dimensions of train:{}'.format(X_train.shape),'\n','Dimensions of test:{}'.format(X_test.shape))
y_train.value_counts()
y_test.value_counts()
y_train.describe()
### Some preprocessing exercise in train dataset: - Inspired  from https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words
def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw review), and 
    # the output is a single string (a preprocessed review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    text_words = letters_only.lower()                          
    #              
    # 
    #4.Remove stopwords and Tokenize the text
    tokens = nltk.word_tokenize(text_words)
    tokens_text = [word for word in tokens if word not in set(nltk.corpus.stopwords.words('english'))]
    #
    #5.Lemmantize using wordnetLemmantiser:
    lemmantizer=WordNetLemmatizer()
    lemma_text = [lemmantizer.lemmatize(tokens) for tokens in tokens_text]
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( lemma_text ))   
# Get the number of reviews based on the dataframe column size


# Initialize an empty list to hold the clean reviews
X_train_clean = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for text in tqdm(X_train):
    # Call our function for each one, and add the result to the list of
    # clean reviewsb
    X_train_clean.append(review_to_words(text))
#https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
##Creating bag of words model :

vectorizer=CountVectorizer(ngram_range=(1,1)) 
train_feature=vectorizer.fit_transform(X_train_clean)

tfidf_transformer=TfidfVectorizer(ngram_range=(1,1))
train_feature_tfidf=tfidf_transformer.fit_transform(X_train_clean)
# Get the number of reviews based on the dataframe column size


# Initialize an empty list to hold the clean reviews
X_test_clean = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for text in tqdm(X_test):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    X_test_clean.append(review_to_words(text))
test_feature=vectorizer.transform(X_test_clean)
test_feature_tfidf=tfidf_transformer.transform(X_test_clean)
prediction=dict()
nb=MultinomialNB()
nb.fit(train_feature, y_train)
prediction['Naive']=nb.predict(test_feature)
print(accuracy_score(y_test,prediction['Naive']))
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
class_names = set(review['Score'])
cnf_matrix = confusion_matrix(y_test, prediction['Naive'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Naive Bayes Model')

nb.fit(train_feature_tfidf,y_train)
prediction['Naive_TFIDF']=nb.predict(test_feature_tfidf)
print(accuracy_score(y_test,prediction['Naive_TFIDF']))
lr=LogisticRegression()
lr.fit(train_feature,y_train)
prediction['Logit']=lr.predict(test_feature)
print(accuracy_score(y_test,prediction['Logit']))
lr.fit(train_feature_tfidf,y_train)
prediction['Logit_TFIDF']=lr.predict(test_feature_tfidf)
print(accuracy_score(y_test,prediction['Logit_TFIDF']))
sm=SMOTE(random_state=100)
X_sm,y_sm=sm.fit_sample(train_feature_tfidf,y_train)
Counter(y_sm)
scorer=make_scorer(accuracy_score)
# parameter grid
naive=MultinomialNB()
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Initialize Grid Search Model
model = GridSearchCV(estimator=naive, param_grid=param_grid, scoring=scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=5)

# Fit Grid Search Model
model.fit(X_sm, y_sm)  # Using the TF-IDF model for training .
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
prediction['Naive_SMOTE']=model.predict(test_feature_tfidf)
class_names = set(review['Score'])
cnf_matrix = confusion_matrix(y_test, prediction['Naive_SMOTE'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, Naive Bayes Model(After SMOTE and Grid Search)')
