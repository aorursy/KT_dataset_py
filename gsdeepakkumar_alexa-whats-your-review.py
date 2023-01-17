import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import string
import itertools

import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc,accuracy_score
from nltk.stem.porter import PorterStemmer
warnings.filterwarnings("ignore")
#nltk.download('punkt')
Kaggle=1
if Kaggle==0:
    reviews =pd.read_csv("amazon_alexa.tsv",sep="\t")
    
else:
    reviews = pd.read_csv("../input/amazon_alexa.tsv",sep="\t")
    
reviews.head()
reviews.describe()
plt.figure(figsize=(8,8))
ax=sns.countplot(reviews['rating'],palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Distribution of the Amazon Alexa Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
variant_rating=reviews.groupby('variation')['rating'].mean().reset_index()
variant_rating.head()
plt.figure(figsize=(8,8))
ax=sns.barplot(x='variation',y='rating',data=variant_rating,palette=sns.color_palette(palette="viridis_r"))
ax.set_title("Average Rating based on Alexa Variant")
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_xlabel("Rating")
ax.set_ylabel("Count")
rating_review=reviews.sort_values(by='rating',ascending=False)
rating_review.head()
for i in rating_review['verified_reviews'].iloc[:6]:
    print(i, '\n')
for i in rating_review['verified_reviews'].iloc[-6:]:
    print(i, '\n')
rating_review['date'] = pd.to_datetime(rating_review['date'], errors='coerce')
month_count = rating_review['date'].dt.month.value_counts()
month_count = month_count.sort_index()
plt.figure(figsize=(9,6))
sns.barplot(month_count.index, month_count.values,color='green',alpha=0.4)
plt.xticks(rotation='vertical')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Reviews per Month")
plt.show()
weekday_count = rating_review['date'].dt.weekday_name.value_counts()
weekday_count = weekday_count.sort_index()
plt.figure(figsize=(9,6))
sns.barplot(weekday_count.index, weekday_count.values,color='green',alpha=0.4,order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.xticks(rotation='vertical')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Reviews by Weekday")
plt.show()
rating_review['weekday']=rating_review['date'].dt.weekday_name
avg_weekday=rating_review.groupby('weekday')['rating'].mean()
plt.figure(figsize=(9,6))
sns.barplot(avg_weekday.index, avg_weekday.values,color='green',alpha=0.4,order=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])
plt.xticks(rotation='vertical')
plt.xlabel('Weekday', fontsize=12)
plt.ylabel('Avg Rating', fontsize=12)
plt.title("Average Rating over the day")
plt.show()
def sentiment(x):
    if x > 3:
        return 'positive'
    else:
        return 'negative'
        
rating = rating_review['rating']
rating=rating.map(sentiment)
review=rating_review['verified_reviews']
rating.describe()
X_train,X_test,y_train,y_test=train_test_split(review,rating,test_size=0.2,stratify=rating,random_state=100)
print("Shape of train is {} and shape of test is {}".format(X_train.shape,X_test.shape));
### Borrowed from https://www.kaggle.com/gpayen/building-a-prediction-model

stemmer = PorterStemmer()

## Stemming :

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

## Tokenisation:

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]
    stems = stem_tokens(tokens, stemmer)
    return ' '.join(stems)

## Removing the punctuation:

intab = string.punctuation
outtab = "                                "
trantab = str.maketrans(intab, outtab)  ### Remove the punctuations 

#--- Training set

corpus = []
for text in X_train:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    corpus.append(text)
corpus[:5]
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(corpus)        
        
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#--- Test set

test_set = []
for text in X_test:
    text = text.lower()
    text = text.translate(trantab)
    text=tokenize(text)
    test_set.append(text)

X_new_counts = count_vect.transform(test_set)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)

prediction = dict()
model =MultinomialNB()
model.fit(X_train_tfidf,y_train)
prediction['Naive Bayes']=model.predict(X_test_tfidf)
print(accuracy_score(y_test,prediction['Naive Bayes']))
df_confusion = pd.crosstab(y_test,prediction['Naive Bayes'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion
model = LogisticRegression()
model.fit(X_train_tfidf,y_train)
prediction['Logit']=model.predict(X_test_tfidf)
print(accuracy_score(y_test,prediction['Logit']))
df_confusion = pd.crosstab(y_test,prediction['Logit'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion
#Borrowed from https://www.kaggle.com/gpayen/building-a-prediction-model
def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# Inspired from https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets

from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=100)
X_sm,y_sm=smote.fit_sample(X_train_tfidf,y_train)  ### Oversampling the training dataset.
## Applying the Naive Bayes and Logit again on the model:
model =MultinomialNB()
model.fit(X_sm,y_sm)
prediction['Naive Bayes_SMOTE']=model.predict(X_test_tfidf)


df_confusion_SMOTE = pd.crosstab(y_test,prediction['Naive Bayes_SMOTE'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion_SMOTE

model = LogisticRegression()
model.fit(X_sm,y_sm)
prediction['Logit_SMOTE']=model.predict(X_test_tfidf)


df_confusion = pd.crosstab(y_test,prediction['Logit_SMOTE'], rownames=['Actual'], colnames=['Predicted'], margins=True)
df_confusion
def formatt(x):
    if x == 'negative':
        return 0
    return 1
vfunc = np.vectorize(formatt)

cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1

plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()