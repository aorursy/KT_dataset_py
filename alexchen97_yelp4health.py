# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#make compatible with Python 2 and Python 3
from __future__ import print_function, division, absolute_import 

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#import packages

import bs4 as bs
import nltk
nltk.download('stopwords')

import re
from nltk.tokenize import sent_tokenize # tokenizes sentences
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

eng_stopwords = stopwords.words('english')
from wordcloud import WordCloud
import multidict as multidict
import os
import re
from PIL import Image
from os import path
df_raw = pd.read_csv("/kaggle/input/live-score/Compiled_Yelp_Scraped_Reviews_Apr_29.csv")

# Clean the dataframe by getting rid of non-essential columns 
df = df_raw.drop(columns = ['0','alias', 'address2', 'formatted_address'], axis = 1)

# Print data shape
print("Review data shape is {}.\n".format(df.shape))
## Yelp phone
yelp_unique_id = df['id'].unique()
print("Based on business id, Yelp Dataset has {} unique businesses.".format(len(yelp_unique_id)))

#Preview data
df.head()
live_raw = pd.read_csv("/kaggle/input/live-score/Restaurant_Scores_-_LIVES_Standard.csv")
# Use the necessary columns:
necessary_list = ['business_id','business_name','business_address','inspection_id','inspection_date','inspection_score','inspection_type','violation_id','violation_description','risk_category']
live = live_raw.loc[:,necessary_list]
#Drop the missing value for inspection_score:
live = live.dropna()

# Print data shape
print("Review data shape is {}.\n".format(live.shape))
# unique business
live_unique_business = live['business_id'].unique()
print("Live Dataset has {} unique business".format(len(live_unique_business)))

live.head()
# plt.figure(figsize=(20,16));
sns.catplot(x="risk_category", y="inspection_score",kind='boxen',order = ["Low Risk", "Moderate Risk", "High Risk"], data=live, palette="Reds")
plt.title("Inspection Score by Risk Category", fontsize=18);
plt.xlabel("Risk Category",fontsize=12)
plt.ylabel("Inspection Score",fontsize=12)
plt.show()
# Phone

## Yelp phone
yelp_unique_phone = df['phone'].unique()
print("Yelp Dataset has {} unique phones".format(len(yelp_unique_phone)))
#LIVE addresses
live_unique_phone= live_raw['business_phone_number'].unique()
print("LIVE Dataset has {} unique phones".format(len(live_unique_phone)))
## Test overlap
yelp_unique_phone_set = set(yelp_unique_phone)
live_unique_phone_set = set(live_unique_phone)
overlap_phone_set = yelp_unique_phone_set.intersection(live_unique_phone_set)
print("For Address name: They have {} overlap address names".format(len(overlap_phone_set)))
##################

# business

# Yelp Business
yelp_unique_business = df['name'].unique()
print("Yelp Dataset has {} unique restaurants".format(len(yelp_unique_business)))
live_unique_business = live['business_name'].unique()
print("LIVE Dataset has {} unique restaurants".format(len(live_unique_business)))
## Test overlap
yelp_unique_business_set = set(yelp_unique_business)
live_unique_business_set = set(live_unique_business)
overlap_set = yelp_unique_business_set.intersection(live_unique_business_set)
print("For business name: They have {} overlap business names".format(len(overlap_set)))
#####################

# Address

#Yelp addresses
yelp_unique_address = df['address1'].unique()
print("Yelp Dataset has {} unique addresses".format(len(yelp_unique_address)))
#LIVE addresses
live_unique_address = live_raw['business_address'].unique()
print("LIVE Dataset has {} unique addresses".format(len(live_unique_address)))
## Test overlap
yelp_unique_address_set = set(yelp_unique_address)
live_unique_address_set = set(live_unique_address)
overlap_address_set = yelp_unique_address_set.intersection(live_unique_address_set)
print("For Address name: They have {} overlap address names".format(len(overlap_address_set)))
def review_cleaner(reviews, lemmatize=True, stem=False):
    '''
    Clean and preprocess a review.

    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    cleaned_reviews=[]
    
    for i, review in enumerate(df['text']):
        # print progress
        if((i+1)%500 == 0):
            print("Done with %d reviews" %(i+1))
            
        #1. Remove HTML tags
        review = bs.BeautifulSoup(review).text

        #2. Use regex to find emojis
        emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)

        #3. Remove punctuation
        review = re.sub("[^a-zA-Z]", " ",review)

        #4. Tokenize into words (all lower case)
        review = review.lower().split()

        #5. Remove stopwords
        eng_stopwords = set(stopwords.words("english"))
            
        clean_review=[]
        for word in review:
            if word not in eng_stopwords:
                if lemmatize is True:
                    word=wnl.lemmatize(word)
                elif stem is True:
                    if word == 'oed':
                        continue
                    word=ps.stem(word)
                clean_review.append(word)

        #6. Join the review to one sentence
        
        review_processed = ' '.join(clean_review+emojis)
        cleaned_reviews.append(review_processed)
    

    return(cleaned_reviews)
original_clean_reviews=review_cleaner(df['text'], lemmatize = False, stem = False)
for j in range(len(original_clean_reviews)):
    print(j, original_clean_reviews[j])
## Add a clean review column
df['clean_review'] = original_clean_reviews
def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict


def makeImage(text):
    alice_mask = np.array(Image.open("/kaggle/input/live-score/Yelp_logo.png"))

    wc = WordCloud(background_color="white", max_words=1000, mask=alice_mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
# wordcloud = WordCloud().generate(df.text[1])
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis("off")

# makeImage(getFrequencyDictForText(df.text[1]))
#### Irem's code:
df.head()
live.head()
new = live[['business_address','business_name','inspection_score']]
new_bus_address = new.groupby(["business_address","business_name"]).mean().reset_index()
new_bus_address.head()
# new_inspection_score = new.groupby("business_address",as_index = False)["inspection_score"].mean().to_frame()
# new_inspection_score.head()
# grouped = new.groupby(["business_address"])
# new_inspection_score = grouped.aggregate(np.mean)
# new_inspection_score
#new['inspection_date'] = new.groupby("business_address")['inspection_score'].mean()
# new_inspection_score = live[['business_address','business_name','inspection_score']].groupby("business_address")['inspection_score'].mean()
# new_group.head()
#new_group.join(new_inspection_score, on="business_address" )
#new_inspection_score.head()
# Live: business address, [average]inspection score

new_df = df[['name','address1','clean_review']]
grouped_df = new_df.groupby(['name','address1'])['clean_review'].apply(lambda x: ' '.join(x)).reset_index()
grouped_df.columns = ['business_name','business_address','clean_review']
grouped_df.sample(10)

reviews = pd.merge(grouped_df, new_bus_address[['business_address','inspection_score']], how='inner',on='business_address' )
reviews.head(10)
#set threshold
threshold = 85

#build a label
reviews['label'] = reviews['inspection_score'].apply(lambda x: 1 if x >=threshold else 0)
print(reviews.shape)
reviews.head(10)
label_1 = reviews[reviews.label ==1 ]
label_0 = reviews[reviews.label ==0 ]
label_1_review_str = ' '.join(label_1['clean_review'])
label_0_review_str = ' '.join(label_0['clean_review'])
wordcloud = WordCloud().generate(label_1_review_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
makeImage(getFrequencyDictForText(label_1_review_str))
makeImage(getFrequencyDictForText(label_0_review_str))
wordcloud = WordCloud().generate(label_1_review_str)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
a = ['a b','b c']
b = ' '.join(a)
b

from sklearn.ensemble import RandomForestClassifier
# CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
count_vect = CountVectorizer(analyzer="word")
X_count = count_vect.fit_transform(reviews['clean_review'])
X_features_count = pd.DataFrame(X_count.toarray())
X_features_count.columns = count_vect.get_feature_names()
print(X_features_count.shape)
X_features_count.head()
tfidf_vect = TfidfVectorizer(analyzer="word")
X_tfidf = tfidf_vect.fit_transform(reviews['clean_review'])
X_features_tfidf = pd.DataFrame(X_tfidf.toarray())
X_features_tfidf.columns = tfidf_vect.get_feature_names()
print(X_features_tfidf.shape)
X_features_tfidf.head()
X_train, X_test, y_train, y_test = train_test_split(X_features_tfidf, reviews['label'], test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1) # instantiate
rf_model = rf.fit(X_train, y_train) # fit
accuracy_rf = rf_model.score(X_test, y_test) # predict + evaluate

print('Random Forest Regression labeling accuracy:', str(round(accuracy_rf*100,2)),'%')
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]

## K fold 
# k_fold = KFold(n_splits=5)
# cross_val_score(rf, X_features_tfidf, reviews['label'], cv=k_fold, scoring='accuracy', n_jobs=-1)
y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')

print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                      round((y_pred==y_test).sum() / len(y_pred),3)))
rf = RandomForestClassifier(n_jobs=-1)
k_fold = KFold(n_splits=5)
cross_val_score(rf, X_features_tfidf, reviews['label'], cv=k_fold, scoring='accuracy', n_jobs=-1)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
# Look at importnace of features for random forest

def plot_model_var_imp(model , X , y):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print ('Training accuracy Random Forest:',model.score( X , y ))

plot_model_var_imp(rf, X_train, y_train)
logreg = LogisticRegression()                           # instantiate
logreg.fit(X_train, y_train)                            # fit
Y_pred = logreg.predict(X_test)                         # predict
acc_logreg = logreg.score(X_test, y_test)                # evaluate

print('Logistic Regression labeling accuracy:', str(round(acc_logreg*100,2)),'%')
knn = KNeighborsClassifier(n_neighbors = 3)                  # instantiate
knn.fit(X_train, y_train)                                    # fit
acc_knn = knn.score(X_test, y_test)                          # predict + evaluate

print('K-Nearest Neighbors labeling accuracy:', str(round(acc_knn*100,2)),'%')
# XGBoost, same API as scikit-learn
gradboost = xgb.XGBClassifier(n_estimators=10)             # instantiate
gradboost.fit(X_train, y_train)                              # fit
accuracy_xgboost = gradboost.score(X_test, y_test)           # predict + evalute

print('XGBoost labeling accuracy:', str(round(accuracy_xgboost*100,2)),'%')

from sklearn.ensemble import RandomForestClassifier
# CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


np.random.seed(100)


def train_predict_sentiment(cleaned_reviews, y=df["..."], ngram=1, max_features=1000):
    '''This function will:
    1. split data into train and test set.
    2. get n-gram counts from cleaned reviews 
    3. train a random forest model using train n-gram counts and y (labels)
    4. test the model on your test split
    5. print accuracy of sentiment prediction on test and training data
    6. print confusion matrix on test data results

    To change n-gram type, set value of ngram argument
    To change the number of features you want the countvectorizer to generate, set the value of max_features argument'''

    print("Creating the bag of words model!\n")
    # CountVectorizer is scikit-learn's bag of words tool, here we show more keywords 
    vectorizer = CountVectorizer(ngram_range=(1, ngram), analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = max_features) 
    
    X_train, X_test, y_train, y_test = train_test_split(cleaned_reviews, y, random_state=0, test_size=.2)

    # Then we use fit_transform() to fit the model / learn the vocabulary,
    # then transform the data into feature vectors.
    # The input should be a list of strings. .toarraty() converts to a numpy array
    
    train_bag = vectorizer.fit_transform(X_train).toarray()
    test_bag = vectorizer.transform(X_test).toarray()
    # print('TOP 20 FEATURES ARE: ',(vectorizer.get_feature_names()[:20]))

    print("Training the random forest classifier!\n")
    # Initialize a Random Forest classifier with 75 trees
    forest = RandomForestClassifier(n_estimators = 50) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the target variable
    forest = forest.fit(train_bag, y_train)


    train_predictions = forest.predict(train_bag)
    test_predictions = forest.predict(test_bag)
    
    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print(" The training accuracy is: ", train_acc, "\n", "The validation accuracy is: ", valid_acc)
    print()
    print('CONFUSION MATRIX:')
    print('         Predicted')
    print('          neg pos')
    print(' Actual')
    c=confusion_matrix(y_test, test_predictions)
    print('  neg  ',c[0])
    print('  pos  ',c[1])

    #Extract feature importnace
    print('\nTOP TEN IMPORTANT FEATURES:')
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:20]
    print([vectorizer.get_feature_names()[ind] for ind in top_10])
# Clean the reviews in the training set 'train' using review_cleaner function defined above
# Here we use the original reviews without lemmatizing and stemming
original_clean_reviews = review_cleaner(df['text'],lemmatize=False,stem=False)
train_predict_sentiment(cleaned_reviews = original_clean_reviews, y = df["..."], ngram=1, max_features=1000)
