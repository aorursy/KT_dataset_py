import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt # data visualization library
%matplotlib inline
import seaborn as sns

import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, recall_score


from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()
from wordcloud import WordCloud, STOPWORDS
from nltk import FreqDist
#select 5000 rows
df = pd.read_csv('/kaggle/input/vehicle/vehicle.csv', nrows= 5000)
df.sample(5)
words = set(nltk.corpus.words.words())
def normalizer(blogs):
    blogs = " ".join(filter(lambda x: x[0]!= '@' , blogs.split()))
    blogs = re.sub('[^a-zA-Z]', ' ', blogs)
    blogs = blogs.lower()
    blogs = re.sub(' +', ' ', blogs).strip()
    blogs = blogs.split()
    blogs = [word for word in blogs if not word in set(stopwords.words('english'))]
    blogs = [lemma.lemmatize(word) for word in blogs]
    
    blogs = " ".join(blogs)
    return blogs
df['normalized_text'] = df.text.apply(normalizer)
df.head()
# Remove Non-English Words from Normalized text
def remove_non_english_words(blog):
    return " ".join(w for w in nltk.wordpunct_tokenize(blog) if w.lower() in words or not w.isalpha())

df['normalized_text'] = df.normalized_text.apply(remove_non_english_words)
df.head()
# all tweets 
all_words = " ".join(df.normalized_text)
wordcloud = WordCloud(height=2000, width=2000, stopwords=STOPWORDS, background_color='white')
wordcloud = wordcloud.generate(all_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
##  create another dataframe dfT having only the columns needed for creating label
dfT=df[['gender', 'age', 'topic', 'sign']]
## Convert age from int type into String
dfT['age']=dfT['age'].astype('str')
## Create a 2D Matrix 'm' which is list of list contaning 'gender', 'age', 'topic', 'sign' for each row
m=[]                              # 2D Matrix having list of list
for i in range(dfT.shape[0]):
    g=[]                          # 1D list of 'gender', 'age', 'topic', 'sign'
    for j in range(dfT.shape[1]):
        g.append(dfT.iloc[i][j])
    m.append(g)
#Add a column called labels
df['labels']=m
df.head()
final_df = df[['normalized_text', 'labels']]
final_df.head()
# Lets Check Distribution of Labels
final_df['labels'].astype('str').value_counts()
## Check for Null Values
final_df.isna().sum()
# No Null Values
X = final_df['normalized_text']
y = final_df['labels']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.25)
# Consider only those rows which occur more than 15% and less than 80 %, also restrict features to 100

vectorizer = CountVectorizer(ngram_range = (1,2), stop_words=stopwords.words('english'), 
                             min_df = 0.15, max_df = 0.8, max_features = 100)
# transform the X data to document_term_matrix

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)
X_train_dtm
# check the vocabulary( First 15 features)
vectorizer.get_feature_names()[:10]

print(X_train_dtm )
# examine vocabulary and document term matrix together
pd.DataFrame(X_train_dtm.toarray(), columns = vectorizer.get_feature_names())


print(X_train_dtm )
# examine vocabulary and document term matrix together
pd.DataFrame(X_test_dtm.toarray(), columns = vectorizer.get_feature_names())
dfT = df[['gender', 'age', 'topic', 'sign']]
dfT['age'] = dfT['age'].astype('str')
keys=[] 
values=[] 

for i in range(dfT.shape[1]): # iterate through all the colummns        
    for j in range(dfT.iloc[:,i].value_counts().shape[0]): # iterate through all the rows of value_counts of that column
        keys.append(dfT.iloc[:,i].value_counts().index[j])         
        values.append(dfT.iloc[:,i].value_counts().iloc[j])
dictionary = dict(zip(keys,values))
print(dictionary)
from sklearn.preprocessing import MultiLabelBinarizer 
mlb = MultiLabelBinarizer(classes=sorted(dictionary.keys()))
y_train_mlb = mlb.fit_transform(y_train)
y_test_mlb = mlb.transform(y_test)
y_train_mlb[0]
y_test_mlb[0]
y_train.iloc[1]
mlb.inverse_transform(y_train_mlb)[1]
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', multi_class='ovr')
ovr = OneVsRestClassifier(lr)

ovr.fit(X_train_dtm, y_train_mlb)
y_pred_ovr_test = ovr.predict(X_test_dtm)
#y_proba_ovr = ovr.predict_proba(X_test_dtm)
y_pred_ovr_test
y_pred_ovr_train = ovr.predict(X_train_dtm)
y_pred_ovr_train
def print_scores(actual, predicted, averaging_type):
    print('\nAVERAGING TYPE==> ',averaging_type)
    print('F1 score: ',f1_score(actual,predicted, average=averaging_type))
    print('Average Precision Score: ',average_precision_score(actual,predicted, average=averaging_type))
    print('Average Recall Score: ',recall_score(actual,predicted, average=averaging_type))
print('--------------------------TRAIN SCORES--------------------------------')
print('Accuracy score: ',accuracy_score(y_train_mlb, y_pred_ovr_train))
print_scores(y_train_mlb, y_pred_ovr_train, 'micro')
print_scores(y_train_mlb, y_pred_ovr_train, 'macro')
print_scores(y_train_mlb, y_pred_ovr_train, 'weighted')
print('--------------------------TEST SCORES--------------------------------')
print('Accuracy score: ',accuracy_score(y_test_mlb, y_pred_ovr_test))
print_scores(y_test_mlb, y_pred_ovr_test, 'micro')
print_scores(y_test_mlb, y_pred_ovr_test, 'macro')
print_scores(y_test_mlb, y_pred_ovr_test, 'weighted')
five_pred = y_pred_ovr_test[:5]
five_actual = y_test_mlb[:5]
five_actual = mlb.inverse_transform(five_actual)
five_actual
five_pred = mlb.inverse_transform(five_pred)
five_pred