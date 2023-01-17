import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt

cols = ['target','id','date','query_string','user','text']
df = pd.read_csv('../input/twitter-sentiment-analysis/sentiment3.csv', header=None, names=cols)
df.head()
df.isnull().sum()
import seaborn as sns
sns.countplot(x = 'target', data = df)
df.drop(['id','date','query_string','user'],axis=1,inplace=True)
df.head()
df[df.target == 4].head()
df[df.target == 0].head()
df['pre_clean_len'] = [len(t) for t in df.text]   
#boxplot for length of the text
fig, ax = plt.subplots(figsize=(5,5))
plt.boxplot(df.pre_clean_len)
plt.show()
df[df.pre_clean_len > 140]
#only one result is more than 175 charactors
df[df.pre_clean_len > 175]  
df.drop(['pre_clean_len'],axis=1,inplace=True)
import string
import nltk
import warnings 
import re
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt   
# remove twitter handles (@user)
df['text'] = np.vectorize(remove_pattern)(df['text'], "@[\w]*")
df.head()
# remove twitter url
df['text'] = np.vectorize(remove_pattern)(df['text'], "https?://[A-Za-z0-9./]+")
df.head()
# remove special characters, numbers, punctuations
df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
df.head()
#remove short word
df['text'] = df['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
df.head()
#tokenization
tokenized_df= df['text'].apply(lambda x: x.split())
tokenized_df.head()
from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_df = tokenized_df.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_df.head()
import nltk
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

tokenized_df = tokenized_df.apply(lambda x: lemmatizer(x))
tokenized_df.head()
for i in range(len(tokenized_df)):
    tokenized_df[i] = ' '.join(tokenized_df[i])

df['text'] = tokenized_df
df.loc[df['target'] ==4, 'target'] = 1
clean_df = pd.DataFrame(df.text,columns=['text'])
clean_df['target'] = df.target
clean_df.to_csv('clean_sentiment.csv',encoding='utf-8')
from wordcloud import WordCloud, STOPWORDS
all_words = ' '.join([text for text in df['text']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
df.head() #prove of token alrady swtiched back together
neg_tweets = df[df.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
pos_tweets = df[df.target == 1 ]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags
# extracting hashtags from positive/negative tweets
HT_positive = hashtag_extract(df['text'][df['target'] == 1])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(df['text'][df['target'] == 0])

# unnesting list
HT_positive = sum(HT_positive,[])
HT_negative = sum(HT_negative,[])
#positive tweet
a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()
#tweete token visualisation
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer(stop_words='english',max_features=10000)   #insert stopword
cvec.fit(df.text)
document_matrix = cvec.transform(df.text)
%%time
neg_batches = np.linspace(0,80000,10).astype(int)
i=0
neg_tf = []
while i < len(neg_batches)-1:
    batch_result = np.sum(document_matrix[neg_batches[i]:neg_batches[i+1]].toarray(),axis=0)
    neg_tf.append(batch_result)
    print(neg_batches[i+1],"entries' term freuquency calculated")
    i += 1
%%time
pos_batches = np.linspace(80000,160000,10).astype(int)
i=0
pos_tf = []
while i < len(pos_batches)-1:
    batch_result = np.sum(document_matrix[pos_batches[i]:pos_batches[i+1]].toarray(),axis=0)
    pos_tf.append(batch_result)
    print(pos_batches[i+1],"entries' term freuquency calculated")
    i += 1
neg = np.sum(neg_tf,axis=0)
pos = np.sum(pos_tf,axis=0)
term_freq_df2 = pd.DataFrame([neg,pos],columns=cvec.get_feature_names()).transpose()
term_freq_df2.columns = ['negative', 'positive']
term_freq_df2['total'] = term_freq_df2['negative'] + term_freq_df2['positive']
term_freq_df2.sort_values(by='total', ascending=False).iloc[:10]
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df2.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')
import seaborn as sns
plt.figure(figsize=(8,6))
ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df2)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')
plt.figure(figsize=(8,6))
ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df2)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')
from sklearn.model_selection import train_test_split

x = df.text
y = df.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.02, random_state=2000)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,2), max_features=60000, stop_words='english')
countX_train = cv.fit_transform(X_train)
countX_test = cv.transform(X_test)
from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             max_features=200000, 
                             stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
#Import svm model
from sklearn import svm
from sklearn.metrics import classification_report
#Create a svm Classifier
clf = svm.SVC(kernel='linear')

#Train the model using the training sets
clf.fit(X_train_vectors, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test_vectors)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train_vectors, y_train)
print(grid.best_estimator_)
print(model_svm.best_score_)