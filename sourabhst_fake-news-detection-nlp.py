import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('../input/textdb3/fake_or_real_news.csv')    #importing the Dataset from local Machine
df.head()
# Check how the Lables are Distributed

df['label'].value_counts()
import string as st    #importing string for the function
#Remove all punctuations

def remove_punctuation(text):
    return ("".join([ch for ch in text if ch not in st.punctuation]))
df['New_text']=df['text'].apply(lambda x: remove_punctuation(x))     
df.head()
df = df.drop(['Unnamed: 0'],axis=1)    #Dropping the feature which we don't need
df.head()
#Convert text in lower case, Split() applied for white space
import re
def tokenize(text):
    text = re.split('\s+', text)
    return [x.lower() for x in text]
df['New_text'] = df['New_text'].apply(lambda msg:tokenize(msg))
df.head()
#Removal of tokens less than length 2

def rem_small_words(text):
    return [x for x in text if len(x)>2]
df['New_text'] = df['New_text'].apply(lambda x: rem_small_words(x))
df.head()
#Remove stopwords
import nltk
from nltk import PorterStemmer, WordNetLemmatizer

def rem_stopword(text):
    return[word for word in text if word not in nltk.corpus.stopwords.words('english')]
df['New_text'] = df['New_text'].apply(lambda x: rem_stopword(x))
df.head()
#Lemmetization
def lemmatizer(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]
df['New_text'] = df['New_text'].apply(lambda x: lemmatizer(x))
df.head(10)
# Create sentences to get clean text as input for vectors
def return_setances(tokens):
    return " ".join([word for word in tokens])
df['New_text'] = df['New_text'].apply(lambda x: return_setances(x))
df.head()
df.sample(10)
from wordcloud import WordCloud

# Create and generate a word cloud image FAKE_NEWS
fake_data = df[df["label"] == "FAKE"]
fake_text = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(fake_text)
plt.figure(figsize= [20,10])
plt.imshow(wordcloud)
plt.axis("off")
plt.title('FAKE NEWS WORDCLOUD',fontsize= 30)
plt.show()



# Create and generate a word cloud image REAL_NEWS
real_data = df[df["label"] == "REAL"]
real_text = ' '.join([text for text in real_data.text])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(real_text)
plt.figure(figsize= [20,10])
plt.imshow(wordcloud)
plt.axis("off")
plt.title('REAL NEWS WORDCLOUD',fontsize= 30)
plt.show()
#Comparing the frequency of Fake and Real News
import seaborn as sns          
print(df.groupby(['label'])['text'].count())
sns.countplot(df['label'])
#Comparing the Total numbers of Characters in the Feature Title

fig, (ax1,ax2)=plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Characters in News Title',fontsize=20)
news_len=df[df['label']=='REAL']['title'].str.len()
ax1.hist(news_len,color='orange',linewidth=2,edgecolor='black')
ax1.set_title('REAL news',fontsize=15)
news_len=df[df['label']=='FAKE']['title'].str.len()
ax2.hist(news_len,linewidth=2,edgecolor='black')
ax2.set_title('Fake news',fontsize=15)
#Comparing the Total numbers of Characters in the Feature Text

fig, (ax1,ax2)=plt.subplots(1,2,figsize=(15,8))
fig.suptitle('Characters in News Text',fontsize=20)
news_len=df[df['label']=='REAL']['text'].str.len()
ax1.hist(news_len,color='orange',linewidth=2,edgecolor='black')
ax1.set_title('REAL news',fontsize=15)
news_len=df[df['label']=='FAKE']['text'].str.len()
ax2.hist(news_len,linewidth=2,edgecolor='black')
ax2.set_title('Fake news',fontsize=15)

#creating a bag of words with the consecutive frequency for fake text

import nltk
import seaborn as sns
fake_text_vis =' '.join([str(x) for x in df[df['label']=='FAKE']['New_text']])
a = nltk.FreqDist(fake_text_vis.split())
d = pd.DataFrame({'Word': list(a.keys()),
                  'Count': list(a.values())})
d.sample(10)
# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Word", y = "Count")
ax.set_xticklabels(d["Word"], rotation=40, ha="right")
ax.set(ylabel = 'Count')
plt.show()
#creating a bag of words with the consecutive frequency for Real text
import nltk
import seaborn as sns
real_text_vis =' '.join([str(x) for x in df[df['label']=='REAL']['New_text']])
a = nltk.FreqDist(real_text_vis.split())
d = pd.DataFrame({'Word': list(a.keys()),
                  'Count': list(a.values())})
d.sample(10)
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Word", y = "Count")
ax.set_xticklabels(d["Word"], rotation=40, ha="right")
ax.set(ylabel = 'Count')
plt.show()
df.head()
df["label"]=df["label"].replace(["FAKE","REAL"],value=[1,0]) #Label Encoding
df.head()
df.info()

X_train,X_test,y_train,y_test = train_test_split(df['New_text'],df['label'],test_size=0.2, random_state = 10)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)
print(X_train.shape)
print(X_test.shape)
from sklearn.metrics import accuracy_score, confusion_matrix
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

model1= RandomForestClassifier()
model1.fit(X_train,y_train)
pred1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test,pred1)
cm1 = confusion_matrix(y_test,pred1)
print("Accuracy score : {}".format(accuracy1))
print("Confusion matrix : \n {}".format(cm1))
# Logistic Regression model
from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression(max_iter = 500)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test,pred2)
cm2 = confusion_matrix(y_test,pred2)
print("Accuracy score : {}".format(accuracy2))
print("Confusion matrix : \n {}".format(cm2))
#Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
model3 = PassiveAggressiveClassifier(max_iter=50)
model3.fit(X_train,y_train)

pred3 = model3.predict(X_test)

accuracy3 = accuracy_score(y_test,pred3)
cm3 = confusion_matrix(y_test,pred3)
print("Accuracy score : {}".format(accuracy3))
print("Confusion matrix : \n {}".format(cm3))
#Support Vector Classification.
from sklearn.svm import SVC
model4=SVC()
model4.fit(X_train,y_train)

pred4 = model4.predict(X_test)

accuracy4 = accuracy_score(y_test,pred4)
cm4 = confusion_matrix(y_test,pred4)
print("Accuracy score : {}".format(accuracy4))
print("Confusion matrix : \n {}".format(cm4))
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model5=DecisionTreeClassifier()
model5.fit(X_train,y_train)

pred5 = model5.predict(X_test)

accuracy5 = accuracy_score(y_test,pred5)
cm5 = confusion_matrix(y_test,pred5)
print("Accuracy score : {}".format(accuracy5))
print("Confusion matrix : \n {}".format(cm5))
#Creating the Dictionary with model name as key adn accuracy as key-value
labels={'RandomForestClassifier':accuracy1,'LogisticRegression':accuracy2,'PassiveAggressiveClassifier':accuracy3,
        'SVC':accuracy4,'DecisionTreeClassifier':accuracy5}
#Plotting accuracy of all the models with Bar-Graphs
plt.figure(figsize=(15,8))
plt.title('Comparing Accuracy of ML Models',fontsize=20)
colors=['red','yellow','orange','magenta','cyan']
plt.xticks(fontsize=10,color='black')
plt.yticks(fontsize=20,color='black')
plt.ylabel('Accuracy',fontsize=20)
plt.xlabel('Models',fontsize=20)
plt.bar(labels.keys(),labels.values(),edgecolor='black',color=colors, linewidth=2,alpha=0.5)
from mlxtend.plotting import plot_confusion_matrix

print("Confusion Matrix for RandomForestClassifier")
print("Accuracy score : {}".format(accuracy1))
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL'])
print("Confusion Matrix for Logistic Regression")
print("Accuracy score : {}".format(accuracy2))

plot_confusion_matrix(conf_mat=cm2,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL'])
print("Confusion Matrix for PassiveAggressiveClassifier")
print("Accuracy score : {}".format(accuracy3))
plot_confusion_matrix(conf_mat=cm3,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL'])
print("Confusion Matrix for SVC")
print("Accuracy score : {}".format(accuracy4))
plot_confusion_matrix(conf_mat=cm4,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL'])
print("Confusion Matrix for DecisionTreeClassifier")
print("Accuracy score : {}".format(accuracy5))
plot_confusion_matrix(conf_mat=cm5,show_absolute=True,
                                show_normed=True,
                                colorbar=True,class_names=['FAKE','REAL'])
