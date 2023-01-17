import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',
             usecols = [0,1], encoding = 'ISO-8859-1')
df.rename(columns = {'v1': 'Category','v2': 'Message'},inplace = True)
df.head()
df.shape
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
%matplotlib inline 
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score,classification_report
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
nltk.download("stopwords")
nltk.download('punkt')
spam_df = df[df['Category'] == 'spam'] #create sub-dataframe of spam text
ham_df = df[df['Category'] == 'ham'] #sub-dataframe of ham text
stop_words = set(stopwords.words('english'))
def wordCount(text): #this contains all the information about the processed length
    try:
        text = text.lower()
        regex = re.compile('['+re.escape(string.punctuation) + '0-9\\r\\t\\n'+']') 
        txt = regex.sub(' ',text)  #remove punctuation
        words = [w for w in txt.split(' ')
                if w not in stop_words and len(w)>3] # remove stop words and words with length smaller than 3 letters
        return len(words)
    except:
        return 0
spam_df['len'] = spam_df['Message'].apply(lambda x: len([w for w in x.split(' ')]))
ham_df['len'] = ham_df['Message'].apply(lambda x: len([w for w in x.split(' ')]))
spam_df['processed_len'] = spam_df['Message'].apply(lambda x: wordCount(x))
ham_df['processed_len'] = ham_df['Message'].apply(lambda x: wordCount(x))
spam_df['punct']=spam_df['Message'].apply(lambda l1: sum([1 for x in l1 if x in set(string.punctuation)]))
ham_df['punct']= ham_df['Message'].apply(lambda l1: sum([1 for x in l1 if x in set(string.punctuation)]))
spam_df.head()
ham_df.head()
print ('spam length info')
print (spam_df[['len', 'processed_len']].describe())
print ('ham length info')
print (ham_df[['len', 'processed_len']].describe())
xmin = 0
xmax = 50
fig, ((ax,ax1),(ax2,ax3)) = plt.subplots (2,2,figsize = (12,9))
spam_df['len'].plot.hist(bins = 20, ax = ax, edgecolor = 'white', color = 'orange') #ax
spam_df['processed_len'].plot.hist(bins = 20, ax = ax1, edgecolor = 'white', color = 'orange') #ax1
ham_df['len'].plot.hist(bins = 20, ax = ax2, edgecolor = 'white', color = 'blue') #ax2
ham_df['processed_len'].plot.hist(bins = 20, ax = ax3, edgecolor = 'white', color = 'blue') #ax3
# 4 lines for ax
ax.tick_params(labelsize = 10) #increases the size(font) of x and y axis numbers
ax.set_xlabel('length of sentence', fontsize = 12) #name x axis
ax.set_ylabel('spam_frequency', fontsize = 12) #name y axis
ax.set_xlim([xmin,xmax]) #set limit which is xmin and xmax
#4 lines for ax1
ax1.tick_params(labelsize = 10)
ax1.set_xlabel('length of processed sentence', fontsize = 12)
ax1.set_ylabel('spam_frequency', fontsize = 12)
ax1.set_xlim([xmin,xmax])
#4 lines for ax2
ax2.tick_params(labelsize = 10)
ax2.set_xlabel('length of sentence', fontsize = 12)
ax2.set_ylabel('ham_frequency', fontsize = 12)
ax2.set_xlim([xmin,xmax])
#4 lines for ax3
ax3.tick_params(labelsize = 10)
ax3.set_xlabel('length of processed sentence', fontsize = 12)
ax3.set_ylabel('ham_frequency', fontsize = 12)
ax3.set_xlim([xmin,xmax])
def tokenize(text):
   ## exclude = set(string.punctuation)
    regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]') #remove punctuation
    text = regex.sub(' ', text)
    tokens = nltk.word_tokenize(text) # tokenize the text
    tokens = list(filter(lambda x: x.lower() not in stop_words, tokens)) # remove stop words
    tokens = [w.lower() for w in tokens if len(w) >=3] 
    tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
    return tokens
spam_df['tokens'] = spam_df['Message'].map(tokenize)
ham_df['tokens'] = ham_df['Message'].map(tokenize)
spam_df.head(3)
ham_df.head(3)
spam_words = []
for token in spam_df['tokens']:
    spam_words = spam_words + token #combine text in different columns in one list
ham_words = []
for token in ham_df['tokens']:
    ham_words += token
spam_count = Counter(spam_words).most_common(10)
ham_count = Counter(ham_words).most_common(10)
spam_count_df = pd.DataFrame(spam_count, columns = ['word', 'count'])
ham_count_df = pd.DataFrame(ham_count, columns = ['word', 'count'])
fig, (ax,ax1) = plt.subplots(1,2,figsize = (18, 6))
# for spam_count_df (spam words and there count)
sns.barplot(x = spam_count_df['word'], y = spam_count_df['count'], ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('word',fontsize = 15)
ax.tick_params(labelsize=15)
ax.set_title('spam top 10 words', fontsize = 15)
# for ham_count_df (ham words and there count)
sns.barplot(x = ham_count_df['word'], y = ham_count_df['count'], ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('word',fontsize = 15)
ax1.tick_params(labelsize=15)
ax1.set_title('ham top 10 words', fontsize = 15)
spam_words_str = ' '.join(spam_words) #joined all the spam words into a paragraph
ham_words_str = ' '.join(ham_words)
spam_word_cloud = WordCloud(width = 600, height = 400, background_color = 'white').generate(spam_words_str)
ham_word_cloud = WordCloud(width = 600, height = 400,background_color = 'white').generate(ham_words_str)
fig, (ax, ax2) = plt.subplots(1,2, figsize = (18,8))
ax.imshow(spam_word_cloud)
ax.axis('off')
ax.set_title('spam word cloud', fontsize = 20)
ax2.imshow(ham_word_cloud)
ax2.axis('off')
ax2.set_title('ham word cloud', fontsize = 20)
plt.show()
df.head()
df['tokens'] = df['Message'].map(tokenize)
def text_join(text):
    return " ".join(text)
df['text'] = df['tokens'].apply(text_join)
df.head()
#Tfidf - Term frequency * inverse document frequency - it return result in form of vector matrix
tv = TfidfVectorizer('english')
features = tv.fit_transform(df['text'])
target = df.Category.map({'ham':0, 'spam':1})
df.head()
print(features) #feature is X
print(target) #target is y
X = features
y = target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)
df['Category'].value_counts()
from collections import Counter #smote used because the dataset was imbalanced(counter helps in counting just)
from imblearn.combine import SMOTETomek
smt=SMOTETomek(0.80) #80% of the data 
X_train_smt,y_train_smt=smt.fit_sample(X_train,y_train)
print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_smt)))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train_smt, y_train_smt)
y_pred = dtc.predict(X_val)
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(dtc.score(X_val, y_val)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
# print(sns.heatmap(cm, annot=True))
print(cm)
print(classification_report(y_val, y_pred))
from scipy.stats import randint
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
#this param_grid intake all the argument of a LOGISTIC REGRESSION MODEL, since we are dealing with logistic regression only,
# and the arguments will be different for different models (*****IMPORTANT *******)
param_grid = {"criterion":['gini','entropy'],
              "max_depth": range(1,30), 
              "max_leaf_nodes": range(2,30), #max_leaf_node could not start from 1
              "min_samples_leaf": randint(1,20),
              "min_samples_split":range(1,30),
             "splitter":['best','random']} 
#just applying cross validation(as this is the parameter which is fed in gridsearchcv)((*****IMPORTANT *******))
cv=KFold(n_splits=5,shuffle = True, random_state = 42)
#here these are the parameters of gridsearchcv(*****IMPORTANT *******)
rsc = RandomizedSearchCV(dtc,param_grid,cv=cv,verbose=2,random_state = 42,n_jobs=-1)
rsc.fit(X_train_smt,y_train_smt)
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(rsc.best_params_)) 
print("Best score is {}".format(rsc.best_score_)) 
y_pred = rsc.predict(X_val)
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(rsc.score(X_val, y_val)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
# print(sns.heatmap(cm, annot=True))
print(cm)
print(classification_report(y_val, y_pred))
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train_smt, y_train_smt)
y_pred = xgb.predict(X_val)
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(xgb.score(X_val, y_val)))
#Importing random forest classifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
#creating a random forest instance   #while initializing no parameter
rfr =RandomForestClassifier(random_state=56)
#train the model
rfr.fit(X_train_smt, y_train_smt)
#score on training data
rfr.score(X_train_smt, y_train_smt)
#score on training data
rfr.score(X_val,y_val)
y_predict = rfr.predict(X_val)
RF = pd.DataFrame({'Actual':y_val, 'Predicted':y_predict})  
RF.head(5)
from sklearn.svm import SVC #imported svc
model = SVC(kernel='rbf',C=30,gamma='auto') #fit model on training data and chcek score of test data
model.fit(X_train_smt, y_train_smt)
model.score(X_val,y_val)
y_predict = model.predict(X_val)
y_predict
mnb = MultinomialNB() #fit model on training data and chcek score of test data
mnb.fit(X_train_smt, y_train_smt)
mnb.score(X_val,y_val)
#this param_grid intake all the argument of a LOGISTIC REGRESSION MODEL, since we are dealing with logistic regression only,
# and the arguments will be different for different models (*****IMPORTANT *******)
param_grid = {"alpha" : [0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,4.0,5.0]} 
#just applying cross validation(as this is the parameter which is fed in gridsearchcv)((*****IMPORTANT *******))
cv=KFold(n_splits=5,shuffle = True, random_state = 42)
#here these are the parameters of gridsearchcv(*****IMPORTANT *******)
rsc = RandomizedSearchCV(mnb,param_grid,cv=cv,verbose=2,random_state = 42,n_jobs=-1)
rsc.fit(X_train_smt,y_train_smt)
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(rsc.best_params_)) 
print("Best score is {}".format(rsc.best_score_)) 
mnb = MultinomialNB(alpha = 0.7) #fit model on training data and chcek score of test data
mnb.fit(X_train_smt, y_train_smt)
mnb.score(X_val,y_val)
y_predict = mnb.predict(X_val)
y_predict
print('Accuracy of Decision Tree Classifier on test set: {:.2f}'.format(dtc.score(X_val, y_val)))
print('Accuracy of XgBoost Classifier on test set: {:.2f}'.format(xgb.score(X_val, y_val)))
print('Accuracy of Random Forest Classifier on test set: {:.2f}'.format(rfr.score(X_val, y_val)))
print('Accuracy of SVC  on test set: {:.2f}'.format(model.score(X_val, y_val)))
print('Accuracy of Multinomial Naive Bayes on test set: {:.2f}'.format(mnb.score(X_val, y_val)))