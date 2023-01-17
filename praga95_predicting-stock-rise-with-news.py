import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
data = pd.read_csv('../input/Combined_News_DJIA.csv')
data.head()
combined=data.copy()
combined['Combined']=combined.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
print("Length of train is",len(train))
print("Length of test is", len(test))
trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
train = combined[combined['Date'] < '2015-01-01']
test = combined[combined['Date'] > '2014-12-31']
non_decrease = train[train['Label']==1]
decrease = train[train['Label']==0]
print(len(non_decrease)/len(train))
non_decrease_test = test[test['Label']==1]
decrease_test = test[test['Label']==0]
print(len(non_decrease_test)/len(test))
import matplotlib.pyplot as plt
%matplotlib inline
from wordcloud import WordCloud,STOPWORDS
import re
import nltk
from nltk.corpus import stopwords
def to_words(content): ### function to clean the words
    letters_only = re.sub("[^a-zA-Z]", " ", content) ### get only letters
    words = letters_only.lower().split()             ### lowercase       
    stops = set(stopwords.words("english"))         ### remove stopwords such as 'the', 'and' etc.         
    meaningful_words = [w for w in words if not w in stops] ### get meaningful words
    return( " ".join( meaningful_words )) 
non_decrease_word=[]
decrease_word=[]
for each in non_decrease['Combined']:
    non_decrease_word.append(to_words(each))

for each in decrease['Combined']:
    decrease_word.append(to_words(each))
wordcloud1 = WordCloud(background_color='black',
                      width=3000,
                      height=2500
                     ).generate(decrease_word[1])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud1)
plt.axis('off')
plt.title("Words which indicate a fall in DJIA ")
plt.show()
wordcloud2 = WordCloud(background_color='green',
                      width=3000,
                      height=2500
                     ).generate(non_decrease_word[3])
plt.figure(1,figsize=(8,8))
plt.imshow(wordcloud2)
plt.axis('off')
plt.title("Words which indicate a rise/stable DJIA ")
plt.show()
example = train.iloc[3,3]
print(example)
example2 = example.lower()
print(example2)
example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)
pd.DataFrame([[x,example3.count(x)] for x in set(example3)], columns = ['Word', 'Count'])
basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

basictest = basicvectorizer.transform(testheadlines)
print(basictest.shape)
Classifiers = [
    LogisticRegression(C=0.1,solver='liblinear',max_iter=2000),
    KNeighborsClassifier(3),
    RandomForestClassifier(n_estimators=500,max_depth=9),
    ]
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(basictrain,train['Label'])
        pred = fit.predict(basictest)
        prob = fit.predict_proba(basictest)[:,1]
    except Exception:
        fit = classifier.fit(basictrain,train['Label'])
        pred = fit.predict(basictest)
        prob = fit.predict_proba(basictest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
train_text = []
test_text = []
for each in train['Combined']:
    train_text.append(to_words(each))

for each in test['Combined']:
    test_text.append(to_words(each))
train_features = tfidf.fit_transform(train_text)
test_features = tfidf.transform(test_text)
Classifiers = [
    LogisticRegression(C=0.1,solver='liblinear',max_iter=2000),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.25, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=500,max_depth=9),
    AdaBoostClassifier(),
    ]
dense_features=train_features.toarray()
dense_test= test_features.toarray()
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(train_features,train['Label'])
        pred = fit.predict(test_features)
        prob = fit.predict_proba(test_features)[:,1]
    except Exception:
        fit = classifier.fit(dense_features,train['Label'])
        pred = fit.predict(dense_test)
        prob = fit.predict_proba(dense_test)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    
df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df
advancedvectorizer = CountVectorizer(ngram_range=(2,2))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)

advancedtest = advancedvectorizer.transform(testheadlines)
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    except Exception:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
    
df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df
advancedvectorizer = CountVectorizer(ngram_range=(3,3))
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)
print(advancedtrain.shape)
advancedtest = advancedvectorizer.transform(testheadlines)
Accuracy=[]
Model=[]
for classifier in Classifiers:
    try:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    except Exception:
        fit = classifier.fit(advancedtrain,train['Label'])
        pred = fit.predict(advancedtest)
        prob = fit.predict_proba(advancedtest)[:,1]
    accuracy = accuracy_score(pred,test['Label'])
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    fpr, tpr, _ = roc_curve(test['Label'],prob)
df=pd.DataFrame(columns = ['Model', 'Accuracy'],index=np.arange(1, len(df)+1))
df.Model=Model
df.Accuracy=Accuracy
df