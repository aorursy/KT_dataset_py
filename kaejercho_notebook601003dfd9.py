# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.interpolate import interp1d
import json
from scipy.optimize import brentq
import os
import re




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_input = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_input = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_input
len(train_input.loc[train_input["target"] == 1]) / len(train_input)
len(train_input.loc[train_input["target"] == 0]) / len(train_input)
X_train, X_dev, y_train, y_dev = train_test_split(train_input,train_input.target.values,test_size=0.3)
X_train
X_dev
#https://towardsdatascience.com/binary-classification-of-disaster-tweets-73efc6744712
#several
def cleaned(text):
    text = re.sub(r"\n","",text) #remove new line
#     text = re.sub(r"\d","",text) #Remove digits
    text = re.sub(r'[^\x00-\x7f]',r' ',text) # remove non-ascii
    text = re.sub(r'[^\w\s]',' ',text) #Remove punctuation
    text = re.sub(r'http\S+|www.\S+', ' ', text) #Remove http
    return text
#remove Emoji
import re
import emoji
    
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#Lemmatize
#https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
#https://stackoverflow.com/questions/47557563/lemmatization-of-all-pandas-cells
#https://medium.com/@yanweiliu/python%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E4%BA%94-%E4%BD%BF%E7%94%A8nltk%E9%80%B2%E8%A1%8C%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86-24fba36f3896
#https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def lemmatize_text(text):
#     wnl = WordNetLemmatizer()
    lemmatized = []
    for word, tag in pos_tag(word_tokenize(text)):
        if tag.startswith("NN"):
            lemmatized.append(lemmatizer.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemmatized.append(lemmatizer.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemmatized.append(lemmatizer.lemmatize(word, pos='a'))
        else:
            lemmatized.append(word)

    return ' '.join(lemmatized)
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
def preProcessData():
    global train_input,X_train,X_dev,test_input


    #lower train_input
    #https://stackoverflow.com/questions/22245171/how-to-lowercase-a-pandas-dataframe-string-column-if-it-has-missing-values
    train_input = train_input.assign(text_1 = train_input.text.str.lower())
    X_train = X_train.assign(text_1 = X_train.text.str.lower())
    X_dev = X_dev.assign(text_1 = X_dev.text.str.lower())
    test_input = test_input.assign(text_1 = test_input.text.str.lower())
    
    #remove html tag
    train_input = train_input.assign(text_1 =train_input.text_1.apply(lambda x : remove_html(x)))
    X_train = X_train.assign(text_1 =X_train.text_1.apply(lambda x : remove_html(x)))
    X_dev = X_dev.assign(text_1 =X_dev.text_1.apply(lambda x : remove_html(x)))
    test_input = test_input.assign(text_1 =test_input.text_1.apply(lambda x : remove_html(x)))
    
    #remove punctuation
    train_input = train_input.assign(text_1 = train_input.text_1.apply(cleaned))
    X_train = X_train.assign(text_1 =X_train.text_1.apply(cleaned))
    X_dev = X_dev.assign(text_1 =X_dev.text_1.apply(cleaned))
    test_input = test_input.assign(text_1 =test_input.text_1.apply(cleaned))

    #Strip the stop words
    #https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
    from nltk.corpus import stopwords 
    stop = stopwords.words('english')
    train_input = train_input.assign(text_1 = train_input.text_1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    X_train = X_train.assign(text_1= X_train.text_1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    X_dev = X_dev.assign(text_1 = X_dev.text_1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    test_input = test_input.assign(text_1 = test_input.text_1.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))


    train_input = train_input.assign(text_1 =train_input.text_1.apply(lambda x: remove_emoji(x)))
    X_train = X_train.assign(text_1=X_train.text_1.apply(lambda x: remove_emoji(x)))
    X_dev = X_dev.assign(text_1=X_dev.text_1.apply(lambda x: remove_emoji(x)))
    test_input = test_input.assign(text_1=test_input.text_1.apply(lambda x: remove_emoji(x)))


    train_input = train_input.assign(text_1 =train_input.text_1.apply(lemmatize_text))
    X_train = X_train.assign(text_1 =X_train.text_1.apply(lemmatize_text))
    X_dev = X_dev.assign(text_1 =X_dev.text_1.apply(lemmatize_text))
    test_input = test_input.assign(text_1 =test_input.text_1.apply(lemmatize_text))
preProcessData()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score

#use skilearn
#https://stackoverflow.com/questions/55994883/how-to-use-countvectorizer-to-test-new-data-after-doing-some-training

max_df = 0
def bow():
    global test_vectors,vectorizer,v_X_train,max_df
    test_df = [1,2,3,4,5,6,7,8,9,10,20,50,100,200]
    test_f1s = []
    for df in test_df:
        vectorizer = CountVectorizer(binary=True,min_df = df)
        v_X_train = vectorizer.fit_transform(X_train.text_1.to_numpy())
        clf = BernoulliNB()
        clf.fit(v_X_train, y_train)
        test_vectors = vectorizer.transform(X_dev.text_1.to_numpy())
        clf_predit = clf.predict(test_vectors)
        test_f1s.append(f1_score(y_dev, clf_predit))
    
    print("f1 scores:{}".format(test_f1s))

    max_df = test_df[test_f1s.index(max(test_f1s))]
    print("min_df with highest f1 score = {}".format(max_df))
    vectorizer = CountVectorizer(binary=True,min_df = max_df)
    v_X_train = vectorizer.fit_transform(X_train.text_1.to_numpy())
    test_vectors = vectorizer.transform(X_dev.text_1.to_numpy())
bow()
##use v_X_train to train, use test_vectors to test
#https://rushter.com/blog/scipy-sparse-matrices/
#https://stackoverflow.com/questions/4319014/iterating-through-a-scipy-sparse-vector-or-matrix
#https://nbviewer.jupyter.org/github/ethen8181/machine-learning/blob/master/text_classification/naive_bayes/naive_bayes.ipynb

# bernoulli_naive_bayes

def make_prediction(class_type,X_test,likelihood,predictions,priors):
    
    for idx, vector in enumerate(X_test):
        # stores the p(C|D) for each class
        posteriors = np.zeros(len(class_type))
        
        # compute p(C = k|D) for the document for all class
        # and return the predicted class with the maximum probability
        for c in class_type:
            
            #p(C = k)
            posterior = priors[c]
            likelihood_subset = likelihood[c, :]
            
            #p(D∣C = k)
            for idx2, feature in enumerate(vector.toarray()[0]):
                if feature:
                    prob = likelihood_subset[idx2]
                else:
                    prob = 1 - likelihood_subset[idx2]
                
                posterior *= prob

            posteriors[c] = posterior
        
        # compute the maximum p(C|D)
        predicted_class = class_type[np.argmax(posteriors)]
        predictions[idx] = predicted_class
    
    return predictions
    

def cal_likelihood(likelihood,X_train,y_train,class_type):
     #word likelihood p(w_t∣C)
    for idx, c in enumerate(class_type):
        subset = X_train[np.equal(y_train, c)] #only keep the document that class equals to current class
        likelihood[idx, :] = np.sum(subset, axis = 0) / subset.shape[0] #count/len of subset
        
    return likelihood

def bernoulli_naive_bayes(X_train,y_train,X_test):
    
    #prior p(c=k)
    total = X_train.shape[0] 
    priors = np.bincount(y_train) / total
    
    #obtain class types
    class_type = [0,1]
    
    #init likelihood
    likelihood = np.zeros((len(class_type), X_train.shape[1]))
    
    #cal likelihood
    likelihood = cal_likelihood(likelihood,X_train,y_train,class_type)
    
    #init predictions
    predictions = np.zeros(X_test.shape[0], dtype = np.int)
    
    #predict
    predictions = make_prediction(class_type,X_test,likelihood,predictions,priors)
    
    return predictions
    
    
    
def get_f1score(predictions,y_dev):
    predict_trues = np.count_nonzero(predictions)
    
    condition_1 = (np.array(predictions) == 1)
    condition_2 = (np.array(y_dev) == 1)
    real_trues = len(np.where(condition_1 & condition_2)[0])
    precision = float(real_trues)/predict_trues if predict_trues else 0
    y_trues = np.count_nonzero(y_dev)
    recall = float(real_trues)/y_trues if y_trues else 0
    
    f1score = (2*(precision*recall))/(precision+recall) if (precision+recall) else 0
    return f1score
    
from sklearn.metrics import f1_score
def makeBernoulli_prediction():
    predictions = bernoulli_naive_bayes(v_X_train, y_train, test_vectors)
    my_bayes_f1score = get_f1score(predictions,y_dev)
    print("Bernoulli F1:{}".format(my_bayes_f1score))
    print("Bernoulli F1 from sklearn:{}".format(f1_score(y_dev, predictions)))
makeBernoulli_prediction()
from sklearn.linear_model import LogisticRegression
def lr():
    global lr_clf
    lr_clf = LogisticRegression(random_state=0).fit(v_X_train, y_train)
    lr_predictions = lr_clf.predict(test_vectors)
    lr_f1score = get_f1score(lr_predictions,y_dev)
    print(lr_f1score)
lr()
pd.DataFrame(lr_clf.coef_[0], 
             vectorizer.get_feature_names(), 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
    
#https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1
pd.DataFrame((lr_clf.coef_[0]), 
         vectorizer.get_feature_names(), 
         columns=['coef'])\
        .sort_values(by='coef', ascending=False).head(10)
from sklearn.svm import LinearSVC
Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
svc_f1scores = []
svc_pds = []
def svc_clf():
    for c in Cs:
        svc_clf = LinearSVC(C=c).fit(v_X_train, y_train)
        svc_predictions = svc_clf.predict(test_vectors)
        svc_f1score = get_f1score(svc_predictions,y_dev)
        svc_f1scores.append(svc_f1score)
        print(pd.DataFrame((svc_clf.coef_[0]), 
                 vectorizer.get_feature_names(), 
                 columns=['coef'])\
                .sort_values(by='coef', ascending=False))
svc_clf()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,svc_f1scores)
# plt.plot(Cs,svc_f1scores)
svc_f1scores
from sklearn.svm import SVC
Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
nl_svc_f1scores = []
def nl_svc_clf():
    for c in Cs:
        nl_svc_clf = SVC(C=c,kernel='rbf').fit(v_X_train, y_train)
        nl_svc_predictions = nl_svc_clf.predict(test_vectors)
        nl_svc_f1score = get_f1score(nl_svc_predictions,y_dev)
        nl_svc_f1scores.append(nl_svc_f1score)
   
nl_svc_clf()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,nl_svc_f1scores)
print(nl_svc_f1scores)
#https://goodboychan.github.io/chans_jupyter/python/datacamp/natural_language_processing/2020/07/17/03-N-Gram-models.html
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
ng_vectorizer = None
ng_v_X_train = None
ng_test_vectors = None
def ng():
    global ng_vectorizer,ng_v_X_train,ng_test_vectors
    
    test_df = [1,2,3,4,5,6,7,8,9,10,20,50,100,200]
    test_f1s = []
    
    for df in test_df:
        ng_vectorizer = CountVectorizer(binary=True,min_df = df,ngram_range=(1, 2))
        ng_v_X_train = ng_vectorizer.fit_transform(X_train.text_1.to_numpy())
        clf = BernoulliNB()
        clf.fit(ng_v_X_train, y_train)
        test_vectors = ng_vectorizer.transform(X_dev.text_1.to_numpy())
        clf_predit = clf.predict(test_vectors)
        test_f1s.append(f1_score(y_dev, clf_predit))
    
    print("f1 scores:{}".format(test_f1s))

    max_df = test_df[test_f1s.index(max(test_f1s))]
    print("min_df with highest f1 score = {}".format(max_df))
    
    ng_vectorizer = CountVectorizer(binary=True,min_df = max_df,ngram_range=(1, 2))
    ng_v_X_train = ng_vectorizer.fit_transform(X_train.text_1.to_numpy())
    print("Vocs count:{}".format(len(ng_vectorizer.get_feature_names())))
    count = 0
    for voc in ng_vectorizer.get_feature_names():
        if len(voc.split(" ")) == 2:
            print(voc)
            count += 1
            if count >= 20:
                break
    ng_test_vectors = ng_vectorizer.transform(X_dev['text'].to_numpy())
ng()
#naive Bayes classifier
def ng_nb():
    ng_test_vectors = ng_vectorizer.transform(X_dev['text'].to_numpy())
    ng_predictions = bernoulli_naive_bayes(ng_v_X_train, y_train, ng_test_vectors)
    my_bayes_f1score = get_f1score(ng_predictions,y_dev)
    print("ng_nb:{}".format(my_bayes_f1score))
ng_nb()
#Logistic regression prediction
def ng_lr():
    ng_lr_clf = LogisticRegression(random_state=0).fit(ng_v_X_train, y_train)
    print(pd.DataFrame((ng_lr_clf.coef_[0]), 
             ng_vectorizer.get_feature_names(), 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False).head(10))
    ng_lr_predictions = ng_lr_clf.predict(ng_test_vectors)
    ng_lr_f1score = get_f1score(ng_lr_predictions,y_dev)
    print("ng_lr_f1score:{}".format(ng_lr_f1score))
ng_lr()
#Linear SVM prediction
Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
ng_svc_f1scores = []
ng_svc_pds = []
def ng_svc():
    for c in Cs:
        ng_svc_clf = LinearSVC(C=c).fit(ng_v_X_train, y_train)
        ng_svc_predictions = ng_svc_clf.predict(ng_test_vectors)
        ng_svc_f1score = get_f1score(ng_svc_predictions,y_dev)
        ng_svc_f1scores.append(ng_svc_f1score)
        print(pd.DataFrame((ng_svc_clf.coef_[0]), 
                 ng_vectorizer.get_feature_names(), 
                 columns=['coef'])\
                .sort_values(by='coef', ascending=False))
ng_svc()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,ng_svc_f1scores)
print(ng_svc_f1scores)
#Non-linear SVM prediction
Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
ng_nl_svc_f1scores = []
def ng_nl_svc():
    for c in Cs:
        ng_nl_svc_clf = SVC(C=c,kernel='rbf').fit(ng_v_X_train, y_train)
        ng_nl_svc_predictions = ng_nl_svc_clf.predict(ng_test_vectors)
        ng_nl_svc_f1score = get_f1score(ng_nl_svc_predictions,y_dev)
        ng_nl_svc_f1scores.append(ng_nl_svc_f1score)
ng_nl_svc()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,nl_svc_f1scores)
print(nl_svc_f1scores)
#url 
# train_input = train_input.assign(url=train_input['text_2_raw'].apply(lambda x:find_url(x)))
# X_train = X_train.assign(url=X_train['text_2_raw'].apply(lambda x:find_url(x)))
# X_dev = X_dev.assign(url=X_dev['text_2_raw'].apply(lambda x:find_url(x)))
#append data
train_input = train_input.assign(text_2_raw = train_input['text'].astype(str) + ' ' + train_input['location'].astype(str) + ' ' + train_input['keyword'].astype(str))
X_train = X_train.assign(text_2_raw = X_train['text'].astype(str) + ' ' + X_train['location'].astype(str) + ' ' + X_train['keyword'].astype(str))
X_dev = X_dev.assign(text_2_raw = X_dev['text'].astype(str) + ' ' + X_dev['location'].astype(str) + ' ' + X_dev['keyword'].astype(str))
test_input = test_input.assign(text_2_raw = test_input['text'].astype(str) + ' ' + test_input['location'].astype(str) + ' ' + test_input['keyword'].astype(str))
def preProcessData2():
    global train_input,X_train,X_dev,test_input
#     train_input = train_input.assign(url=train_input['text_2_raw'].apply(lambda x:find_url(x)))
#     X_train = X_train.assign(url=X_train['text_2_raw'].apply(lambda x:find_url(x)))
#     X_dev = X_dev.assign(url=X_dev['text_2_raw'].apply(lambda x:find_url(x)))

    # #lower train_input
    # #https://stackoverflow.com/questions/22245171/how-to-lowercase-a-pandas-dataframe-string-column-if-it-has-missing-values
    train_input = train_input.assign(text_2= train_input.text_2_raw.str.lower())
    X_train = X_train.assign(text_2 = X_train.text_2_raw.str.lower())
    X_dev = X_dev.assign(text_2 = X_dev.text_2_raw.str.lower())
    test_input = test_input.assign(text_2 = test_input.text_2_raw.str.lower())
    
    train_input = train_input.assign(tetext_2xt_1 =train_input.text_2.apply(lambda x : remove_html(x)))
    X_train = X_train.assign(text_2 =X_train.text_2.apply(lambda x : remove_html(x)))
    X_dev = X_dev.assign(text_2 =X_dev.text_2.apply(lambda x : remove_html(x)))
    test_input = test_input.assign(text_2 =test_input.text_2.apply(lambda x : remove_html(x)))

    #remove punctuation
    train_input = train_input.assign(text_2 = train_input.text_2.apply(cleaned))
    X_train = X_train.assign(text_2 =X_train.text_2.apply(cleaned))
    X_dev = X_dev.assign(text_2 =X_dev.text_2.apply(cleaned))
    test_input = test_input.assign(text_2 =test_input.text_2.apply(cleaned))

    #Strip the stop words
    #https://stackoverflow.com/questions/29523254/python-remove-stop-words-from-pandas-dataframe
    from nltk.corpus import stopwords 
    stop = stopwords.words('english')
    train_input = train_input.assign(text_2 = train_input.text_2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    X_train = X_train.assign(text_2= X_train.text_2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    X_dev = X_dev.assign(text_2 = X_dev.text_2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))
    test_input = test_input.assign(text_2 = test_input.text_2.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])))


    train_input = train_input.assign(text_2 =train_input.text_2.apply(lambda x: remove_emoji(x)))
    X_train = X_train.assign(text_2=X_train.text_2.apply(lambda x: remove_emoji(x)))
    X_dev = X_dev.assign(text_2=X_dev.text_2.apply(lambda x: remove_emoji(x)))
    test_input = test_input.assign(text_2=test_input.text_2.apply(lambda x: remove_emoji(x)))

    train_input = train_input.assign(text_2 =train_input.text_2.apply(lemmatize_text))
    X_train = X_train.assign(text_2 =X_train.text_2.apply(lemmatize_text))
    X_dev = X_dev.assign(text_2 =X_dev.text_2.apply(lemmatize_text))
    test_input = test_input.assign(text_2 =test_input.text_2.apply(lemmatize_text))
preProcessData2()

def bow2():
    global test_vectors,vectorizer,v_X_train,max_df
    test_df = [1,2,3,4,5,6,7,8,9,10,20,50,100,200]
    test_f1s = []
    for df in test_df:
        vectorizer = CountVectorizer(binary=True,min_df = df)
        v_X_train = vectorizer.fit_transform(X_train.text_2.to_numpy())
        clf = BernoulliNB()
        clf.fit(v_X_train, y_train)
        test_vectors = vectorizer.transform(X_dev.text_2.to_numpy())
        clf_predit = clf.predict(test_vectors)
        test_f1s.append(f1_score(y_dev, clf_predit))

#     max_df = test_df[test_f1s.index(max(test_f1s))]
    print("min_df with highest f1 score = {}".format(max_df))
    vectorizer = CountVectorizer(binary=True,min_df = max_df)
    v_X_train = vectorizer.fit_transform(X_train.text_2.to_numpy())
    test_vectors = vectorizer.transform(X_dev.text_2.to_numpy())
bow2()
#call function from f-h again
makeBernoulli_prediction()
lr()
pd.DataFrame(lr_clf.coef_[0], 
             vectorizer.get_feature_names(), 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)
    
#https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1
pd.DataFrame((lr_clf.coef_[0]), 
         vectorizer.get_feature_names(), 
         columns=['coef'])\
        .sort_values(by='coef', ascending=False).head(10)
svc_f1scores = []
svc_pds = []
svc_clf()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,svc_f1scores)
print(svc_f1scores)
svc_pds[0]
svc_pds[1]
svc_pds[2]
svc_pds[3]
svc_pds[4]
nl_svc_f1scores = []
nl_svc_clf()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,nl_svc_f1scores)
print(nl_svc_f1scores)
def ng():
    global ng_vectorizer,ng_v_X_train,ng_test_vectors
    ng_vectorizer = CountVectorizer(binary=True,min_df = 10,ngram_range=(1, 2))
    ng_v_X_train = ng_vectorizer.fit_transform(X_train['text_2'].to_numpy())
    ng_test_vectors = ng_vectorizer.transform(X_dev['text_2'].to_numpy())
ng()
ng_nb()
ng_lr()
ng_svc_f1scores = []
ng_svc_pds = []
ng_svc()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,ng_svc_f1scores)
print(ng_svc_f1scores)
ng_nl_svc_f1scores = []
ng_nl_svc()
plt.xlabel('C')
plt.ylabel('F1-Score')
plt.semilogx(Cs,ng_nl_svc_f1scores)
print(ng_nl_svc_f1scores)
vectorizer = CountVectorizer(binary=True,min_df = max_df)
v_X_train = vectorizer.fit_transform(train_input['text_1'].to_numpy())
test_vectors = vectorizer.transform(test_input['text_1'].to_numpy())
lr_clf = LogisticRegression(random_state=0).fit(v_X_train, train_input.target.values)
lr_predictions = lr_clf.predict(test_vectors)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = lr_predictions
sample_submission.to_csv("submission.csv", index=False)
