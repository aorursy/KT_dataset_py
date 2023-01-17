# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



file_path = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        file_path.append(os.path.join(dirname,filename))

# Any results you write to the current directory are saved as output.



train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test     = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

submission    = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")



train.shape , test.shape , submission.shape

train_col_name = train.columns

test_col_name = test.columns
train.dtypes
categorical_variables = ["keyword","location","text"]

continious_variables  = ["target"]
dq_categorical_var = train[categorical_variables].describe()

dq_categorical_var = dq_categorical_var.T

dq_categorical_var["total_record"] = train["id"].count()

dq_categorical_var["missing_value%"] = (1- dq_categorical_var["count"]/dq_categorical_var["total_record"])*100

dq_categorical_var.T
dq_cont_var= train[continious_variables].describe()

dq_cont_var = dq_cont_var.T

dq_cont_var["total_record"] = train["id"].count()

dq_cont_var["missing_value%"] = (1- dq_cont_var["count"]/dq_cont_var["total_record"])*100

dq_cont_var.T
col_name = train.columns

exclude_col = ["id","text"]

for c in col_name :

    if c not in exclude_col : 

        print(train[c].value_counts())
import matplotlib.pyplot as plt

from wordcloud import WordCloud , STOPWORDS



def word_cloud(data):

    stopword = set(STOPWORDS)

    comment_words = ' '

    for val in data:

        val = str(val)

        tokens = val.split()

        for i in range(len(tokens)) :

            tokens[i] = tokens[i].lower()

        for words in tokens:

            comment_words = comment_words + words + ' '

    

    wordcloud = WordCloud(width = 800 , height = 800 , background_color = 'white', stopwords = stopword , min_font_size = 9).generate(comment_words)

    plt.figure(figsize= (8,8), facecolor = None)

    plt.imshow(wordcloud)
word_cloud(train["text"])
import nltk

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



def word_freq(text):

    sent = []

    for txt in text :

        sent.append(txt)

    

    word_tokens = nltk.word_tokenize("".join([s for s in sent]))

    word_freq = nltk.FreqDist(word_tokens)

    print(len(word_tokens))

    sns.set(rc={'figure.figsize':(11.7,8.27)})

    sns.set_style('darkgrid')

    word_freq.plot(50)
word_freq(train.text)
from bs4 import BeautifulSoup

import re

import nltk

import emoji

import string
# emoticons

def load_dict_smileys():

    return {

        ":‑)":"smiley",

        ":-]":"smiley",

        ":-3":"smiley",

        ":->":"smiley",

        "8-)":"smiley",

        ":-}":"smiley",

        ":)":"smiley",

        ":]":"smiley",

        ":3":"smiley",

        ":>":"smiley",

        "8)":"smiley",

        ":}":"smiley",

        ":o)":"smiley",

        ":c)":"smiley",

        ":^)":"smiley",

        "=]":"smiley",

        "=)":"smiley",

        ":-))":"smiley",

        ":‑D":"smiley",

        "8‑D":"smiley",

        "x‑D":"smiley",

        "X‑D":"smiley",

        ":D":"smiley",

        "8D":"smiley",

        "xD":"smiley",

        "XD":"smiley",

        ":‑(":"sad",

        ":‑c":"sad",

        ":‑<":"sad",

        ":‑[":"sad",

        ":(":"sad",

        ":c":"sad",

        ":<":"sad",

        ":[":"sad",

        ":-||":"sad",

        ">:[":"sad",

        ":{":"sad",

        ":@":"sad",

        ">:(":"sad",

        ":'‑(":"sad",

        ":'(":"sad",

        ":‑P":"playful",

        "X‑P":"playful",

        "x‑p":"playful",

        ":‑p":"playful",

        ":‑Þ":"playful",

        ":‑þ":"playful",

        ":‑b":"playful",

        ":P":"playful",

        "XP":"playful",

        "xp":"playful",

        ":p":"playful",

        ":Þ":"playful",

        ":þ":"playful",

        ":b":"playful",

        "<3":"love"

        }



# self defined contractions

def load_dict_contractions():

    

    return {

        "ain't":"is not",

        "amn't":"am not",

        "aren't":"are not",

        "can't":"cannot",

        "'cause":"because",

        "couldn't":"could not",

        "couldn't've":"could not have",

        "could've":"could have",

        "daren't":"dare not",

        "daresn't":"dare not",

        "dasn't":"dare not",

        "didn't":"did not",

        "doesn't":"does not",

        "don't":"do not",

        "e'er":"ever",

        "em":"them",

        "everyone's":"everyone is",

        "finna":"fixing to",

        "gimme":"give me",

        "gonna":"going to",

        "gon't":"go not",

        "gotta":"got to",

        "hadn't":"had not",

        "hasn't":"has not",

        "haven't":"have not",

        "he'd":"he would",

        "he'll":"he will",

        "he's":"he is",

        "he've":"he have",

        "how'd":"how would",

        "how'll":"how will",

        "how're":"how are",

        "how's":"how is",

        "I'd":"I would",

        "I'll":"I will",

        "I'm":"I am",

        "I'm'a":"I am about to",

        "I'm'o":"I am going to",

        "isn't":"is not",

        "it'd":"it would",

        "it'll":"it will",

        "it's":"it is",

        "I've":"I have",

        "kinda":"kind of",

        "let's":"let us",

        "mayn't":"may not",

        "may've":"may have",

        "mightn't":"might not",

        "might've":"might have",

        "mustn't":"must not",

        "mustn't've":"must not have",

        "must've":"must have",

        "needn't":"need not",

        "ne'er":"never",

        "o'":"of",

        "o'er":"over",

        "ol'":"old",

        "oughtn't":"ought not",

        "shalln't":"shall not",

        "shan't":"shall not",

        "she'd":"she would",

        "she'll":"she will",

        "she's":"she is",

        "shouldn't":"should not",

        "shouldn't've":"should not have",

        "should've":"should have",

        "somebody's":"somebody is",

        "someone's":"someone is",

        "something's":"something is",

        "that'd":"that would",

        "that'll":"that will",

        "that're":"that are",

        "that's":"that is",

        "there'd":"there would",

        "there'll":"there will",

        "there're":"there are",

        "there's":"there is",

        "these're":"these are",

        "they'd":"they would",

        "they'll":"they will",

        "they're":"they are",

        "they've":"they have",

        "this's":"this is",

        "those're":"those are",

        "'tis":"it is",

        "'twas":"it was",

        "wanna":"want to",

        "wasn't":"was not",

        "we'd":"we would",

        "we'd've":"we would have",

        "we'll":"we will",

        "we're":"we are",

        "weren't":"were not",

        "we've":"we have",

        "what'd":"what did",

        "what'll":"what will",

        "what're":"what are",

        "what's":"what is",

        "what've":"what have",

        "when's":"when is",

        "where'd":"where did",

        "where're":"where are",

        "where's":"where is",

        "where've":"where have",

        "which's":"which is",

        "who'd":"who would",

        "who'd've":"who would have",

        "who'll":"who will",

        "who're":"who are",

        "who's":"who is",

        "who've":"who have",

        "why'd":"why did",

        "why're":"why are",

        "why's":"why is",

        "won't":"will not",

        "wouldn't":"would not",

        "would've":"would have",

        "y'all":"you all",

        "you'd":"you would",

        "you'll":"you will",

        "you're":"you are",

        "you've":"you have",

        "Whatcha":"What are you",

        "luv":"love",

        "sux":"sucks"

        }
def clean_text(txt):

    #remove html tag

    txt = BeautifulSoup(txt).get_text()

    #special cases not handled before

    txt = txt.replace('\x92',"''")

    #removal of hash tags

    txt = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", txt).split())

    #removal url 

    txt = ' '.join(re.sub("(\w+:\/\/\S+)", " ", txt).split())

    #replace contractions

    contractions = load_dict_contractions()

    txt = txt.strip()

    words = txt.split()

    reformed = [contractions[word] if word in contractions else word for word in words]

    txt = " ".join(reformed)    

    #remove special characters

    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))

    txt = filter(None, [pattern.sub(' ', token) for token in txt])

    txt = ''.join(txt)

    #convert to lower case

    txt = txt.lower()

    #handle emoticons

    SMILEY = load_dict_smileys()  

    txt = txt.split()

    reformed =  [SMILEY[word] if word in SMILEY else word for word in txt]

    txt = " ".join(reformed)

    #handle emojis

    txt = emoji.demojize(txt)

    txt = txt.replace(":"," ")

    return txt
train = np.array(train)

test  = np.array(test)

#train

for i in range(len(train)):

    #print(train[i][3])

    train[i][3] = clean_text(train[i][3])

train = pd.DataFrame(train)

train.columns = train_col_name

#test

for i in range(len(test)):

    #print(train[i][3])

    test[i][3] = clean_text(test[i][3])

test = pd.DataFrame(test)

test.columns = test_col_name
word_cloud(train["text"])
word_freq(train.text)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



def bow_extractor(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)

    features = vectorizer.fit_transform(corpus)

    return vectorizer, features



def tfidf_extractor(corpus, ngram_range=(1,1)):

    vectorizer = TfidfVectorizer(norm='l2',smooth_idf=True,use_idf=True,ngram_range=ngram_range ,max_features = 200,min_df = 5,stop_words='english')

    features = vectorizer.fit_transform(corpus)

    return vectorizer, features
#bow_vectorizer , bow_train_features = bow_extractor(train.text,(1,1))

#bow_test_features = bow_vectorizer.transform(test.text)

#tfidf_vectorizer , tfidf_train_features = tfidf_extractor(train.text, (1,1))

#tfidf_test_vectorizer = tfidf_vectorizer.transform(test.text)
from sklearn.model_selection import train_test_split

from sklearn import metrics

import pickle



def modelling_data(data, labels , split_size):

    train_X, test_X, train_Y , test_Y = train_test_split(data,labels, test_size = split_size,random_state = 42)

    return train_X, test_X, train_Y , test_Y



def model_accuracy(actuals , pred,model_result,flag):

    if flag == 0 :

        #print("Accuracy:", np.round(metrics.accuracy_score(actuals, pred),2))

        model_result[2] = np.round(metrics.accuracy_score(actuals, pred),2)

        #print("Precision:", np.round(metrics.precision_score(actuals,pred),2))

        model_result[3] = np.round(metrics.precision_score(actuals,pred),2)

        #print("Recall:",np.round(metrics.recall_score(actuals,pred),2))

        model_result[4] = np.round(metrics.recall_score(actuals,pred),2)

        #print("F1 Score",np.round(metrics.f1_score(actuals, pred),2))

        model_result[5] = np.round(metrics.f1_score(actuals, pred),2)

        #print("Confusion_Matrix:", metrics.confusion_matrix(actuals,pred))

        flag = flag  + 1

    elif flag  == 1 :

        #print("Accuracy:", np.round(metrics.accuracy_score(actuals, pred),2))

        model_result[6] = np.round(metrics.accuracy_score(actuals, pred),2)

        #print("Precision:", np.round(metrics.precision_score(actuals,pred),2))

        model_result[7] = np.round(metrics.precision_score(actuals,pred),2)

        #print("Recall:",np.round(metrics.recall_score(actuals,pred),2))

        model_result[8] = np.round(metrics.recall_score(actuals,pred),2)

        #print("F1 Score",np.round(metrics.f1_score(actuals, pred),2))

        model_result[9] = np.round(metrics.f1_score(actuals, pred),2)

        #print("Confusion_Matrix:", metrics.confusion_matrix(actuals,pred))

        flag = flag  + 1

        

    return model_result , flag

    

def model_score(model ,trainX, trainY, testX, testY,model_name,embedding_type):

    flag = 0

    model_result = [None]*10

    model_result[0] = model_name

    model_result[1] = embedding_type

    print(model)

    model.fit(trainX, trainY)

    train_pred = model.predict(trainX)

    test_pred  = model.predict(testX)    

    #print("Training_Model_Performance:") 

    model_result , flag = model_accuracy(trainY,train_pred,model_result,flag)

    #print("Test_Model_Performance:") 

    model_result , flag = model_accuracy(testY,test_pred,model_result,flag)  

    return model_result, pd.DataFrame(train_pred), pd.DataFrame(test_pred),model



def model_summary(model_output,column_names):

    summary = pd.DataFrame(model_output)

    summary.columns = column_names

    return summary



def decile_analysis():

    pass



def save_ml_model(file_name , model):

    with open(file_name,'wb') as file_name:

        pickle.dump(model,file_name)
trainX, testX, trainY, testY = modelling_data(train.text, train.target , .40)

pd.DataFrame(trainX).to_csv('/kaggle/working/trainX.csv')

pd.DataFrame(testX).to_csv('/kaggle/working/testX.csv')

pd.DataFrame(trainY).to_csv("/kaggle/working/trainY.csv")

pd.DataFrame(testY).to_csv("/kaggle/working/testY.csv")



#store model results

model_output = []

columns = ['Model_Name','Embeddings_Type','Train_Accuracy','Train_Precision','Train_Recall','Train_F1','Test_Accuracy','Test_Precision','Test_Recal','Test_F1']
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
bow_vectorizer , bow_train_features = bow_extractor(trainX,(1,1))

bow_test_featurs = bow_vectorizer.transform(testX)

save_ml_model('bow_vectorizer',bow_vectorizer)
from sklearn.naive_bayes import MultinomialNB



nb_model = MultinomialNB()

model_result, nb_bow_train_pred, nb_bow_test_pred,trained_model = model_score(nb_model,bow_train_features,trainY.astype("int"), bow_test_featurs,testY.astype("int"),'nb_bow_model','bag_of words')

model_output.append(model_result)

nb_bow_train_pred.to_csv('/kaggle/working/nb_bow_train_pred.csv')

nb_bow_test_pred.to_csv('/kaggle/working/nb_bow_test_pred.csv')

save_ml_model('nb_bow_model',trained_model)



sgd_model = SGDClassifier(loss='hinge',penalty = 'l2')

model_result,sgd_bow_train_pred, sgd_bow_test_pred,trained_model  = model_score(sgd_model,bow_train_features,trainY.astype("int"), bow_test_featurs,testY.astype("int"),'sgd_bow_model','bag_of words')

model_output.append(model_result)

sgd_bow_train_pred.to_csv('/kaggle/working/sgd_bow_train_pred.csv')

sgd_bow_test_pred.to_csv('/kaggle/working/sgd_bow_test_pred.csv')

save_ml_model('sgd_bow',trained_model)



svc_model = SVC()

model_result,svm_bow_train_pred,svm_bow_test_pred,trained_model = model_score(svc_model,bow_train_features,trainY.astype("int"), bow_test_featurs,testY.astype("int"),'svc_bow_model','bag_of words')

model_output.append(model_result)

svm_bow_train_pred.to_csv('/kaggle/working/svm_bow_train_pred.csv')

svm_bow_test_pred.to_csv('/kaggle/working/svm_bow_test_pred.csv')

save_ml_model('svm_bow',trained_model)

#nagram = 1

tfidf_vectorizer_1 , tfidf_train_features_1 = tfidf_extractor(trainX, (1,1))

tfidf_test_featurs_1 = tfidf_vectorizer_1.transform(testX)

save_ml_model('tfidf_vectorizer',tfidf_vectorizer_1)



#ngram = 2

tfidf_vectorizer_2 , tfidf_train_features_2 = tfidf_extractor(trainX, (1,2))

tfidf_test_featurs_2 = tfidf_vectorizer_2.transform(testX)

save_ml_model('tfidf_vectorizer_ngram2',tfidf_vectorizer_2)

#ngram = 1

nb_model = MultinomialNB()

model_result,nb_tfidf_ngm1_train_pred,nb_tfidf_ngm1_test_pred,trained_model = model_score(nb_model,tfidf_train_features_1,trainY.astype("int"), tfidf_test_featurs_1,testY.astype("int"),'nb_tfidf_model','tfidf')

model_output.append(model_result)

nb_tfidf_ngm1_train_pred.to_csv('/kaggle/working/nb_tfidf_ngm1_train_pred.csv')

nb_tfidf_ngm1_test_pred.to_csv('/kaggle/working/nb_tfidf_ngm1_test_pred.csv')

save_ml_model('nb_tfidf_ngm1',trained_model)





#ngram = 2

nb_model = MultinomialNB()

model_result,nb_tfidf_ngm2_train_pred,nb_tfidf_ngm2_test_pred,trained_model = model_score(nb_model,tfidf_train_features_2,trainY.astype("int"), tfidf_test_featurs_2,testY.astype("int"),'nb_tfidf_model','tfidf_ngram2')

model_output.append(model_result)

nb_tfidf_ngm2_train_pred.to_csv('/kaggle/working/nb_tfidf_ngm2_train_pred.csv')

nb_tfidf_ngm2_test_pred.to_csv('/kaggle/working/nb_tfidf_ngm2_test_pred.csv')

save_ml_model('nb_tfidf_ngm2',trained_model)


#ngram =1

sgd_model = SGDClassifier(loss='hinge',penalty = 'l2')

model_result,sgd_tfidf_ngm1_train_pred,sgd_tfidf_ngm1_test_pred,trained_model = model_score(sgd_model,tfidf_train_features_1,trainY.astype("int"), tfidf_test_featurs_1,testY.astype("int"),'sgd_tfidf_model','tfidf')

model_output.append(model_result)

sgd_tfidf_ngm1_train_pred.to_csv('/kaggle/working/sgd_tfidf_ngm1_train_pred.csv')

sgd_tfidf_ngm1_test_pred.to_csv('/kaggle/working/sgd_tfidf_ngm1_test_pred.csv')

save_ml_model('sgd_tfidf_ngm1',trained_model)



#ngram= 2

sgd_model = SGDClassifier(loss='hinge',penalty = 'l2')

model_result,sgd_tfidf_ngm2_train_pred,sgd_tfidf_ngm2_test_pred,trained_model = model_score(sgd_model,tfidf_train_features_2,trainY.astype("int"), tfidf_test_featurs_2,testY.astype("int"),'sgd_tfidf_model','tfidf_ngram2')

model_output.append(model_result)

sgd_tfidf_ngm2_train_pred.to_csv('/kaggle/working/sgd_tfidf_ngm2_train_pred.csv')

sgd_tfidf_ngm2_test_pred.to_csv('/kaggle/working/sgd_tfidf_ngm2_test_pred.csv')

save_ml_model('sgd_tfidf_ngm2',trained_model)





#hyper_param tunning with ngram = 1

params = {

    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],

    "alpha" : [0.0001, 0.001, 0.01, 0.1],

    "penalty" : ["l2", "l1", "none"],

}



model = SGDClassifier(max_iter=1000)

sgd_opt_tfidf_clf = GridSearchCV(model, param_grid=params , cv=10, scoring='f1', verbose=0, n_jobs=-1)

sgd_opt_tfidf_clf.fit(tfidf_train_features_1,trainY.astype("int"))

sgd_opt_tfidf_clf_best = sgd_opt_tfidf_clf.best_estimator_

model_result,sgd_tfidf_opt_train_pred,sgd_tfidf_opt_test_pred,trained_model = model_score(sgd_opt_tfidf_clf_best,tfidf_train_features_1,trainY.astype("int"), tfidf_test_featurs_1,testY.astype("int"),'sgd_tfidf_opt_model','tfidf')

model_output.append(model_result)

sgd_tfidf_opt_train_pred.to_csv('/kaggle/working/sgd_tfidf_opt_train_pred.csv')

sgd_tfidf_opt_test_pred.to_csv('/kaggle/working/sgd_tfidf_opt_test_pred.csv')

save_ml_model('sgd_tfidf_opt',trained_model)
svc_model = SVC()

model_result,svm_tfidf_train_pred,svm_tfidf_test_pred,trained_model = model_score(svc_model,tfidf_train_features_1,trainY.astype("int"), tfidf_test_featurs_1,testY.astype("int"),'svc_tfidf_model','tfidf')

model_output.append(model_result)

svm_tfidf_train_pred.to_csv('/kaggle/working/svm_tfidf_train_pred.csv')

svm_tfidf_test_pred.to_csv('/kaggle/working/svm_tfidf_test_pred.csv')

save_ml_model('svm_tfidf',trained_model)





#hyper_param tunning with ngram = 1



tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],

                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},

                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],

                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},

                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}

                   ]



model = SVC()

svm_opt_tfidf_clf = GridSearchCV(model, param_grid=tuned_parameters , cv=10, scoring='f1', verbose=0, n_jobs=-1)

svm_opt_tfidf_clf.fit(tfidf_train_features_1,trainY.astype("int"))

svm_opt_tfidf_clf_best_model = svm_opt_tfidf_clf.best_estimator_

model_result,svm_tfidf_opt_train_pred,svm_tfidf_opt_test_pred,trained_model = model_score(svm_opt_tfidf_clf_best_model,tfidf_train_features_1,trainY.astype("int"), tfidf_test_featurs_1,testY.astype("int"),'svm_tfidf_opt_model','tfidf')

model_output.append(model_result)

svm_tfidf_opt_train_pred.to_csv('/kaggle/working/svm_tfidf_opt_train_pred.csv')

svm_tfidf_opt_test_pred.to_csv('/kaggle/working/svm_tfidf_opt_test_pred.csv')

save_ml_model('svm_tfidf_opt',trained_model)
summary = model_summary(model_output,columns)

summary.to_csv('/kaggle/working/model_summary.csv')

summary
training_data_X = pd.read_csv('/kaggle/input/modelling-dataset/trainX.csv') 

training_data_Y = pd.read_csv('/kaggle/input/modelling-dataset/trainY.csv') 

training_data = pd.concat([training_data_X[['Unnamed: 0','text']], training_data_Y['target']],axis  = 1)

training_data.columns = ["Index" ,"text","Target"]



test_data_X     = pd.read_csv('/kaggle/input/modelling-dataset/testX.csv')

test_data_y     = pd.read_csv('/kaggle/input/modelling-dataset/testY.csv')

test_data = pd.concat([test_data_X[['Unnamed: 0','text']], test_data_y['target']],axis  = 1)

test_data.columns = ["Index" ,"text","Target"]

org_data      = pd.concat([training_data,test_data],axis =0)



#nb model

nb_bow_train = pd.read_csv('/kaggle/input/modelling-dataset/nb_bow_train_pred.csv')

nb_bow_train.index = training_data.Index

nb_bow_test =  pd.read_csv('/kaggle/input/modelling-dataset/nb_bow_test_pred.csv')

nb_bow_test.index = test_data.Index

nb_bow_merge = pd.DataFrame(pd.concat([nb_bow_train['0'],nb_bow_test['0']],axis  =0))

nb_bow_merge.columns = ['nb_bow']



nb_tfidf_ngm1_train = pd.read_csv('/kaggle/input/modelling-dataset/nb_tfidf_ngm1_train_pred.csv')

nb_tfidf_ngm1_train.index = training_data.Index

nb_tfidf_ngm1_test = pd.read_csv('/kaggle/input/modelling-dataset/nb_tfidf_ngm1_test_pred.csv')

nb_tfidf_ngm1_test.index = test_data.Index

nb_tfidf_ngm1_merge = pd.DataFrame(pd.concat([nb_tfidf_ngm1_train['0'],nb_tfidf_ngm1_test['0']],axis  =0))

nb_tfidf_ngm1_merge.columns = ['nb_tfidf_ngm1']



nb_tfidf_ngm2_train = pd.read_csv('/kaggle/input/modelling-dataset/nb_tfidf_ngm2_train_pred.csv')

nb_tfidf_ngm2_train.index = training_data.Index

nb_tfidf_ngm2_test = pd.read_csv('/kaggle/input/modelling-dataset/nb_tfidf_ngm2_test_pred.csv')

nb_tfidf_ngm2_test.index = test_data.Index

nb_tfidf_ngm2_merge = pd.DataFrame(pd.concat([nb_tfidf_ngm2_train['0'],nb_tfidf_ngm2_test['0']],axis  =0))

nb_tfidf_ngm2_merge.columns = ['nb_tfidf_ngm2']



#sgd model

sgd_bow_train = pd.read_csv('/kaggle/input/modelling-dataset/sgd_bow_train_pred.csv')

sgd_bow_train.index = training_data.Index

sgd_bow_test =  pd.read_csv('/kaggle/input/modelling-dataset/sgd_bow_test_pred.csv')

sgd_bow_test.index = test_data.Index

sdg_bow_merge = pd.DataFrame(pd.concat([sgd_bow_train['0'],sgd_bow_test['0']],axis  =0))

sdg_bow_merge.columns = ['sdg_bow']



sgd_tfidf_ngm1_train = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_ngm1_train_pred.csv')

sgd_tfidf_ngm1_train.index= training_data.Index

sgd_tfidf_ngm1_test = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_ngm1_test_pred.csv')

sgd_tfidf_ngm1_test.index = test_data.Index

sgd_tfidf_ngm1_merge = pd.DataFrame(pd.concat([sgd_tfidf_ngm1_train['0'],sgd_tfidf_ngm1_test['0']],axis  =0))

sgd_tfidf_ngm1_merge.columns = ['sgd_tfidf_ngm1']



sgd_tfidf_ngm2_train = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_ngm2_train_pred.csv')

sgd_tfidf_ngm2_train.index= training_data.Index

sgd_tfidf_ngm2_test = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_ngm2_test_pred.csv')

sgd_tfidf_ngm2_test.index = test_data.Index

sgd_tfidf_ngm2_merge = pd.DataFrame(pd.concat([sgd_tfidf_ngm2_train['0'],sgd_tfidf_ngm2_test['0']],axis  =0))

sgd_tfidf_ngm2_merge.columns = ['sgd_tfidf_ngm2']



sgd_tfidf_opt_train = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_opt_train_pred.csv')

sgd_tfidf_opt_train.index = training_data.Index

sgd_tfidf_opt_test = pd.read_csv('/kaggle/input/modelling-dataset/sgd_tfidf_opt_test_pred (1).csv')

sgd_tfidf_opt_test.index = test_data.Index

sgd_tfidf_opt_merge = pd.DataFrame(pd.concat([sgd_tfidf_opt_train['0'],sgd_tfidf_opt_test['0']],axis  =0))

sgd_tfidf_opt_merge.columns = ['sgd_tfidf_opt']



#svm model

svm_bow_train = pd.read_csv('/kaggle/input/modelling-dataset/svm_bow_train_pred.csv')

svm_bow_train.index = training_data.Index

svm_bow_test =  pd.read_csv('/kaggle/input/modelling-dataset/svm_bow_test_pred.csv')

svm_bow_test.index= test_data.Index

svm_bow_merge = pd.DataFrame(pd.concat([svm_bow_train['0'],svm_bow_test['0']],axis  =0))

svm_bow_merge.columns = ['svm_bow']



svm_tfidf_ngm1_train = pd.read_csv('/kaggle/input/modelling-dataset/svm_tfidf_train_pred.csv')

svm_tfidf_ngm1_train.index = training_data.Index

svm_tfidf_ngm1_test = pd.read_csv('/kaggle/input/modelling-dataset/svm_tfidf_test_pred.csv')

svm_tfidf_ngm1_test.index= test_data.Index

svm_tfidf_ngm1_merge = pd.DataFrame(pd.concat([svm_tfidf_ngm1_train['0'],svm_tfidf_ngm1_test['0']],axis  =0))

svm_tfidf_ngm1_merge.columns = ['svm_tfidf_ngm1']





svm_tfidf_opt_train = pd.read_csv('/kaggle/input/modelling-dataset/svm_tfidf_opt_train_pred.csv')

svm_tfidf_opt_train.index = training_data.Index

svm_tfidf_opt_test = pd.read_csv('/kaggle/input/modelling-dataset/svm_tfidf_opt_test_pred.csv')

svm_tfidf_opt_test.index = test_data.Index

svm_tfidf_opt_merge = pd.DataFrame(pd.concat([svm_tfidf_opt_train['0'],svm_tfidf_opt_test['0']],axis  =0))

svm_tfidf_opt_merge.columns = ['svm_tfidf_opt']





#create training dataset

ensemble_modelling_dataset = pd.concat([nb_bow_merge,nb_tfidf_ngm1_merge,nb_tfidf_ngm2_merge,sdg_bow_merge,sgd_tfidf_ngm1_merge,sgd_tfidf_ngm2_merge,sgd_tfidf_opt_merge,svm_bow_merge,svm_tfidf_ngm1_merge,svm_tfidf_opt_merge],axis = 1)

ensemble_modelling_dataset = pd.merge(ensemble_modelling_dataset, org_data[["Target","Index"]] , left_index = True , right_on ='Index',how ='inner')

ensemble_modelling_dataset.drop("Index",axis = 1 , inplace = True)

ensemble_modelling_dataset.to_csv('/kaggle/working/ensemble_modelling_dataset.csv')

ensemble_modelling_dataset.head()
import xgboost as xgb

from xgboost.sklearn import XGBClassifier



trainX, testX, trainY, testY = modelling_data(ensemble_modelling_dataset[[ c for c in ensemble_modelling_dataset.columns if c != 'Target']], ensemble_modelling_dataset.Target , .40)

def model_fit(alg, trainX,trainY,testX, testY, usecv = True , cv_folds = 10 , stopping_rounds = 50):

    if usecv == True:

        xgb_param = alg.get_xgb_params()

        xgb_train = xgb.DMatrix(trainX.values , label = trainY.values)

        cvresult = xgb.cv(xgb_param,xgb_train, 

                          num_boost_round = alg.get_params()['n_estimators'],

                          nfold = cv_folds,

                         metrics = 'auc',

                         early_stopping_rounds = stopping_rounds)

        

        alg.set_params(n_estimators = cvresult.shape[0])

        #fit

        alg.fit(trainX, trainY)

        #predict

        dtrain_pred = alg.predict(trainX)

        dtest_pred = alg.predict(testX)

        print("train>>>>>>")        

        model_accuracy(trainY,dtrain_pred)

        print("test>>>>>>")

        model_accuracy(testY,dtest_pred)

        

        return cvresult

 

def model_accuracy(actuals , pred):

    print("Accuracy:", np.round(metrics.accuracy_score(actuals, pred),2))

    print("Precision:", np.round(metrics.precision_score(actuals,pred),2))

    print("Recall:",np.round(metrics.recall_score(actuals,pred),2))

    print("F1 Score",np.round(metrics.f1_score(actuals, pred),2))

    print("Confusion_Matrix:", metrics.confusion_matrix(actuals,pred))

xgb1 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=5,

 min_child_weight=1,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

cv_results = model_fit(xgb1,trainX, trainY, testX, testY)

save_ml_model('xgboost',xgb1)
from sklearn.model_selection import GridSearchCV



param_test1 = {

    'max_depth': range(2,7),

    'min_child_weight' : range(2,6)

}

gridsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate = .1, n_esitmators = 140 ,max_depth = 12 ,min_child_weight =4,

                                                    gamma =0,subsample = .8 , colsample_bytree = .8, objective ='binary:logistic',

                                                    ntread =4,scale_pos_weight = 1,seed = 27), 

                          param_grid = param_test1,scoring ='f1',n_jobs =4,iid = False , cv = 10)



gridsearch1.fit(trainX,trainY)



gridsearch1.best_params_ , gridsearch1.best_index_, gridsearch1.best_score_
xgb2 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=2,

 min_child_weight=2,

 gamma=0,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

cv_results = model_fit(xgb2,trainX, trainY, testX, testY)
param_test3 = {

 'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=2,

 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test3, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch3.fit(trainX,trainY)
gsearch3.best_params_ , gsearch3.best_index_, gsearch3.best_score_
xgb3 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=2,

 min_child_weight=2,

 gamma=0.1,

 subsample=0.8,

 colsample_bytree=0.8,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

cv_results = model_fit(xgb3,trainX, trainY, testX, testY)
param_test4 = {

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

}

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=2,

 min_child_weight=2, gamma=0.1, subsample=0.8, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test4, scoring='f1',n_jobs=4,iid=False, cv=5)

gsearch4.fit(trainX,trainY)
gsearch4.best_params_ , gsearch4.best_index_, gsearch4.best_score_
xgb4 = XGBClassifier(

 learning_rate =0.1,

 n_estimators=1000,

 max_depth=2,

 min_child_weight=2,

 gamma=0.1,

 subsample=0.8,

 colsample_bytree=0.6,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 seed=27)

cv_results = model_fit(xgb4,trainX, trainY, testX, testY)
param_test6 = {

    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 0.05]

}

gsearch6 = GridSearchCV(estimator = XGBClassifier(learning_rate = .1 ,n_estimators = 177 , max_depth = 2,

                                                min_child_weight =2 , gamma = 0.1 , subsample = .8, colsample_bytree = .6,

                                                objective = 'binary:logistic', nthread = 4 , scale_pos_weight =1 , seed =27),

                        param_grid = param_test6, scoring = 'f1',n_jobs = 4 , iid = False , cv = 5

                        )

gsearch6.fit(trainX, trainY)
gsearch6.best_params_ , gsearch6.best_index_, gsearch6.best_score_
xgb5 = XGBClassifier(

 learning_rate =0.01,

 n_estimators=5000,

 max_depth=2,

 min_child_weight=2,

 gamma=0.1,

 subsample=0.8,

 colsample_bytree=0.6,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 reg_alpha = 1,

 seed=27)

cv_results = model_fit(xgb5,trainX, trainY, testX, testY)
import pickle



#load the embeddings 

tfidf_vectorizer = pickle.load(open("/kaggle/input/trained-model/tfidf_vectorizer","rb"))

tfidf_vectorizer_ngram2 = pickle.load(open("/kaggle/input/trained-model/tfidf_vectorizer_ngram2","rb"))

bow_vectorizer = pickle.load(open("/kaggle/input/trained-model/bow_vectorizer","rb"))



test_tfidf_vectorizer =  tfidf_vectorizer.transform(test.text)

test_tfidf_vectorizer_ngram2 =  tfidf_vectorizer_ngram2.transform(test.text)

test_bow_vectorizer =  bow_vectorizer.transform(test.text)
#load models

svm_bow_model = pickle.load(open ("/kaggle/input/trained-model/svm_bow" ,"rb"))

nb_tfidf_ngm1_model = pickle.load(open ("/kaggle/input/trained-model/nb_tfidf_ngm1" ,"rb"))

sgd_tfidf_ngm2_model = pickle.load(open ("/kaggle/input/trained-model/sgd_tfidf_ngm2" ,"rb"))

svm_tfidf_opt_model = pickle.load(open ("/kaggle/input/trained-model/svm_tfidf_opt" ,"rb"))

nb_bow_model_model = pickle.load(open ("/kaggle/input/trained-model/nb_bow_model" ,"rb"))

sgd_tfidf_ngm1_model = pickle.load(open ("/kaggle/input/trained-model/sgd_tfidf_ngm1" ,"rb"))

sgd_tfidf_opt_model = pickle.load(open ("/kaggle/input/trained-model/sgd_tfidf_opt" ,"rb"))

svm_tfidf_model = pickle.load(open ("/kaggle/input/trained-model/svm_tfidf" ,"rb"))

xgboost_model = pickle.load(open ("/kaggle/input/trained-model/xgboost" ,"rb"))

nb_tfidf_ngm2_model = pickle.load(open ("/kaggle/input/trained-model/nb_tfidf_ngm2" ,"rb"))

sgd_bow_model = pickle.load(open ("/kaggle/input/trained-model/sgd_bow" ,"rb"))
#predictions

#level1 prediction

test_model_leve1_score = pd.DataFrame()

test_model_leve1_score["nb_bow"]  = nb_bow_model_model.predict(test_bow_vectorizer)

test_model_leve1_score["nb_tfidf_ngm1"]  = nb_tfidf_ngm1_model.predict(test_tfidf_vectorizer)

test_model_leve1_score["nb_tfidf_ngm2"]  = nb_tfidf_ngm2_model.predict(test_tfidf_vectorizer_ngram2)

test_model_leve1_score["sdg_bow"]  = sgd_bow_model.predict(test_bow_vectorizer)

test_model_leve1_score["sgd_tfidf_ngm1"]  = sgd_tfidf_ngm1_model.predict(test_tfidf_vectorizer)

test_model_leve1_score["sgd_tfidf_ngm2"]  = sgd_tfidf_ngm2_model.predict(test_tfidf_vectorizer_ngram2)

test_model_leve1_score["sgd_tfidf_opt"]  = sgd_tfidf_opt_model.predict(test_tfidf_vectorizer_ngram2)

test_model_leve1_score["svm_bow"]  = svm_bow_model.predict(test_bow_vectorizer)

test_model_leve1_score["svm_tfidf_ngm1"]  = svm_tfidf_model.predict(test_tfidf_vectorizer)

test_model_leve1_score["svm_tfidf_opt"]  = svm_tfidf_opt_model.predict(test_tfidf_vectorizer_ngram2)

final_prediction = xgboost_model.predict(test_model_leve1_score)

test["target"] = final_prediction

test[["id","target"]].to_csv("/kaggle/working/test.csv")