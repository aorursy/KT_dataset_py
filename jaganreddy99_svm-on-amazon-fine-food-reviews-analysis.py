import gensim
%matplotlib inline
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import matplotlib.pyplot as pt
from  nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import seaborn as sn
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,f1_score,log_loss,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

data = pd.read_pickle("../input/final_data.pkl")
data.shape
#noe lets split the data 
from sklearn.model_selection import train_test_split
x,x_test,y,y_test = train_test_split(data['CleanedText'],data['Score'],train_size=0.8,shuffle=False)
x_train,x_cv,y_train,y_cv = train_test_split(x,y,train_size=0.8,shuffle=False)
#Lets Vecotirize
#bagof words
bag_words = CountVectorizer()
x_train_bag= bag_words.fit_transform(x_train)
x_test_bag= bag_words.transform(x_test)
x_cv_bag= bag_words.transform(x_cv)

print('After vectorizing shape of x Train',x_train_bag.shape)
print('After vectorizing shape of x Test',x_test_bag.shape)
print('After vectorizing shape of x CV',x_cv_bag.shape)
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
def linear_svm(x_train,y_train,x_cv,y_cv):
   
    alpha = [10 ** x for x in range(-4,4) ]
    for penalty in ['l1','l2']:
        train_f1_score = []
        cv_f1_score = []
        for i in alpha:
            clf = SGDClassifier(alpha=i,penalty=penalty,loss='hinge')
            clf.fit(x_train,y_train)
            pred_y_train = clf.predict(x_train)
            pred_y_cv = clf.predict(x_cv)
            print('for value of alpha:',i,'and penalty:',penalty,' f1_score of cv data is',f1_score(y_cv,pred_y_cv))
            train_f1_score.append(f1_score(y_train,pred_y_train))
            cv_f1_score.append(f1_score(y_cv,pred_y_cv))
        fig, ax = plt.subplots()
        
        ax.set_xscale('log')
        ax.plot(alpha, cv_f1_score,c='g',marker='o', label='CV_f1 score')
        for i, txt in enumerate(np.round(cv_f1_score,2)):
            ax.annotate((alpha[i],np.round(txt,2)), (alpha[i],cv_f1_score[i]))
        ax.plot(alpha, train_f1_score,c='b',marker='o',label='Train_f1 score')
        for i, txt in enumerate(np.round(train_f1_score,2)):
            ax.annotate((alpha[i],np.round(txt,2)), (alpha[i],train_f1_score[i]))
    
        #plt.grid()
        
        print('\n\n----------------plot for penalty :',penalty,'----------------------------')
        
        plt.title("F1 score for each alpha")
        plt.xlabel("alpha values's",)
        plt.ylabel("f1 score")
        
        plt.legend()
        plt.show()
            
import seaborn as sns
def best_svm_linear(x_train,y_train,x_test,y_test,c,penalty):
    
    
    
        clf_dt_best = SGDClassifier(alpha=c,penalty=penalty,loss='hinge')
        clf_dt_best.fit(x_train, y_train)
        predict_y_train = clf_dt_best.predict(x_train)
        print('For values of c = ', c, "The train f1  score is:",f1_score(y_train,predict_y_train))
        predict_y_test = clf_dt_best.predict(x_test)
        print('For values of  c = ', c, "The test  f1 score is:",f1_score(y_test,predict_y_test))
        acc_t = accuracy_score(y_train,predict_y_train)
        print('Accuracy on train data is ',acc_t)
        acc = accuracy_score(y_test,predict_y_test)
        print('Accuracy on test data is ',acc)
        c_1 = confusion_matrix(y_train, predict_y_train)
        C = confusion_matrix(y_test, predict_y_test)
        print("-"*20, "Confusion matrix on train data", "-"*20)
        plt.figure(figsize=(20,7))
    
        sns.heatmap(c_1, annot=True, cmap="YlGnBu", fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print("-"*20, "Confusion matrix on test data", "-"*20)
        plt.figure(figsize=(20,7))
    
        sns.heatmap(C, annot=True, cmap="YlGnBu", fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print(classification_report(y_test, predict_y_test))

        return clf_dt_best,acc,acc_t,c,penalty
    
        
    
linear_svm(x_train_bag,y_train,x_cv_bag,y_cv)
best_model_bag,acc_test_bag,acc_train_bag,c_bag,penalty_bag = best_svm_linear(x_train_bag,y_train,x_test_bag,y_test,0.001,'l2')
from sklearn.model_selection import GridSearchCV
import seaborn as sns
def tuning_svm(x_train,y_train):
    c = [10 ** x for x in range(-4,4) ]
    gamma = [10 ** x for x in range(-4,4) ]
    param_grid = {'C':c,'gamma':gamma}
    clf = SVC()
    clf_search = GridSearchCV(clf,param_grid,scoring='f1',verbose=1,cv=5)
    clf_search.fit(x_train,y_train)
    print("Best HyperParameter: ",clf_search.best_params_)
    print("Best f1_score,",clf_search.best_score_)
    scores = clf_search.cv_results_['mean_test_score'].reshape(len(c),
                                                     len(gamma))
    sns.heatmap(scores, annot=True, cmap="YlGnBu", fmt="d",)
    plt.xlabel('C value')
    plt.ylabel('gamma value')
    plt.title('f1_score for differnet c and gamma values')
    plt.show()
    
    


from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
def kernel_svm(x_train,y_train,x_cv,y_cv):
    #standard =StandardScaler(with_mean=False) 
    #x_train = standard.fit_transform(x_train)
    #x_cv = standard.transform(x_cv)
    alpha = [10 ** x for x in range(-4,4) ]
    train_f1_score = []
    cv_f1_score = []
    for i in alpha:
            clf = SVC(C=i,kernel='rbf')
            clf.fit(x_train,y_train)
            pred_y_train = clf.predict(x_train)
            pred_y_cv = clf.predict(x_cv)
            print('for value of c:',i, 'f1_score of cv data is',f1_score(y_cv,pred_y_cv))
            train_f1_score.append(f1_score(y_train,pred_y_train))
            cv_f1_score.append(f1_score(y_cv,pred_y_cv))
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.plot(alpha, cv_f1_score,c='g',marker='o',label='cv_f1 score')
    for i, txt in enumerate(np.round(cv_f1_score,3)):
        ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_f1_score[i]))
    ax.plot(alpha, train_f1_score,c='b',marker='o',label='Train_f1 score')
    for i, txt in enumerate(np.round(train_f1_score,3)):
         ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],train_f1_score[i]))
    
    plt.grid()
        
   
    plt.title("F1 score for each alpha")
    plt.xlabel("alpha values's",)
    plt.ylabel("f1 score")
    
    plt.legend()
    plt.show()
            
        
import seaborn as sns
def best_svm_kernal(x_train,y_train,x_test,y_test,c):
        #standard = StandardScaler(with_mean=False)
        #x_train = standard.fit_transform(x_train)
        #x_test = standard.transform(x_test)
    
    
        clf_dt_best = SVC(C=c)
        clf_dt_best.fit(x_train, y_train)
        predict_y_train = clf_dt_best.predict(x_train)
        print('For values of c = ', c, "The train f1  score is:",f1_score(y_train,predict_y_train))
        predict_y_test = clf_dt_best.predict(x_test)
        print('For values of  c = ', c, "The test  f1 score is:",f1_score(y_test,predict_y_test))
        acc_t = accuracy_score(y_train,predict_y_train)
        print('Accuracy on train data is ',acc_t)
        acc = accuracy_score(y_test,predict_y_test)
        print('Accuracy on test data is ',acc)
        c_1 = confusion_matrix(y_train, predict_y_train)
        C = confusion_matrix(y_test, predict_y_test)
        print("-"*20, "Confusion matrix on train data", "-"*20)
        plt.figure(figsize=(20,7))
    
        sns.heatmap(c_1, annot=True, cmap="YlGnBu", fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print("-"*20, "Confusion matrix on test data", "-"*20)
        plt.figure(figsize=(20,7))
    
        sns.heatmap(C, annot=True, cmap="YlGnBu", fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('Original Class')
        plt.show()
        print(classification_report(y_test, predict_y_test))

        return clf_dt_best,acc,acc_t,c
    
sample_data = data.head(10000)
sample_data.shape
#noe lets split the data 
from sklearn.model_selection import train_test_split
x_train_s,x_test_s,ytrain_s,y_test_s = train_test_split(sample_data['CleanedText'],sample_data['Score'],train_size=0.8,shuffle=False)



bag_words_con = CountVectorizer()
X_train_bag_c= bag_words_con.fit_transform(x_train_s)
X_test_bag_c= bag_words_con.transform(x_test_s)

print('After vectorizing with 2000 features shape of x Train',X_train_bag_c.shape)
print('After vectorizing with 2000 features shape of x Test',X_test_bag_c.shape)


tuning_svm(X_train_bag_c,ytrain_s)
clf_dt_best_bag_k,bag_acc_k,bag_acc_train_k,c_k = best_svm_kernal(X_train_bag_c,y_train_s,X_test_bag_c,y_test_s,5)
#Lets Vecotirize
#bagof words
tfidf_words = TfidfVectorizer()
x_train_tfidf= tfidf_words.fit_transform(x_train)
x_test_tfidf= tfidf_words.transform(x_test)
x_cv_tfidf = tfidf_words.transform(x_cv)

print('After vectorizing shape of x Train',x_train_tfidf.shape)
print('After vectorizing shape of x Test',x_test_tfidf.shape)
print('After vectorizing shape of x CV',x_cv_tfidf.shape)
linear_svm(x_train_tfidf,y_train,x_cv_tfidf,y_cv)
best_model_tfidf,acc_test_tfidf,acc_train_tfidf,c_tfidf,penalty_tfidf = best_svm_linear(x_train_tfidf,y_train,x_test_tfidf,y_test,0.0001,'l2')
tfidf_words_con = TfidfVectorizer()
X_train_tfidf_c= tfidf_words_con.fit_transform(x_train_s)
X_test_tfidf_c= tfidf_words_con.transform(x_test_s)
X_cv_tfidf_c= tfidf_words_con.transform(x_cv_s)
print('After vectorizing with 2000 features shape of x Train',X_train_tfidf_c.shape)
print('After vectorizing with 2000 features shape of x Test',X_test_tfidf_c.shape)
print('After vectorizing with 2000 features shape of x CV',X_cv_tfidf_c.shape)
kernel_svm(X_train_tfidf_c,y_train_s,X_cv_tfidf_c,y_cv_s)
clf_dt_best_tfidf_k,tfidf_acc_k,tfidf_acc_train_k,c_tfidf_k = best_svm_kernal(X_train_tfidf_c,y_train_s,X_test_tfidf_c,y_test_s,100)
#lets define some functions to  clean the reviews

#to remove HTML Tags
def clean_html(x):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', x)
    return cleantext
#  to remove unwanted charecteres like '!',',' etc.

def cleansen(x):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',x)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

#stop words

stop_words = set(stopwords.words('english'))
#intialising stremming
stemmer = nltk.stem.SnowballStemmer('english')
import datetime
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
i=0
list_sent_train=[]
for sent in x_train:
    filtered_sentence=[]
    sent = sent.decode('utf-8')
  
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_sent_train.append(filtered_sentence)
w2v_model=gensim.models.Word2Vec(list_sent_train,min_count=5,size=50, workers=4) 
x_train_avgw2v= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_train: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_train_avgw2v.append(sent_vec)
print(len(x_train_avgw2v))
print(len(x_train_avgw2v[0]))
list_sent_test=[]
for sent in x_test:
    filtered_sentence=[]
    sent = sent.decode('utf-8')
    sent=clean_html(sent)
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_sent_test.append(filtered_sentence)
x_test_avgw2v= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_test_avgw2v.append(sent_vec)
print(len(x_test_avgw2v))
print(len(x_test_avgw2v[0]))
list_sent_cv=[]
for sent in x_cv:
    filtered_sentence=[]
    sent = sent.decode('utf-8')
    sent=clean_html(sent)
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_sent_cv.append(filtered_sentence)
x_cv_avgw2v= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_cv: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_cv_avgw2v.append(sent_vec)
print(len(x_cv_avgw2v))
print(len(x_cv_avgw2v[0]))
linear_svm(x_train_avgw2v,y_train,x_cv_avgw2v,y_cv)
best_svm_linear(x_train_avgw2v,y_train,x_test_avgw2v,y_test,0.001,'l2')

list_sent_train_s=[]
for sent in x_train_s:
    filtered_sentence_s=[]
    sent = sent.decode('utf-8')
    sent=clean_html(sent)
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence_s.append(cleaned_words.lower())
            else:
                continue 
    list_sent_train_s.append(filtered_sentence_s)
w2v_model_s = gensim.models.Word2Vec(list_sent_train_s,min_count=5,size=50, workers=4)
x_train_avgw2v_s= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_train_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_s.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_train_avgw2v_s.append(sent_vec)
print(len(x_train_avgw2v_s))
print(len(x_train_avgw2v_s[0]))
list_sent_test_s=[]
for sent in x_test_s:
    filtered_sentence=[]
    sent = sent.decode('utf-8')
    sent=clean_html(sent)
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_sent_test_s.append(filtered_sentence)
x_test_avgw2v_s= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_test_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_test_avgw2v_s.append(sent_vec)
print(len(x_test_avgw2v_s))
print(len(x_test_avgw2v_s[0]))
list_sent_cv_s=[]
for sent in x_cv_s:
    filtered_sentence=[]
    sent = sent.decode('utf-8')
    sent=clean_html(sent)
    for w in sent.split():
        for cleaned_words in w.split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_sent_cv_s.append(filtered_sentence)
x_cv_avgw2v_s= []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_sent_cv_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    x_cv_avgw2v_s.append(sent_vec)
print(len(x_cv_avgw2v_s))
print(len(x_cv_avgw2v_s[0]))
kernel_svm(x_train_avgw2v_s,y_train_s,x_cv_avgw2v_s,y_cv_s)
best_svm_kernal(x_train_avgw2v_s,y_train_s,x_test_avgw2v_s,y_test_s,150)
tfidf_feat = tfidf_words.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

x_train_tfidfwv= []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_sent_train: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_train_tfidfwv.append(sent_vec)
    row += 1
print('train shape',len(x_train_tfidfwv),len(x_train_tfidfwv[0]))

x_test_tfidfwv= []
for sent in list_sent_test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_test_tfidfwv.append(sent_vec)
    row += 1
print('test shape',len(x_test_tfidfwv),len(x_test_tfidfwv[0]))
x_cv_tfidfwv= []
for sent in list_sent_cv: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_cv_tfidfwv.append(sent_vec)
    row += 1
print('test shape',len(x_cv_tfidfwv),len(x_cv_tfidfwv[0]))
x_train_tfidfwv = np.nan_to_num(x_train_tfidfwv)
x_test_tfidfwv = np.nan_to_num(x_test_tfidfwv)
x_cv_tfidfwv = np.nan_to_num(x_cv_tfidfwv)
linear_svm(x_train_tfidfwv,y_train,x_cv_tfidfwv,y_cv)
best_svm_linear(x_train_tfidfwv,y_train,x_test_tfidfwv,y_test,0.01,'l2')
tfidf_feats = tfidf_words_con.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

x_train_tfidfwv_s= []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_sent_train_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_s.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf_s[row, tfidf_feats_s.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_train_tfidfwv_s.append(sent_vec)
    row += 1
print('train shape',len(x_train_tfidfwv_s),len(x_train_tfidfwv_s[0]))
x_test_tfidfwv_s= []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_sent_test_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_s.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf_t[row, tfidf_feats_s.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_test_tfidfwv_s.append(sent_vec)
    row += 1
print('test shape',len(x_test_tfidfwv_s),len(x_test_tfidfwv_s[0]))
x_cv_tfidfwv_s= []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_sent_cv_s: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_s.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = final_tf_idf_cv[row, tfidf_feats_s.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    sent_vec /= weight_sum
    x_cv_tfidfwv_s.append(sent_vec)
    row += 1
print('cv shape',len(x_cv_tfidfwv_s),len(x_cv_tfidfwv_s[0]))
x_train_tfidfwv_s = np.nan_to_num(x_train_tfidfwv_s)
x_test_tfidfwv_s = np.nan_to_num(x_test_tfidfwv_s)
x_cv_tfidfwv_s = np.nan_to_num(x_cv_tfidfwv_s)
kernel_svm(x_train_tfidfwv_s,y_train_s,x_cv_tfidfwv_s,y_cv_s)
best_svm_kernal(x_train_tfidfwv_s,y_train_s,x_test_tfidfwv_s,y_test_s,10)
x = ['This is the worst product i have ever bought']
x = bag_words.transform(x)
pred = best_model_bag.predict(x)
if (pred == 1):
    print('postivite review ')
else:
    print('Negative review')