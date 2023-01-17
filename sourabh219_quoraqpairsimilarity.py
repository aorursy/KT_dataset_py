!pip install fuzzywuzzy
!pip install distance
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
data=pd.read_csv('../input/quoraqpairtrain/train.csv')
from nltk.corpus import stopwords
import nltk
import ssl
import re
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
print("Top 5 Datapoints : \n",data.head())
print()
print("Shape of Data : ",data.shape)
print()
print("Shape of Data where questions are same : ",data[data['is_duplicate']==1].shape)
print()
print("Shape of Data where questions are different : ",data[data['is_duplicate']==0].shape)
print()
print("Distribution of the two classes :\n",data['is_duplicate'].value_counts())
print()

def preprocess(x):
        x=str(x).lower()
        x=x.replace(',000,000','m').replace(',000','k').replace("'","`").replace("won't",'will not').replace("can't","can not").replace('cannot','can not').replace("n't",'not').replace("what's","what is").replace("it's","it is").replace("'ve","have").replace("i'm","i am").replace("'re","are").replace("he's","he is").replace("she's","she is ").replace("'s","own").replace("%"," precent").replace("₹"," rupeee").replace("$","dollar").replace("€","euro").replace("'ll","will")
        porter = PorterStemmer()
        pattern = re.compile('\W')
    
        if type(x) == type(''):
            x = re.sub(pattern, ' ', x)
        if type(x) == type(''):
          x = porter.stem(x)
          example1 = BeautifulSoup(x)
          x = example1.get_text()      
        return str(x)
data.isnull().sum() #Now we have no null entries in our dataset as you can seee we had null entries in qustions only and we cannot afford to fill these entries with any other technique to handle nul values.
stop_words=stopwords.words('english')
def token_features(q1,q2):
  feature=[0.0]*10
  
  q1_tokens=q1.split()
  q2_tokens=q2.split()

  if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return feature
  q1_words=set([word for word in q1_tokens if word not in stop_words])
  q2_words=set([word for word in q2_tokens if word not in stop_words]) 

  q1_stop_words=set([word for word in q1_tokens if word in stop_words])
  q2_stop_words=set([word for word in q2_tokens if word in stop_words]) 

  common_word_count=len(q1_words.intersection(q2_words))
  common_token_count=len(set(q1_tokens).intersection(set(q2_tokens)))
  common_stop_count = len(q1_stop_words.intersection(q2_stop_words))

  feature[0]=common_word_count/(min(len(q1_words),len(q2_words))+0.0001)
  feature[1]=common_word_count/(max(len(q1_words),len(q2_words))+0.0001)
  feature[2]=common_stop_count/(min(len(q1_stop_words),len(q2_stop_words))+0.0001)
  feature[3]=common_stop_count/(max(len(q1_stop_words),len(q2_stop_words))+0.0001)
  feature[4]=common_token_count/(min(len(q1_tokens),len(q2_tokens))+0.0001)
  feature[5]=common_token_count/(max(len(q1_tokens),len(q2_tokens))+0.0001)
  feature[6] = int(q1_tokens[-1] == q2_tokens[-1])
  feature[7] = int(q1_tokens[0] == q2_tokens[0])  
  feature[8] = abs(len(q1_tokens) - len(q2_tokens))
    #Average Token Length of both Questions
  feature[9] = (len(q1_tokens) + len(q2_tokens))/2
  return feature
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)
def Jaccard_dist(x,y):
    return (len(set(str(x)).intersection(set(str(y))))/len(set(str(x)).union(set(str(y)))))
#Featurization
from tqdm import tqdm
def featurize(data):
    data["question1"] = data["question1"].apply(preprocess)
    data["question2"] = data["question2"].apply(preprocess)
    data=data[data['question1'].notnull()]
    data=data[data['question2'].notnull()]
    data.reset_index(inplace=True,drop=True)
    for j,i in enumerate(data['question1']):
        if len(i)<1:
            data.drop([j],inplace=True)
    data.reset_index(inplace=True,drop=True)    
    for j,i in enumerate(data['question2']):
        if len(i)<1:
            data.drop([j],inplace=True)
    data.reset_index(inplace=True,drop=True)
    
    vc=data['question1'].value_counts()
    vc2=data['question2'].value_counts()
    data['qid1_freq']=pd.Series()
    data['qid2_freq']=pd.Series()
    data['len_of_q1']=pd.Series()
    data['len_of_q2']=pd.Series()
    data['words_in_q1']=pd.Series()
    data['words_in_q2']=pd.Series()
    data['common_words_count']=pd.Series()
    data['common_words_of_total']=pd.Series()
    data['total_words_count']=pd.Series()
    data['sum_of_freq']=pd.Series()
    data['diff_of_freq']=pd.Series()
    for i in tqdm(range(0,len(data))):
        data['qid1_freq'][i]=vc[data['question1'][i]]
        data['qid2_freq'][i]=vc2[data['question2'][i]]
        data['len_of_q1'][i]=len(data['question1'][i])
        data['len_of_q2'][i]=len(data['question2'][i])
        data['words_in_q1'][i]=len(data['question1'][i].split())
        data['words_in_q2'][i]=len(data['question2'][i].split())
        data['common_words_count'][i]=len(set(data['question1'][i]).intersection(set(data['question2'][i])))
        data['total_words_count'][i]=len(data['question1'][i].split()+data['question2'][i].split())
        data['common_words_of_total'][i]=len(set(data['question1'][i]).intersection(set(data['question2'][i])))/len(data['question1'][i].split()+data['question2'][i].split())
        data['sum_of_freq'][i]=vc[data['question1'][i]]+vc2[data['question2'][i]]
        data['diff_of_freq'][i]=abs(vc[data['question1'][i]]-vc2[data['question2'][i]])
    features=data.apply(lambda x:token_features(x['question1'],x['question2']),axis=1)
    data["cwc_min"]= list(map(lambda x: x[0],features))
    data["cwc_max"]= list(map(lambda x: x[1], features))
    data["csc_min"]= list(map(lambda x: x[2], features))
    data["csc_max"] = list(map(lambda x: x[3], features))
    data["ctc_min"]= list(map(lambda x: x[4], features))
    data["ctc_max"]= list(map(lambda x: x[5], features))
    data["last_word_eq"]= list(map(lambda x: x[6], features))
    data["first_word_eq"]= list(map(lambda x: x[7], features))
    data["abs_len_diff"]= list(map(lambda x: x[8], features))
    data["mean_len"]=list(map(lambda x: x[9], features))
    #Fuzzy_Features
    # The token sort approach involves tokenizing the string in question, sorting the tokens alphabetically, and 
    # then joining them back into a string We then compare the transformed strings with a simple ratio().
    data["token_sort_ratio"]      = data.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    data["fuzz_ratio"]            = data.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    data["fuzz_partial_ratio"]    = data.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    data["longest_substr_ratio"]  = data.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    data["Jaccard_dist"]  = data.apply(lambda x: Jaccard_dist(x["question1"], x["question2"]), axis=1)
    return data
data=featurize(data)
import spacy
from scipy.sparse import hstack
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
X=data.drop('is_duplicate',axis=1)
y=data['is_duplicate']
from sklearn.model_selection import train_test_split #To avoid data leakage problem we did the splitting at the start
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=41,stratify=y)

X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)

questions=list(X_train['question1']+X_train['question2'])
tfidf_vectorizer=TfidfVectorizer()
tfidf_vectorizer.fit_transform(questions)
word_and_its_tfidf=dict(zip(tfidf_vectorizer.get_feature_names(),tfidf_vectorizer.idf_))

tfidf_vectorizer.transform(list(X_test['question1']+X_test['question2']))
word_and_its_tfidf_test=dict(zip(tfidf_vectorizer.get_feature_names(),tfidf_vectorizer.idf_))

def featurize_with_glove(tfidf_matrix,data):
    vecs1=[]
    for qu1 in tqdm(list(data['question1'])):
        doc1=nlp(qu1)
        mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
        for word1 in doc1:
            vec1=word1.vector
            try:
                idf = tfidf_matrix[str(word1)]
            except:
                idf = 0
            mean_vec1 += vec1 * idf
        mean_vec1 = mean_vec1.mean(axis=0)
        vecs1.append(mean_vec1)
    #data=data.join(pd.DataFrame(np.array(vecs1).T.tolist()),lsuffix='_')
    vecs2=[]
    for qu2 in tqdm(list(data['question2'])):
        doc2=nlp(qu2)
        mean_vec2 = np.zeros([len(doc2), len(doc2[0].vector)])
        for word2 in doc2:
            vec2=word2.vector
            try:
                idf = tfidf_matrix[str(word2)]
            except:
                idf = 0
            mean_vec2 += vec2 * idf
        mean_vec2 = mean_vec2.mean(axis=0)
        vecs2.append(mean_vec2)
    #data=data.join(pd.DataFrame(np.array(vecs2).T.tolist()),rsuffix='_')
    return data,vecs1,vecs2
X_train,train_vec1,train_vec2=featurize_with_glove(word_and_its_tfidf,X_train)
X_test,test_vec1,test_vec2=featurize_with_glove(word_and_its_tfidf_test,X_test)
X_tr=X_train.iloc[:,5:]
X_tst=X_test.iloc[:,5:]
import pickle
pickle.dump(X_tr,open('X_tr.pickle','wb'))
pickle.dump(X_tst,open('X_tst.pickle','wb'))
pickle.dump(y_train,open('y_train.pickle','wb'))
pickle.dump(y_test,open('y_test.pickle','wb'))
pickle.dump(tfidf_vectorizer,open('tfidf_vectorizer.pickle','wb'))
A=pd.DataFrame(train_vec1, columns=['q2_0',
'q2_1',
'q2_2',
'q2_3',
'q2_4',
'q2_5',
'q2_6',
'q2_7',
'q2_8',
'q2_9',
'q2_10',
'q2_11',
'q2_12',
'q2_13',
'q2_14',
'q2_15',
'q2_16',
'q2_17',
'q2_18',
'q2_19',
'q2_20',
'q2_21',
'q2_22',
'q2_23',
'q2_24',
'q2_25',
'q2_26',
'q2_27',
'q2_28',
'q2_29',
'q2_30',
'q2_31',
'q2_32',
'q2_33',
'q2_34',
'q2_35',
'q2_36',
'q2_37',
'q2_38',
'q2_39',
'q2_40',
'q2_41',
'q2_42',
'q2_43',
'q2_44',
'q2_45',
'q2_46',
'q2_47',
'q2_48',
'q2_49',
'q2_50',
'q2_51',
'q2_52',
'q2_53',
'q2_54',
'q2_55',
'q2_56',
'q2_57',
'q2_58',
'q2_59',
'q2_60',
'q2_61',
'q2_62',
'q2_63',
'q2_64',
'q2_65',
'q2_66',
'q2_67',
'q2_68',
'q2_69',
'q2_70',
'q2_71',
'q2_72',
'q2_73',
'q2_74',
'q2_75',
'q2_76',
'q2_77',
'q2_78',
'q2_79',
'q2_80',
'q2_81',
'q2_82',
'q2_83',
'q2_84',
'q2_85',
'q2_86',
'q2_87',
'q2_88',
'q2_89',
'q2_90',
'q2_91',
'q2_92',
'q2_93',
'q2_94',
'q2_95'])
B=pd.DataFrame(train_vec2, columns=['q1_0',
'q1_1',
'q1_2',
'q1_3',
'q1_4',
'q1_5',
'q1_6',
'q1_7',
'q1_8',
'q1_9',
'q1_10',
'q1_11',
'q1_12',
'q1_13',
'q1_14',
'q1_15',
'q1_16',
'q1_17',
'q1_18',
'q1_19',
'q1_20',
'q1_21',
'q1_22',
'q1_23',
'q1_24',
'q1_25',
'q1_26',
'q1_27',
'q1_28',
'q1_29',
'q1_30',
'q1_31',
'q1_32',
'q1_33',
'q1_34',
'q1_35',
'q1_36',
'q1_37',
'q1_38',
'q1_39',
'q1_40',
'q1_41',
'q1_42',
'q1_43',
'q1_44',
'q1_45',
'q1_46',
'q1_47',
'q1_48',
'q1_49',
'q1_50',
'q1_51',
'q1_52',
'q1_53',
'q1_54',
'q1_55',
'q1_56',
'q1_57',
'q1_58',
'q1_59',
'q1_60',
'q1_61',
'q1_62',
'q1_63',
'q1_64',
'q1_65',
'q1_66',
'q1_67',
'q1_68',
'q1_69',
'q1_70',
'q1_71',
'q1_72',
'q1_73',
'q1_74',
'q1_75',
'q1_76',
'q1_77',
'q1_78',
'q1_79',
'q1_80',
'q1_81',
'q1_82',
'q1_83',
'q1_84',
'q1_85',
'q1_86',
'q1_87',
'q1_88',
'q1_89',
'q1_90',
'q1_91',
'q1_92',
'q1_93',
'q1_94',
'q1_95'])
X_tr
X_tr=np.hstack([X_tr,B,A])
A=pd.DataFrame(test_vec1, columns=['q2_0',
'q2_1',
'q2_2',
'q2_3',
'q2_4',
'q2_5',
'q2_6',
'q2_7',
'q2_8',
'q2_9',
'q2_10',
'q2_11',
'q2_12',
'q2_13',
'q2_14',
'q2_15',
'q2_16',
'q2_17',
'q2_18',
'q2_19',
'q2_20',
'q2_21',
'q2_22',
'q2_23',
'q2_24',
'q2_25',
'q2_26',
'q2_27',
'q2_28',
'q2_29',
'q2_30',
'q2_31',
'q2_32',
'q2_33',
'q2_34',
'q2_35',
'q2_36',
'q2_37',
'q2_38',
'q2_39',
'q2_40',
'q2_41',
'q2_42',
'q2_43',
'q2_44',
'q2_45',
'q2_46',
'q2_47',
'q2_48',
'q2_49',
'q2_50',
'q2_51',
'q2_52',
'q2_53',
'q2_54',
'q2_55',
'q2_56',
'q2_57',
'q2_58',
'q2_59',
'q2_60',
'q2_61',
'q2_62',
'q2_63',
'q2_64',
'q2_65',
'q2_66',
'q2_67',
'q2_68',
'q2_69',
'q2_70',
'q2_71',
'q2_72',
'q2_73',
'q2_74',
'q2_75',
'q2_76',
'q2_77',
'q2_78',
'q2_79',
'q2_80',
'q2_81',
'q2_82',
'q2_83',
'q2_84',
'q2_85',
'q2_86',
'q2_87',
'q2_88',
'q2_89',
'q2_90',
'q2_91',
'q2_92',
'q2_93',
'q2_94',
'q2_95'])
B=pd.DataFrame(test_vec2, columns=['q1_0',
'q1_1',
'q1_2',
'q1_3',
'q1_4',
'q1_5',
'q1_6',
'q1_7',
'q1_8',
'q1_9',
'q1_10',
'q1_11',
'q1_12',
'q1_13',
'q1_14',
'q1_15',
'q1_16',
'q1_17',
'q1_18',
'q1_19',
'q1_20',
'q1_21',
'q1_22',
'q1_23',
'q1_24',
'q1_25',
'q1_26',
'q1_27',
'q1_28',
'q1_29',
'q1_30',
'q1_31',
'q1_32',
'q1_33',
'q1_34',
'q1_35',
'q1_36',
'q1_37',
'q1_38',
'q1_39',
'q1_40',
'q1_41',
'q1_42',
'q1_43',
'q1_44',
'q1_45',
'q1_46',
'q1_47',
'q1_48',
'q1_49',
'q1_50',
'q1_51',
'q1_52',
'q1_53',
'q1_54',
'q1_55',
'q1_56',
'q1_57',
'q1_58',
'q1_59',
'q1_60',
'q1_61',
'q1_62',
'q1_63',
'q1_64',
'q1_65',
'q1_66',
'q1_67',
'q1_68',
'q1_69',
'q1_70',
'q1_71',
'q1_72',
'q1_73',
'q1_74',
'q1_75',
'q1_76',
'q1_77',
'q1_78',
'q1_79',
'q1_80',
'q1_81',
'q1_82',
'q1_83',
'q1_84',
'q1_85',
'q1_86',
'q1_87',
'q1_88',
'q1_89',
'q1_90',
'q1_91',
'q1_92',
'q1_93',
'q1_94',
'q1_95'])
X_tst=np.hstack((X_tst,B,A))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_tr,y_train)
lr.score(X_tst,y_test)
import pickle
with open('X_tr.pickle','wb') as f:
    pickle.dump(X_tr,f)
with open('X_tst.pickle','wb') as f:
    pickle.dump(X_tst,f)
with open('y_train.pickle','wb') as f:
    pickle.dump(y_train,f)
with open('y_test.pickle','wb') as f:
    pickle.dump(y_test,f)
with open('tfidf_vectorizer.pickle','wb') as f:
    pickle.dump(tfidf_vectorizer,f)
with open('glove_nlp.pickle','wb') as f:
    pickle.dump(nlp,f)
with open('logistc_regressor.pickle','wb') as f:
    pickle.dump(lr,f)
import pickle
with open('X_tr.pickle','rb') as f:
    X_tr=pickle.load(f)
with open('X_tst.pickle','rb') as f:
    X_tst=pickle.load(f)
with open('y_train.pickle','rb') as f:
    y_train=pickle.load(f)
with open('y_test.pickle','rb') as f:
    y_test=pickle.load(f)
with open('tfidf_vectorizer.pickle','rb') as f:
    tfidf_vectorizer=pickle.load(f)
with open('glove_nlp.pickle','rb') as f:
    nlp=pickle.load(f)
with open('logistc_regressor.pickle','rb') as f:
    lr=pickle.load(f)
y_pred=lr.predict(X_tst)
y_pred
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test.values,y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test.values,y_pred))
from sklearn.ensemble import RandomForestClassifier
Rfc=RandomForestClassifier(n_estimators=500,n_jobs=-1)
Rfc.fit(X_tr,y_train)
roc_auc_score(y_test,Rfc.predict(X_tst))
from sklearn.metrics import log_loss
print("RandomForest log loss",log_loss(y_test,Rfc.predict_proba(X_tst)))
print("RandomForest Train log loss",log_loss(y_train,Rfc.predict_proba(X_tr)))
print()
print("Logistic Regression log loss",log_loss(y_test,lr.predict_proba(X_tst)))
print()
print("CatBoostClassifier Train log loss",log_loss(y_train,cat.predict_proba(X_tr)))
print("CatBoostClassifier log loss",log_loss(y_test,cat.predict_proba(X_tst)))
Rfc.feature_importances_
from sklearn.metrics import classification_report
print(classification_report(y_test.values,Rfc.predict(X_tst)))
from catboost import CatBoostClassifier
cat=CatBoostClassifier()
cat.fit(X_tr,y_train)
roc_auc_score(y_test.values,cat.predict(X_tst))
print(classification_report(y_test.values,cat.predict(X_tst)))
cat.score(X_tst,y_test)
