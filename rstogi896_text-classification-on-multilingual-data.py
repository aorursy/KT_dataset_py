import pandas as pd
import spacy # fast NLP
import pandas as pd # dataframes
import langid # language identification (i.e. what language is this?)
from nltk.classify.textcat import TextCat # language identification from NLTK
from matplotlib.pyplot import plot # not as good as ggplot in R :p
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import hstack, csr_matrix
import spacy
train=pd.read_csv('../input/train.csv')
train.shape
train.head()
test=pd.read_csv('../input/test.csv')
test.shape
test.head()
train['Consumer_complaint_summary']=train['Consumer-complaint-summary']
test['Consumer_complaint_summary']=test['Consumer-complaint-summary']
from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(train['Consumer_complaint_summary'], title="Word Cloud of Consumer-complaint-summary in Train data")
from wordcloud import WordCloud, STOPWORDS

# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
plot_wordcloud(test['Consumer_complaint_summary'], title="Word Cloud of Consumer-complaint-summary in Test data")
train['Consumer_complaint_summary']=train['Consumer_complaint_summary'].str.lower()
test['Consumer_complaint_summary']=test['Consumer_complaint_summary'].str.lower()
train.head()
train.drop(['Consumer-complaint-summary'],axis=1,inplace=True)
test.drop(['Consumer-complaint-summary'],axis=1,inplace=True)
mapping = {'Yes':0, 'No':1}
train = train.replace({'Consumer-disputes':mapping})
test = test.replace({'Consumer-disputes':mapping})
train['Consumer-disputes'].fillna(1, inplace=True)
test['Consumer-disputes'].fillna(1, inplace=True)
train = train.replace(np.nan, 'Company has responded to the consumer and the CFPB and chooses not to provide a public response ', regex=True)
test = test.replace(np.nan, 'Company has responded to the consumer and the CFPB and chooses not to provide a public response ', regex=True)
mapping = {'Closed with explanation':0, 'Closed with non-monetary relief':1, 'Closed':2, 'Closed with monetary relief':3 ,'Untimely response':4}
train = train.replace({'Complaint-Status':mapping})
tr_train=pd.get_dummies(train['Transaction-Type'],drop_first=True)
tr_test=pd.get_dummies(test['Transaction-Type'],drop_first=True)
cor_train=pd.get_dummies(train['Company-response'],drop_first=True)
cor_test=pd.get_dummies(test['Company-response'],drop_first=True)
cor_train.shape,cor_test.shape
train=pd.concat([train,tr_train,cor_train],axis=1)
train.drop(['Transaction-Type','Company-response','Complaint-reason'],axis=1,inplace=True)
test=pd.concat([test,tr_test,cor_test],axis=1)
test.drop(['Transaction-Type','Company-response','Complaint-reason'],axis=1,inplace=True)
import datetime
train['Date-received'] = pd.to_datetime(train['Date-received'])
train['Date-sent-to-company'] = pd.to_datetime(train['Date-sent-to-company'])
train['date_diff'] = (train['Date-sent-to-company'] - train['Date-received']).dt.days
test['Date-received'] = pd.to_datetime(test['Date-received'])
test['Date-sent-to-company'] = pd.to_datetime(test['Date-sent-to-company'])
test['date_diff'] = (test['Date-sent-to-company'] - test['Date-received']).dt.days
#test_id=test['Complaint-ID']
train=train.drop(['Date-received','Date-sent-to-company'],axis=1)
test=test.drop(['Date-received','Date-sent-to-company'],axis=1)
train.shape,test.shape
nltk.download('stopwords')
stop = stopwords.words('english')
train[train['Consumer_complaint_summary'].duplicated(keep=False)].sort_values('Consumer_complaint_summary').head(5)
train = train.drop_duplicates('Consumer_complaint_summary')
train.shape,test.shape
# get the language id for each text
ids_langid_train = train['Consumer_complaint_summary'].apply(langid.classify)
# get the language id for each text
ids_langid_test = test['Consumer_complaint_summary'].apply(langid.classify)
# get just the language label
langs_train = ids_langid_train.apply(lambda tuple: tuple[0])
# get just the language label
langs_test = ids_langid_test.apply(lambda tuple: tuple[0])
print("Number of tagged languages (estimated):")
print(len(langs_train.unique()))

# percent of the total dataset in English
print("Percent of data in English (estimated):")
print((sum(langs_train=="en")/len(langs_train))*100)
# how many unique language labels were applied?
print("Number of tagged languages (estimated):")
print(len(langs_test.unique()))

# percent of the total dataset in English
print("Percent of data in English (estimated):")
print((sum(langs_test=="en")/len(langs_test))*100)
langs_train_df = pd.DataFrame(langs_train)
# count the number of times we see each language
langs_train_count = langs_train_df.Consumer_complaint_summary.value_counts()

langs_train_count.plot.bar(figsize=(20,10), fontsize=20)
langs_train_df.head()
langs_test_df = pd.DataFrame(langs_test)
# count the number of times we see each language
langs_test_count = langs_test_df.Consumer_complaint_summary.value_counts()

langs_test_count.plot.bar(figsize=(20,10), fontsize=20)
langs_test_df.head()
langs_train_df.shape,langs_test_df.shape
spanish_complain_train = train['Consumer_complaint_summary'][langs_train == "es"]
French_complain_train = train['Consumer_complaint_summary'][langs_train == "fr"]
english_complain_train = train['Consumer_complaint_summary'][langs_train == "en"]
spanish_complain_train.shape,French_complain_train.shape,english_complain_train.shape
spanish_complain_test = test['Consumer_complaint_summary'][langs_test == "es"]
French_complain_test = test['Consumer_complaint_summary'][langs_test == "fr"]
english_complain_test = test['Consumer_complaint_summary'][langs_test == "en"]
spanish_complain_test.shape,French_complain_test.shape,english_complain_test.shape
spanish_complain_train.head()
type(spanish_complain_train)
l=spanish_complain_train.index.values
l=l.tolist()
train_spanish=train.loc[l,:]
train_spanish.head()
l1=French_complain_train.index.values
l1=l1.tolist()
len(l1)
train_French=train.loc[l1,:]
train_French.head()
l2=english_complain_train.index.values
l2=l2.tolist()
len(l2)
m=l+l1+l2
len(m)
t=len(l)+len(l1)+len(l2)
print(t)
len(m)
le=l2
train_english=train.loc[le,:]
train_english.head()
train_english.shape
l4=spanish_complain_test.index.values
l4=l4.tolist()
len(l4)
l5=French_complain_test.index.values
l5=l5.tolist()
len(l5)
l6=english_complain_test.index.values
l6=l6.tolist()
z=l4+l5+l6
len(z)
t_new = np.array(z,dtype=object)
T=np.arange(test.shape[0])
T
c=([x for x in T if x not in t_new])
len(c)
test_rest=test.loc[c,:]
test_rest.head(50)
test_spanish=test.loc[l4,:]
test_spanish.head()
test_French=test.loc[l5,:]
test_French.head()
lte=l6+c
test_english=test.loc[lte,:]
test_english.head()
test_english.shape,test_French.shape,test_spanish.shape
y_english=train.loc[le,:'Complaint-Status']
y_english.head()
y_French=train.loc[l1,:'Complaint-Status']
y_French.head()
len(l),len(l1),len(le)
y_spanish=train.loc[l,:'Complaint-Status']
y_spanish.head()
y_english.shape,y_French.shape,y_spanish.shape
sum=len(y_english)+len(y_spanish)+len(y_French)
sum
y_english=y_english['Complaint-Status'].values
y_French=y_French['Complaint-Status'].values
y_spanish=y_spanish['Complaint-Status'].values
train.head()
test.head()
test_id=test['Complaint-ID']
train1=train
test1=test
train.drop(['Complaint-ID','Complaint-Status'],axis=1,inplace=True)
test.drop(['Complaint-ID'],axis=1,inplace=True)
tfidf_vec_spanish = TfidfVectorizer(sublinear_tf=True,norm='l2',encoding='latin-1',stop_words='english')
tfidf_vec_spanish.fit_transform(train_spanish['Consumer_complaint_summary'].values.tolist() + test_spanish['Consumer_complaint_summary'].values.tolist())
train_tfidf_spanish = tfidf_vec_spanish.transform(train_spanish['Consumer_complaint_summary'].values.tolist())
test_tfidf_spanish = tfidf_vec_spanish.transform(test_spanish['Consumer_complaint_summary'].values.tolist())
tfidf_vec_French = TfidfVectorizer(sublinear_tf=True,norm='l2',encoding='latin-1',stop_words='english')
tfidf_vec_French.fit_transform(train_French['Consumer_complaint_summary'].values.tolist() + test_French['Consumer_complaint_summary'].values.tolist())
train_tfidf_French = tfidf_vec_French.transform(train_French['Consumer_complaint_summary'].values.tolist())
test_tfidf_French = tfidf_vec_French.transform(test_French['Consumer_complaint_summary'].values.tolist())
tfidf_vec_english = TfidfVectorizer(sublinear_tf=True,norm='l2',encoding='latin-1',stop_words='english')
tfidf_vec_english.fit_transform(train_english['Consumer_complaint_summary'].values.tolist() + test_english['Consumer_complaint_summary'].values.tolist())
train_tfidf_english = tfidf_vec_english.transform(train_english['Consumer_complaint_summary'].values.tolist())
test_tfidf_english = tfidf_vec_english.transform(test_english['Consumer_complaint_summary'].values.tolist())
train_tfidf_spanish
test_tfidf_spanish
test_tfidf_French
test_tfidf_english
train_spanish=train_spanish.drop(['Consumer_complaint_summary','Complaint-Status'],axis=1)
test_spanish=test_spanish.drop(['Consumer_complaint_summary'],axis=1)
train_French=train_French.drop(['Consumer_complaint_summary','Complaint-Status'],axis=1)
test_French=test_French.drop(['Consumer_complaint_summary'],axis=1)
train_english=train_english.drop(['Consumer_complaint_summary','Complaint-Status'],axis=1)
test_english=test_english.drop(['Consumer_complaint_summary'],axis=1)
train_spanish=train_spanish.drop(['Complaint-ID'],axis=1)
test_spanish=test_spanish.drop(['Complaint-ID'],axis=1)
train_French=train_French.drop(['Complaint-ID'],axis=1)
test_French=test_French.drop(['Complaint-ID'],axis=1)
train_english=train_english.drop(['Complaint-ID'],axis=1)
test_english=test_english.drop(['Complaint-ID'],axis=1)
train_spanish.head()
train_features_spanish = hstack([
    train_tfidf_spanish,
    train_spanish,],'csr'
)
train_features_French = hstack([
    train_tfidf_French,
    train_French,],'csr'
)
train_features_english = hstack([
    train_tfidf_english,
    train_english,],'csr'
)
test_features_spanish = hstack([
    test_tfidf_spanish,
    test_spanish,],'csr'
)
test_features_French = hstack([
    test_tfidf_French,
    test_French,],'csr'
)
test_features_english = hstack([
    test_tfidf_english,
    test_english,],'csr'
)
from sklearn.svm import LinearSVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
def multAcc(pred, dtrain):
    label = dtrain.get_label()
    acc = accuracy_score(label, pred)
    return 'maccuracy', acc
params = {'objective':'multi:softmax',
          'num_class':5,
          'eval_metric':'auc',
          'max_depth':6,
         }
dtrain_english = xgb.DMatrix(data=train_features_english, label=y_english)
dtest_english = xgb.DMatrix(data=test_features_english)
#for getting better accuracy increase no of iteration
#I think around 1000.
clfeng = xgb.train(params, dtrain_english,10, maximize=True, feval=multAcc)
y_pred_english=clfeng.predict(dtest_english)
dtrain_spanish = xgb.DMatrix(data=train_features_spanish, label=y_spanish)
dtest_spanish = xgb.DMatrix(data=test_features_spanish)
#for getting better accuracy increase no of iteration.
#I think around 500.
clfspa = xgb.train(params, dtrain_spanish,10, maximize=True, feval=multAcc)
y_pred_spanish=clfspa.predict(dtest_spanish)
dtrain_French = xgb.DMatrix(data=train_features_French, label=y_French)
dtest_French = xgb.DMatrix(data=test_features_French)
#for getting better accuracy increase no of iteration
#I think around 500.
clfFre = xgb.train(params, dtrain_French,10, maximize=True, feval=multAcc)
y_pred_French=clfFre.predict(dtest_French)
y_pred_english.shape,y_pred_French.shape,y_pred_spanish.shape
k=0
for i in y_pred_French:
    if i==0.0:
        k=k+1
k
k1=0
for i in y_pred_english:
    if i==0.0:
        k1=k1+1
k1
k2=0
for i in y_pred_spanish:
    if i==0.0:
        k2=k2+1
k2
sumal=k+k1+k2
sumal
y_pred_english=y_pred_english.astype(int)
y_pred_spanish=y_pred_spanish.astype(int)
y_pred_French=y_pred_French.astype(int)
class_names=['Closed with explanation',
             'Closed with non-monetary relief',
             'Closed',
             'Closed with monetary relief',
              'Untimely response']
a=[]
for i in y_pred_english:
    a.append(class_names[i])
a1=[]
for i in y_pred_spanish:
    a1.append(class_names[i])
a2=[]
for i in y_pred_French:
    a2.append(class_names[i])
sum=(len(a)+len(a1)+len(a2))
sum
combined=a1+a2+a
len(combined)
len(combined)
z=l4+l5+lte
len(z)
keys = z
values = combined
dictionary = dict(zip(keys, values))
data=dict(sorted(dictionary.items()))
data1=pd.DataFrame.from_dict(data, orient='index')
data1['Complaint-ID'] = test_id
#data1 = data[['Complaint-ID','Complaint-Status']]
data1['Complaint-Status']=data1[0]
data1.head()
data1=data1.loc[:,['Complaint-ID','Complaint-Status']]
data1.head()
data1.shape
data1.to_csv("Brainwaves.csv", index=False)
