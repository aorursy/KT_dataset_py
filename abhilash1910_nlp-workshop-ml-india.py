# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#see the input

!ls ../input
!ls ../input/quora-insincere-questions-classification
#import libraries useful for the entire module
import pandas as pd 
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD,SparsePCA
from sklearn.metrics import classification_report,confusion_matrix
from nltk.tokenize import word_tokenize
from collections import defaultdict
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import nltk
from nltk.corpus import stopwords
import string
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn import preprocessing,metrics,manifold
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_val_predict
from imblearn.over_sampling import ADASYN,SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import collections
import keras as k
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
%matplotlib inline
from sklearn.preprocessing import RobustScaler
import xgboost
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,r2_score,recall_score,confusion_matrix,precision_recall_curve
from collections import Counter
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedShuffleSplit

train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
print(train_df.head())
print(test_df.head())
print("===========")
print("Training Shape".format(),train_df.shape)
print("Testing Shape".format(),test_df.shape)
print("The type of columns in the dataset")
print("Columns".format(),train_df.columns)

#Insincere question test
train_ext=train_df[train_df['target']==1]['question_text']
print(train_ext)


#Training data statistics

#Estimate the value counts
count_types=train_df['target'].value_counts()
print("Extracting counts".format(),count_types)
#Count targets with value 1

count_ones=train_df[train_df['target']==1].shape[0]
print(count_ones)
count_zeros=train_df[train_df['target']==0].shape[0]
print(count_zeros)

#Matplot to plot the amount of questions from either types
def plot_counts(count_ones,count_zeros):
    plt.rcParams['figure.figsize']=(6,6)
    plt.bar(0,count_ones,width=0.6,label='InSincere Questions',color='Red')
    plt.legend()
    plt.bar(2,count_zeros,width=0.6,label='Sincere Questions',color='Green')
    plt.legend()
    plt.ylabel('Count of Questions (in M)')
    plt.xlabel('Types of Questions')
    plt.show()

    
#Seaborn Dist plot for analysing the length of a question sentence    
def plot_wordcount(count_ones_length,count_zeros_length):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    sns.distplot(count_zeros_length,ax=ax1,color='Blue')
    ax1.set_title('Sincere Question Length')
    sns.distplot(count_ones_length,ax=ax2,color='Red')
    ax2.set_title('Insincere Question Length')
    fig.suptitle('Average Length of Words in Question')
    plt.show()    

#Generic Plotter
def plot_count(count_punct_ones,count_punct_zeros,title_1,title_2,subtitle):
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
    sns.distplot(count_punct_zeros,ax=ax1,color='Blue')
    ax1.set_title(title_1)
    sns.distplot(count_punct_ones,ax=ax2,color='Red')
    ax2.set_title(title_2)
    fig.suptitle(subtitle)
    plt.show()    


#preliminary word cloud statistics
def display_cloud(data):
    stopwords=set(STOPWORDS)
    wordcloud=WordCloud(stopwords=stopwords,max_font_size=120,max_words=300,width=800,height=400,background_color='white',min_font_size=5).generate(str(data))
    plt.figure(figsize=(24,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Word Cloud of the questions")
    plt.show()


    
    
#Extract the length of the insincere-sincere questions
def word_length(x):
    return len(x)


count_ones_length=train_df[train_df['target']==1]['question_text'].str.split().apply(lambda z:word_length(z))
print("Length of each insincere questions".format(),count_ones_length[:5])
count_zeros_length=train_df[train_df['target']==0]['question_text'].str.split().apply(lambda z: word_length(z))
print("Length of each sincere questions".format(),count_zeros_length[:5])

#Plots
plot_counts(count_ones,count_zeros)
plot_wordcount(count_ones_length,count_zeros_length)
display_cloud(train_df['question_text'])
display_cloud(train_df[train_df['target']==1]['question_text'])
display_cloud(train_df[train_df['target']==0]['question_text'])
#Other analysis
stops=set(stopwords.words('english'))
count_punct_ones=train_df[train_df['target']==1]['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
count_punct_zeros=train_df[train_df['target']==0]['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
title_1='Sincere Question Punctuations'
title_2='Insincere Question Punctuations'
subtitle='Punctuations in Questions'
plot_count(count_punct_ones,count_punct_zeros,title_1,title_2,subtitle)

count_avg_ones=train_df[train_df['target']==1]['question_text'].apply(lambda z : np.mean([len(z) for w in str(z).split()]))
count_avg_zeros=train_df[train_df['target']==0]['question_text'].apply(lambda z: np.mean([len(z) for w in str(z).split()]))
title_1='Insincere Question Average Length'
title_2='Sincere Question Average Length'
subtitle='Average Length'
plot_count(count_avg_ones,count_avg_zeros,title_1,title_2,subtitle)
#Test dataset Word Statistics

def plot_testcount(count_test,title1,subtitle):
    fig,(ax1)=plt.subplots(1,figsize=(15,5))
    sns.distplot(count_test,ax=ax1,color='Orange')
    ax1.set_title(title_1)
    fig.suptitle(subtitle)
    plt.show()    

count_test=test_df['question_text'].str.split().apply(lambda z:word_length(z))
print("Length of each test questions".format(),count_test[:5])

#Plots
plot_testcount(count_test,'Questions in test','Total Questions')
display_cloud(test_df['question_text'])
#Other analysis
stops=set(stopwords.words('english'))
count_puncttest=test_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
title_1='Test Question Punctuations'
subtitle='Punctuations in Questions'
plot_testcount(count_puncttest,title_1,subtitle)

count_avg_test=test_df['question_text'].apply(lambda z : np.mean([len(z) for w in str(z).split()]))
title_1='Test Question Average Length'
subtitle='Average Length'
plot_testcount(count_avg_test,title_1,subtitle)
#Gram analysis on Training set
stopword=set(stopwords.words('english'))
def gram_analysis(data,gram):
    tokens=[t for t in data.lower().split(" ") if t!="" if t not in stopword]
    ngrams=zip(*[tokens[i:] for i in range(gram)])
    final_tokens=[" ".join(z) for z in ngrams]
    return final_tokens

#analyse most common Sentences
def mostcommon_words(data):
    counter=Counter(data)
    commonwords=counter.most_common()
    x_coord,y_coord=[],[]
    for words,occ in commonwords[:20]:
        if words not in stopword:
            x_coord.append(occ)
            y_coord.append(words)
            
    sns.barplot(x=x_coord,y=y_coord,saturation=1,orient="h")

#Create frequency grams for analysis
    
def create_dict(data,grams):
    freq_dict=defaultdict(int)
    for sentence in data:
        for tokens in gram_analysis(sentence,grams):
            freq_dict[tokens]+=1
    return freq_dict

def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["n_gram_words"].values[::-1],
        x=df["n_gram_frequency"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace



def create_new_df(freq_dict,):
    freq_df=pd.DataFrame(sorted(freq_dict.items(),key=lambda z:z[1])[::-1])
    freq_df.columns=['n_gram_words','n_gram_frequency']
    #print(freq_df.head())
    #plt.barh(freq_df['n_gram_words'][:20],freq_df['n_gram_frequency'][:20],linewidth=0.3)
    #plt.show()
    trace=horizontal_bar_chart(freq_df[:20],'blue')
    return trace
    
def plot_grams(trace_zero,trace_one):
    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
    fig.append_trace(trace_zero, 1, 1)
    fig.append_trace(trace_ones, 1, 2)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
    py.iplot(fig, filename='word-plots')
    
    
train_df_zero=train_df[train_df['target']==0]['question_text']
train_df_ones=train_df[train_df['target']==1]['question_text']

#mostcommon_words(train_df_zero)
print("Bi-gram analysis")
freq_train_df_zero=create_dict(train_df_zero[:200],2)
#print(freq_train_df_zero)
trace_zero=create_new_df(freq_train_df_zero)
freq_train_df_ones=create_dict(train_df_ones[:200],2)
#print(freq_train_df_zero)
trace_ones=create_new_df(freq_train_df_ones)
plot_grams(trace_zero,trace_ones)


print("Tri-gram analysis")
freq_train_df_zero=create_dict(train_df_zero[:200],3)
#print(freq_train_df_zero)
trace_zero=create_new_df(freq_train_df_zero)
freq_train_df_ones=create_dict(train_df_ones[:200],3)
#print(freq_train_df_zero)
trace_ones=create_new_df(freq_train_df_ones)
plot_grams(trace_zero,trace_ones)

#Sentence Analysis
mostcommon_words(train_df_zero)
#Sentence Analysis
mostcommon_words(train_df_ones)
#Similar gram analysis using nltk after regexing

import re
from nltk.util import ngrams
def gram_analyse(s):
    
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, 5))
    return output

for j in range(0,10):

    out=gram_analyse(train_df_ones.iloc[j])
    print("Output grams",out)


#Regex cleaning

def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data
def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

def remove_url(data):
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data
def clean_data(data):
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    data=emoji_clean.sub(r'',data)
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data


train_df['question_text']=train_df['question_text'].apply(lambda z : remove_url(z))
train_df['question_text']=train_df['question_text'].apply(lambda z: clean_data(z))
train_df['question_text']=train_df['question_text'].apply(lambda z: remove_html(z))
train_df['question_text']=train_df['question_text'].apply(lambda z: remove_punctuations(z))

print("Cleaned Train Insincere Question Set")
print(train_df[train_df['target']==1]['question_text'].head())
print("Cleaned Train Sincere Question Set")
print(train_df[train_df['target']==0]['question_text'].head())

#Lemmatizing the corpus as a backup

from nltk.stem import WordNetLemmatizer


def lemma_traincorpus(data):
    lemmatizer=WordNetLemmatizer()
    out_data=""
    for words in data:
        out_data+= lemmatizer.lemmatize(words)
    return out_data

train_df['question_text']=train_df['question_text'].apply(lambda z: lemma_traincorpus(z))
print(train_df.head())
#tfidf vectorization

tfidf_vect=TfidfVectorizer(stop_words='english',ngram_range=(1,3))
#tfidf_vect.fit_transform(train_df['question_text'].values.tolist() + test_df['question_text'].values.tolist())
train_tfidf=tfidf_vect.fit_transform(train_df['question_text'].values.tolist())
test_tfidf=tfidf_vect.fit_transform(test_df['question_text'].values.tolist())
print(train_tfidf)

#count vectorization
def count_vectorize(train_data,test_data):
    count_vectorize=CountVectorizer(stop_words='english',ngram_range=(1,3),analyzer='word',token_pattern='r\w{1,}')
    count_vectorize.fit_transform(train_data['question_text'].values.tolist() + test_data['question_text'].values.tolist())
    train_count=count_vectorize.transform(train_data['question_text'].values.tolist())
    test_count=count_vectorize.transform(test_data['question_text'].values.tolist())
    return train_count,test_count


train_y=train_df['target']

train_x=train_df['question_text']
print(train_y)
print(train_x)
#Some preliminary models for sequential evaluation on the training set

#train_y=train_df['target'].values

# models=[]
# models.append(('LR',LogisticRegression()))
# models.append(('KNN',KNeighborsClassifier()))
# models.append(('LDA',LinearDiscriminantAnalysis()))
# models.append(('DT',DecisionTreeClassifier()))
# #models.append(('SVC',SVC()))
# model_result=[]
# scoring='accuracy'
# for name,model in models:
#     kfold=KFold(n_splits=10,random_state=7)
#     results=cross_val_score(model,train_tfidf,train_y,cv=kfold,scoring=scoring)
#     print("Classifiers: ",name, "Has a training score of", round(results.mean(), 2) * 100, "% accuracy score")
#     model_result.append(results.mean())

#split the training set into train and test evaluation sets
print(train_tfidf.shape)

train_x,test_x,train_y,test_y=train_test_split(train_tfidf,train_y,test_size=0.2,random_state=42)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
# models=[]
# #models.append(('LR',LogisticRegression()))
# models.append(('KNN',KNeighborsClassifier()))
# models.append(('LDA',LinearDiscriminantAnalysis()))
# models.append(('DT',DecisionTreeClassifier()))

# #Now trying out different stats models
# def model_training(model,train_x,test_x,train_y,test_y):
#     model.fit(train_x,train_y)
#     pred=model.predict(test_x)
#     print("Evaluate confusion matrix")
#     print(confusion_matrix(test_y,pred))
#     print(accuracy_score(test_y,pred))
#     return accuracy_score(test_y,pred)
    
# for name,mods in models:
#     accuracy_score=model_training(mods,train_x,test_x,train_y,test_y)
#     print(accuracy_score)
    

#Logistic Regression
model=LogisticRegression()
model.fit(train_x,train_y)
pred=model.predict(test_x)
print("Evaluate confusion matrix")
print(confusion_matrix(test_y,pred))
print(accuracy_score(test_y,pred))
    
#Naive Bayes

model=MultinomialNB()
model.fit(train_x,train_y)
pred=model.predict(test_x)
print("Evaluate confusion matrix")
print(confusion_matrix(test_y,pred))
print(accuracy_score(test_y,pred))
    
#LDA- Linear Disciminant Analysis

model=LinearDiscriminantAnalysis()
x_train=train_x.toarray()
model.fit(x_train,train_y)
pred=model.predict(test_x)
print("Evaluate confusion matrix")
print(confusion_matrix(test_y,pred))
print(accuracy_score(test_y,pred))
    
#XGBoost
from xgboost import XGBClassifier as xg
model_xgb= xg(n_estimators=100,random_state=42)
model_xgb.fit(train_x,train_y)
y_pred_xgb=model_xgb.predict(test_x)
print("Confusion matrix")

print(confusion_matrix(test_y,y_pred_xgb))
print(accuracy_score(test_y,y_pred_xgb.round()))
#LightGBM
from lightgbm import LGBMClassifier as lg
model_lgbm= lg(n_estimators=100,random_state=42)
model_lgbm.fit(train_x,train_y)
y_pred_lgbm=model_lgbm.predict(test_x)
print("Confusion matrix")
print(confusion_matrix(test_y,y_pred_lgbm))
print(accuracy_score(test_y,y_pred_lgbm.round()))
#Decision Trees
model_dt=DecisionTreeClassifier(random_state=42)
model_dt.fit(train_x,train_y)
y_pred=model_dt.predict(test_x)
print("Confusion matrix")
print(confusion_matrix(test_y,y_pred))
print(accuracy_score(test_y,y_pred.round()))

#Random Forest
model_dt=RandomForestClassifier(random_state=42)
model_dt.fit(train_x,train_y)
y_pred=model_dt.predict(test_x)
print("Confusion matrix")
print(confusion_matrix(test_y,y_pred))
print(accuracy_score(test_y,y_pred.round()))

def dimen_reduc_plot(test_data,test_label,option):
    tsvd= TruncatedSVD(n_components=2,algorithm="randomized",random_state=42)
    tsne=TSNE(n_components=2,random_state=42) #not recommended instead use PCA
    pca=SparsePCA(n_components=2,random_state=42)
    if(option==1):
        tsvd_result=tsvd.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        
        sns.scatterplot(x=tsvd_result[:,0],y=tsvd_result[:,1],hue=test_label        )
        
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(tsvd_result[:,0],tsvd_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_Tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSVD")
        plt.show()
    if(option==2):
        tsne_result=tsne.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=tsne_result[:,0],y=tsne_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=tsne_result[:,0],y=tsne_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("PCA")
        plt.show() 
    if(option==3):
        pca_result=pca.fit_transform(test_data.toarray())
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=pca_result[:,0],y=pca_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=pca_result[:,0],y=pca_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSNE")
        plt.show()

dimen_reduc_plot(train_x,train_y,1)
dimen_reduc_plot(train_x,train_y,3)
dimen_reduc_plot(train_x,train_y,2)

def dimen_reduc_plot(test_data,test_label,option):
    tsvd= TruncatedSVD(n_components=2,algorithm="randomized",random_state=42)
    tsne=TSNE(n_components=2,random_state=42) #not recommended instead use PCA
    pca=SparsePCA(n_components=2,random_state=42)
    if(option==1):
        tsvd_result=tsvd.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        
        sns.scatterplot(x=tsvd_result[:,0],y=tsvd_result[:,1],hue=test_label        )
        
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(tsvd_result[:,0],tsvd_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_Tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSVD")
        plt.show()
    if(option==2):
        tsne_result=tsne.fit_transform(test_data)
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=tsne_result[:,0],y=tsne_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=tsne_result[:,0],y=tsne_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("PCA")
        plt.show() 
    if(option==3):
        pca_result=pca.fit_transform(test_data.toarray())
        plt.figure(figsize=(10,8))
        colors=['orange','red']
        sns.scatterplot(x=pca_result[:,0],y=pca_result[:,1],hue=test_label)
        plt.show()
        plt.figure(figsize=(10,10))
        plt.scatter(x=pca_result[:,0],y=pca_result[:,1],c=test_label,cmap=matplotlib.colors.ListedColormap(colors))
        color_red=mpatches.Patch(color='red',label='False_tweet')
        color_orange=mpatches.Patch(color='orange',label='Real_Tweet')
        plt.legend(handles=[color_orange,color_red])
        plt.title("TSNE")
        plt.show()

dimen_reduc_plot(test_x,test_y,1)
dimen_reduc_plot(test_x,test_y,3)
dimen_reduc_plot(test_x,test_y,2)

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense,Flatten,Conv2D,Conv1D,GlobalMaxPooling1D
from keras.optimizers import Adam
#import MiniAttention.MiniAttention as ma
import numpy as np  
import pandas as pd 
import re           
from bs4 import BeautifulSoup 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
# Load the input features
train_df=pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
test_df=pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
train_set,test_set=train_test_split(train_df,test_size=0.2,random_state=2017)
print(train_set.shape)
print(test_set.shape)
maxlen=1000
max_features=5000 
embed_size=300

#clean some null words or use the previously cleaned & lemmatized corpus

train_x=train_set['question_text'].fillna('_na_').values
val_x=test_set['question_text'].fillna('_na_').values

#Tokenizing steps- must be remembered
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_x))
train_x=tokenizer.texts_to_sequences(train_x)
val_x=tokenizer.texts_to_sequences(val_x)

#Pad the sequence- To allow same length for all vectorized words
train_x=pad_sequences(train_x,maxlen=maxlen)
val_x=pad_sequences(val_x,maxlen=maxlen)


#get the target values - either using values or using Label Encoder
train_y=train_set['target'].values
val_y=test_set['target'].values
print("Padded and Tokenized Training Sequence".format(),train_x.shape)
print("Target Values Shape".format(),train_y.shape)
print("Padded and Tokenized Training Sequence".format(),val_x.shape)
print("Target Values Shape".format(),val_y.shape)
print(train_x[2])
#Creating a basic model- without pretrained embeddings
model=Sequential()
model.add(Embedding(max_features,embed_size,input_length=maxlen))
model.add(LSTM(60))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
#Fit the model with the inputs

model.fit(train_x,train_y,batch_size=512,epochs=2,verbose=2)
#fit and validate together
model.fit(train_x,train_y,batch_size=128,epochs=3,verbose=2,validation_data=(val_x,val_y))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import math
from tqdm import tqdm
!pip install --upgrade keras
#CuDNN uses the Nvidia GPU for faster training

inp=Input(shape=(maxlen,))
z=Embedding(max_features,embed_size)(inp)
z=Bidirectional(LSTM(60,return_sequences='True'))(z)
z=GlobalMaxPool1D()(z)
z=Dense(16,activation='relu')(z)
z=Dense(1,activation='sigmoid')(z)
model=Model(inputs=inp,outputs=z)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#fit and validate together
model.fit(train_x,train_y,batch_size=512,epochs=1,verbose=2,validation_data=(val_x,val_y))

#see the input

!ls ../input
!ls ../input/quora-insincere-questions-classification
!ls ../input/wikinews300d1msubwordvec

#visualize the embeddings
plt.plot(embedding_matrix[10])
plt.plot(embedding_matrix[5])
# Rebuilding the model with pre-trained embeddings from Glove

inp=Input(shape=(maxlen,))
z=Embedding(max_features,embed_size,weights=[embedding_matrix])(inp)
z=Bidirectional(LSTM(60,return_sequences='True'))(z)
z=GlobalMaxPool1D()(z)
z=Dense(16,activation='relu')(z)
z=Dense(1,activation='sigmoid')(z)
model=Model(inputs=inp,outputs=z)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#fit and validate together- with glove embedding
model.fit(train_x,train_y,batch_size=512,epochs=1,verbose=2,validation_data=(val_x,val_y))



#Fast Text Embeddings for training 
EMBEDDING_FILE = '../input/wikinews300d1msubwordvec/wiki-news-300d-1M-subword.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#visualize the embeddings-FastText
plt.plot(embedding_matrix[10])
plt.plot(embedding_matrix[5])
inp=Input(shape=(maxlen,))
z=Embedding(max_features,embed_size,weights=[embedding_matrix])(inp)
z=Bidirectional(LSTM(60,return_sequences='True'))(z)
z=GlobalMaxPool1D()(z)
z=Dense(16,activation='relu')(z)
z=Dense(1,activation='sigmoid')(z)
model=Model(inputs=inp,outputs=z)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#fit and validate together- with glove embedding
model.fit(train_x,train_y,batch_size=512,epochs=1,verbose=2,validation_data=(val_x,val_y))
! ls ../input

!ls ../input/googlenewsvectorsnegative300
!ls ../input/paragram-300-sl999
! ls ../input/word2vec-sample
!ls ../input



#Paragram Embeddings for training 
EMBEDDING_FILE = '../input/paragram-300-sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        
def add_corpus(data):
    corpus=[]
    for i in tqdm(data):
        words=[word for word in (i)]
        corpus.append(words)
    return corpus
def create_embedding(data):
    embedding_map={}
    file=open('../input/paragram-300-sl999/paragram_300_sl999.txt','r')
    for  f in file:
        values=f.split(' ')
        word=values[0]
        coef=np.asarray(values[1:],dtype='float32')
        embedding_map[word]=coef
    file.close()
    return embedding_map
def  embedding_preprocess(data):
    #max_word_length=1000
    max_sequence_length=100
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(data)
    sequences=tokenizer.texts_to_sequences(data)
    
    word_idx=tokenizer.word_index
    data_pad=pad_sequences(sequences,padding="post",maxlen=max_sequence_length)
    emb_dim=data.get('a').shape[0]
    num_length=len(word_idx)+1
    emb_mat=np.zeros((num_length,emb_dim))
    for word,idx in tqdm(word_idx.items()):
        if idx > num_length:
            continue
        elif idx < num_length:
            emb_vector=data.get(word)
            if emb_vector is not None: 
                emb_mat[idx]=emb_vector
    
    return emb_mat,word_idx,data_pad,num_length
    
    
corpus_train_data=add_corpus(train_x)
print("corpus created")

embedding_map= create_embedding(corpus_train_data)
print("Embedding matrix created")
emb_mat,word_idx,pad_data,num_words=embedding_preprocess(embedding_map)
print(pad_data.shape)

print("Visualise embedded vectors")
plt.plot(emb_mat[10])
plt.plot(emb_mat[20])
plt.plot(emb_mat[50])
plt.title("Embedding Vectors")
plt.show()
from gensim.models import KeyedVectors
EMBEDDING_FILE = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = (np.random.rand(nb_words, embed_size) - 0.5) / 5.0
for word, i in word_index.items():
    if i >= max_features: continue
    if word in embeddings_index:
        embedding_vector = embeddings_index.get_vector(word)
        embedding_matrix[i] = embedding_vector
inp=Input(shape=(maxlen,))
z=Embedding(max_features,embed_size,weights=[embedding_matrix])(inp)
z=Bidirectional(LSTM(60,return_sequences='True'))(z)
z=GlobalMaxPool1D()(z)
z=Dense(16,activation='relu')(z)
z=Dense(1,activation='sigmoid')(z)
model=Model(inputs=inp,outputs=z)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#fit and validate together- with glove embedding
model.fit(train_x,train_y,batch_size=512,epochs=1,verbose=2,validation_data=(val_x,val_y))
# Word2vec examples

from gensim.models import Word2Vec

model = Word2Vec.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)


dog = model['dog']

print(model.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))

print(model.similarity('woman', 'man'))