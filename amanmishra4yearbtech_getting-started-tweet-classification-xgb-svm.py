import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from collections import Counter
from nltk import PorterStemmer, SnowballStemmer, WordNetLemmatizer 
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import  MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder as le
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv('../input/nlp-getting-started/train.csv')
test_data = pd.read_csv('../input/nlp-getting-started/test.csv')
sample_submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

train_data
# list down some elements of test data
test_data.head(5)
sample_submission.head(5)
# using the pandas groupby function to calculate samples per labels

count_data = train_data[['text','target']].groupby('target').count().reset_index()
count_data
# plotting instances per labels

sns.barplot(x = 'target',y = 'text', data = count_data)
plt.title('no. of instances vs labels')
## merge train and test data to perform common operations on both
## due to merging an extra column of target will form, which contains nan values

dataset = pd.concat([train_data, test_data])
print(dataset.shape)
dataset.head(5)
# checking null values in both entire dataset and train data
print(dataset.isnull().sum())
print('-'*100)
print(train_data.isnull().sum())
# checking null values in both entire dataset and train data
print(dataset.info())
print('-'*100)
print(train_data.info())
dataset['len_letters'] = dataset['text'].apply(len)
train_data['len_letters'] = train_data['text'].apply(len)
dataset.head(5)
fig,axes = plt.subplots(ncols = 2)
sns.distplot(train_data['len_letters'][train_data['target'] == 1 ],label = 'disaster',color = 'r' ,ax = axes[0]) ## denoting disaster tweets length
sns.distplot(train_data['len_letters'][train_data['target'] == 0 ],label = 'non-disaster',color = 'g', ax= axes[1] ) ## denoting not disaster tweets length
# getting length of data corresponding to every label
train_data[train_data['target']==1].describe()  # for disastorous label

train_data[train_data['target']==0].describe()  # for non-disastorous label
# filling every unknown keyword with not_known
dataset['keyword'] = dataset['keyword'].fillna('not_known')
dataset['text'][dataset['keyword'] == 'blaze'] 
train_data['location'].isnull().sum()
# now individual label by label
print('For disastrous, unknown locations : ',train_data['location'][train_data['target']==1].isnull().sum())
print('For non disastrous, unknown locations : ',train_data['location'][train_data['target']==0].isnull().sum())

train_data['location'][train_data['target']==1].sample(5) # randomly seen some samples origin location
# filling localation with unknown
dataset['location'].fillna('unknown', inplace = True)
train_data['location'].fillna('unknown', inplace = True)
train_data['location'].sample(5)
# now encoding labels using their frequency wise ratio for locations, may be the no. of times a place occurs may affect the tweet
num_locations = Counter(dataset['location'])
num_locations


# label encoding
dataset['location'] = le().fit_transform(dataset['location'])
dataset
#now time for analysing the text

#converting the text into lower case
dataset['text'] = dataset['text'].map(lambda x: x.lower())
train_data['text'] = train_data['text'].map(lambda x: x.lower())

dataset['text']
# dividing training data text on the basis of labels and then find common words in it

dis_text = train_data['text'][train_data['target']==1]
dis_text

ndis_text = train_data['text'][train_data['target']==0]
ndis_text
dis_count = dis_text.map(lambda x: nltk.word_tokenize(x)) # for disastrous 
ndis_count = ndis_text.map(lambda x : nltk.word_tokenize(x)) # for non disastrous
dataset['text'] = dataset['text'].map(lambda x : nltk.word_tokenize(x)) # for whole dataset tokenize
ndis_count
# let's investigate which words are more common in each cases
from nltk import FreqDist
# for target =1
collection_words_dist = []
for i in list(dis_count):
    collection_words_dist.extend(i)
    
map_1 = FreqDist(collection_words_dist)
map_1
# plot some of them
plt.figure(figsize = (8,8))
sns.barplot(x = list(dict(map_1.most_common(15)).keys()), y = list(dict(map_1.most_common(15)).values())) 
# for target =0
collection_words_ndist = []
for i in list(ndis_count):
    collection_words_ndist.extend(i)
    
map_0 = FreqDist(collection_words_ndist)
map_0
# plot some of them
plt.figure(figsize = (8,8))
sns.barplot(x = list(dict(map_0.most_common(15)).keys()), y = list(dict(map_0.most_common(15)).values())) 
# lematiing the text to obtain root level text
## lematization is beneficial if appropriate pos tags is used 
# function to map nltk tags with wordnet tags
lemmatizer = WordNetLemmatizer()
def nltk_tags_2_word_tags(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
    
# function to create lemmatized sentences from tokenized words
def lemmatized_sentences(tokenized_sentence):
    pos = nltk.pos_tag(tokenized_sentence)  # returns a tuple of words with their nltk tags
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tags_2_word_tags(x[1])), pos)
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
    
  

# now converting our tokenized lowered words into lemmatized sentences

dis_lem = dis_count.apply(lemmatized_sentences)
ndis_lem = ndis_count.apply(lemmatized_sentences)
dataset['text'] = dataset['text'].apply(lemmatized_sentences)

# finally dataset is lemmatized let's see
dataset

dis_lem 
import re   #regex library 

# function to remove patterns
def remove_pattern(input_txt, pattern):
    reg_obj = re.compile(pattern)
    input_txt = reg_obj.sub(r'', input_txt)
        
    return input_txt   

dataset['text'] = dataset['text'].apply(lambda x: remove_pattern(x,"@[\w]*"))
# Reference : https://www.kaggle.com/shahules/tweets-complete-eda-and-basic-modeling


dataset['text'] = dataset['text'].apply(lambda x: remove_pattern(x,'https?://\S+|www\.\S+'))
dataset['text'] = dataset['text'].apply(lambda x: remove_pattern(x,'<.*?>'))
    
dataset['text'] = dataset['text'].apply(lambda x: remove_pattern(x,"[^a-zA-Z# ]"))
# now using tf-idf to create new words using bigram + unigram

tfidf = TfidfVectorizer(ngram_range = (1,1),max_df=0.90, min_df=2,stop_words = 'english')
text_set = tfidf.fit_transform(dataset['text'])
text_set
from scipy.sparse import hstack
dataset_dtm = hstack((text_set,np.array(dataset['location'])[:,None]))

dataset_dtm=text_set
dataset_dtm
dataset_dtm = dataset_dtm.tocsr()  # converting to sparse row format
x_train = dataset_dtm[0:len(train_data)]
x_test = dataset_dtm[len(train_data):]
x_train.shape
x_train
y_train = train_data['target']
len(y_train)
kfold = StratifiedKFold(n_splits = 5 )

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(kernel = 'rbf',probability = True))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
#classifiers.append(LGBMClassifier(objective='classification', random_state=random_state))
#classifiers.append(AdaBoostClassifier(ExtraTreesClassifier(random_state=2,max_depth = None,min_samples_split= 2,min_samples_leaf = 1,bootstrap = False,n_estimators =320), random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(XGBClassifier(random_state=random_state))
#classifiers.append(LinearDiscriminantAnalysis())
"""

cv_results = []
for classifier in classifiers :
    cv_results.append(-cross_val_score(classifier,x_train, y = y_train, scoring = 'accuracy', cv = kfold , n_jobs =-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVR","DecisionTree","lgbm",
"RandomForest","ExtraTrees","GradientBoosting","KNeighboors","LogisticRegression","xgboost","LDA"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_res
"""

parameter = {'solver':['liblinear','lbfgs'],
            'max_iter':[200,400]}

Logis_clf = LogisticRegression()

lreg = GridSearchCV(Logis_clf, param_grid = parameter, cv = 3, verbose=True, n_jobs=-1)
lreg.fit(x_train, y_train) # training the model

lreg_best = lreg.best_estimator_

print(lreg.best_score_)
print(lreg.best_params_)
# feeding raw models into stacking ensemble as the metal model will extract tht best out of each one
from vecstack import stacking
from sklearn.metrics import accuracy_score,f1_score

S_train, S_test = stacking(classifiers,                   
                           x_train, y_train, x_test,   
                           regression= False,
                          
     
                           mode='oof_pred_bag', 
       
                           needs_proba=True,
         
                           save_dir=None, 
             
    
                           n_folds=5, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)
S_train
S_train.shape
argmax_train = []
argmax_test = []
for i in range(0,S_train.shape[1],2):
    argmax_train.append( np.argmax(S_train[:,i:i+2],axis=1))
    argmax_test.append( np.argmax(S_test[:,i:i+2],axis=1))
argmax_train = np.array(argmax_train,dtype= np.int64).T
argmax_test = np.array(argmax_test,dtype= np.int64).T

argmax_train
argmax_train.shape
# here using overall probabilities for meta model
## from sklearn.metrics import f1_score
modelc = LogisticRegression()
    
model1c = modelc.fit(S_train, y_train)
y_pred1c = model1c.predict_proba(S_train)
y_predc = model1c.predict_proba(S_test)

print('Final test prediction score: [%.8f]' % accuracy_score(y_train, np.argmax(y_pred1,axis=1)))
print('Final f1-score test prediction: [%.8f]' % f1_score(y_train, np.argmax(y_pred1,axis=1)))

# here using predictions for metal model
## from sklearn.metrics import f1_score
model = XGBClassifier(random_state=2, objective = 'reg:linear', n_jobs=-1, learning_rate= 0.5, 
                      n_estimators=30, max_depth=20)
    
model1 = model.fit(argmax_train, y_train)
y_pred1 = model1.predict_proba(argmax_train)
y_pred = model1.predict_proba(argmax_test)

print('Final test prediction score: [%.8f]' % accuracy_score(y_train, np.argmax(y_pred1,axis=1)))
print('Final f1-score test prediction: [%.8f]' % f1_score(y_train, np.argmax(y_pred1,axis=1)))

## checking the distribution of prediction
sns.distplot(y_pred)
sns.distplot(y_predc)
             
sample_submission['target'] = np.argmax(y_pred+y_predc,axis=1)

sample_submission.to_csv('submission_with_stacking.csv', index = False)