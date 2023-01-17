# PREDICTING THE EFFECTS OF GENETIC VARIATIONS USING LGBM
# BY - OMKAR SABNIS - 29-05-2018
#Importing library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
# READING THE DATASETS
train = pd.read_csv("../input/training_variants")
trainx = pd.read_csv('../input/training_text',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
train = pd.merge(train, trainx, how = 'left', on = 'ID').fillna('')
train.head()
test = pd.read_csv("../input/stage2_test_variants.csv")
testx = pd.read_csv('../input/stage2_test_text.csv',sep = '\|\|', engine= 'python', header=None, 
                     skiprows=1, names=["ID","Text"])
test = pd.merge(test, testx, how = 'left', on = 'ID').fillna('')
test.head()
train.Gene.nunique()
train['Gene'].unique()

k = train.groupby('Gene')['Gene'].count()

plt.figure(figsize=(12,6))
plt.hist(k, bins=150,log=True)
plt.xlabel('Number of times Gene appared')
plt.ylabel('Log of count')
plt.title('Appearence of gene')
plt.show()

#count Gene
from collections import Counter
plt.figure(figsize=(12,10))
sns.countplot((train['Gene']))
plt.xticks()
genecount = Counter(train['Gene'])
print('Genes and their appearence:')
print(genecount,'\n',len(genecount))

train.Variation.nunique()
train['Variation'].unique()

k = train.groupby('Variation')['Variation'].count()
plt.title('Graph of Gene vs Count')
plt.figure(figsize=(12,6))
def textlen(train):
    k = train['Text'].apply(lambda x: len(str(x).split()))
    l = train['Text'].apply(lambda x: len(str(x)))
    return k, l

train['Text_no_word'], train['Text_no_char'] = textlen(train)
test['Text_no_word'], test['Text_no_char'] = textlen(test)
tfidf = TfidfVectorizer(
	min_df=1, max_features=1600, strip_accents='unicode',lowercase =True,
	analyzer='word', token_pattern=r'\w+', ngram_range=(1, 3), use_idf=True, 
	smooth_idf=True, sublinear_tf=True, stop_words = 'english')
X_train = tfidf.fit_transform(train['Text']).toarray()
print(X_train)
X_test = tfidf.fit_transform(test['Text']).toarray()

def encoding(df,col):
    le = LabelEncoder()
    for i in col:
        df[i] = le.fit_transform(df[i])
train.columns
col = ['Gene', 'Variation', 'Class']
encoding(train,col)
encoding(test,['Gene', 'Variation'])

X_train = pd.DataFrame(X_train)
X_train = X_train.join(train[['Gene', 'Variation', 'Text_no_word','Text_no_char']]) 
X_test = pd.DataFrame(X_test)
X_test = X_test.join(test[['Gene', 'Variation', 'Text_no_word','Text_no_char']])
# FEATURE SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = train['Class']
xtr,xvl,ytr,yvl = train_test_split(X_train,y_train,test_size=0.3,random_state=10)
# NAIVE BAYES
nbc = GaussianNB()
nbc.fit(xtr,ytr)
y_nbcP = nbc.predict(xvl)
y_nbc = nbc.predict_proba(X_test)
print("Confusion Matrix using Naive Bayes:")
print(confusion_matrix(yvl,y_nbcP))
print("\n")

# RANDOM FOREST
rfc = RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=4)
rfc.fit(xtr,ytr)
y_rfcp = rfc.predict(xvl)
y_rfc=rfc.predict_proba(X_test)
print("Confusion Matrix using Random Forest:")
print(confusion_matrix(yvl,y_rfcp))
def runLgb(Xtr,Xvl,ytr,yvl,test,num_rounds=10,max_depth=10,eta=0.5,subsample=0.8,
           colsample=0.8,min_child_weight=1,early_stopping_rounds=50,seeds_val=2017):
    
    param = {'task': 'train',
             'boosting_type': 'gbdt',
             'objective':'multiclass',
             'num_class':9,
             'learning_rate':eta,
             'metric':{'multi_logloss'},
             'max_depth':max_depth,
             #'min_child_weight':min_child_weight,
             'bagging_fraction':subsample,
             'feature_fraction':colsample,
             'bagging_seed':seeds_val,
             'num_iterations': num_rounds, 
             'num_leaves': 95,           
             'min_data_in_leaf': 60, 
             'lambda_l1': 1.0,
             'verbose':10,
             'nthread':-1}
    lgtrain = lgb.Dataset(Xtr,label=ytr)
    lgval = lgb.Dataset(Xvl,label=yvl)
    model = lgb.train(param,lgtrain,num_rounds,valid_sets=lgval,
                      early_stopping_rounds=early_stopping_rounds,verbose_eval=20)
    pred_val = model.predict(Xvl,num_iteration = model.best_iteration)
    pred_test = model.predict(test,num_iteration=model.best_iteration)
    return pred_test,pred_val,model
kf = KFold(n_splits=10,random_state=111,shuffle=True)
cv_score = []
pred_test_full=0

for train_index,test_index in kf.split(X_train):
    Xtr,Xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train[train_index],y_train[test_index]
    
    pred_test,pred_val,model = runLgb(Xtr,Xvl,ytr,yvl,X_test,num_rounds=10,max_depth=3,
                            eta=0.02,)
    pred_test_full +=pred_test
pred_test = pred_test_full/10
# SUBMISSION OF FILE IN CSV FORMAT:
submit = pd.DataFrame(test.ID)
submit = submit.join(pd.DataFrame(pred_test))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
submit.to_csv('submission.csv', index=False) 