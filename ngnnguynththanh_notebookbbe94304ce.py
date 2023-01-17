import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



import os
import string
import gc

path = '/kaggle/input/isods2020/'
os.listdir(path)
train_data = pd.read_csv(path + 'training_data_sentiment.csv')
test_data = pd.read_csv(path + 'testing_data_sentiment.csv')
print('Size of train_data: ',train_data.shape)
print('Size of test_data: ',test_data.shape)
train_data.head()
test_data.head()
train_data['is_unsatisfied'].value_counts()
train_data['is_unsatisfied'].hist()
def Q_and_A(idx, data):
    print(train_data.question[idx])
    print(train_data.answer[idx])
    q = train_data.question[idx].split('|||')
    a = train_data.answer[idx].split('|||')
    if q[0] == '': q = q[1:]
    print(len(q) , '-' , len(a))
    for i in range(min(len(q),len(a))):
        print(q[i])
        print(a[i])
        print('\n')
Q_and_A(1, train_data)
Q_and_A(2,train_data)
def clean_text(text):
    
    table = str.maketrans('','',string.punctuation)
    text = text.lower().split()
    text = [word.translate(table) for word in text]
    text = ' '.join(text)
    
    return text

def clean_data(messages):
    
    messages = messages.split('|||')
    messages = map(clean_text,messages)
    messages = '|||'.join(messages)
    
    return messages

train_data['Where'] = 'train'
test_data['Where'] ='test'
data = train_data.append(test_data)
data
num_data = []
data['clear_ques'] = [clean_data(x) for x in data.question]
data['join_ques'] = [x.replace('|||',' ') for x in data.clear_ques]
data['len_ques'] = [len(x.split('|||'))/10 for x in data.clear_ques]
data['max_word_ques'] = [max([len(words.split()) for words in x.split('|||')])/10 for x in data.clear_ques ]
data['min_word_ques'] = [min([len(words.split()) for words in x.split('|||')])/10 for x in data.clear_ques ]
data['mean_word_ques'] = [sum([len(words.split()) for words in x.split('|||')])/len([words for words in x.split('|||')])/10 for x in data.clear_ques]
                         
data['clear_ans'] = [clean_data(x) for x in data.answer]
data['join_ans'] = [x.replace('|||',' ') for x in data.clear_ans]
data['len_ans'] = [len(x.split('|||'))/10 for x in data.clear_ans]
data['max_word_ans'] = [max([len(words.split()) for words in x.split('|||')])/10 for x in data.clear_ans ]
data['min_word_ans'] = [min([len(words.split()) for words in x.split('|||')])/10 for x in data.clear_ans ]
data['mean_word_ans'] = [sum([len(words.split()) for words in x.split('|||')])/len([words for words in x.split('|||')])/10 for x in data.clear_ans]

data['Q_A_len_diff'] = list((np.array(data['len_ques'])- np.array(data['len_ans']))/(np.array(data['len_ques'])+np.array(data['len_ans'])/2)*10)                        
data['label'] = data['is_unsatisfied'].apply(lambda x: 1 if x == 'Y' else 0)
num_data += ['len_ques', 'max_word_ques', 'min_word_ques', 'mean_word_ques',
             'len_ans', 'max_word_ans', 'min_word_ans', 'mean_word_ans','Q_A_len_diff']
columns = ['ques','ans']
n_tfidf = 500

for i, column in enumerate(columns):
    
    tfidf = TfidfVectorizer(max_features= n_tfidf)
    data[[column + str(i) for i in range(n_tfidf)]] = tfidf.fit_transform(data['join_'+ column]).toarray()
    num_data += [column+ str(i) for i in range(n_tfidf)]
    
#     del tfidf
data.head()
train_data = data[data.Where == 'train']
test_data = data[data.Where == 'test']
del data
print('Size of train_data: ',train_data.shape)
print('Size of test_data: ',test_data.shape)
X = train_data[num_data]
y = train_data['label']
X.shape, y.shape
FOLD_SIZE = 5
skf = StratifiedKFold(n_splits = FOLD_SIZE, shuffle=True, random_state = 123)
# # Logistic regression
# logreg = LogisticRegression(class_weight='balanced',max_iter = 2000)
# param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1,2,3,3,4,5,10,20]}
# clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
# clf.fit(X,y)
# print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))

# Random forest classifier
# forest = RandomForestClassifier(random_state = 1)
# n_estimators = [50, 100, 300, 500, 800]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10]
# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)
# gridF = GridSearchCV(forest, hyperF,scoring = 'roc_auc', cv = FOLD_SIZE , verbose = 1, 
#                       n_jobs = -1)
# gridF = gridF.fit(X,y)
# print(gridF)

cv_score_lr = []
cv_score_rf = []
i = 1
for train_index, valid_index in skf.split(X,y):
    print('{} of sKFold {}...'.format(i,FOLD_SIZE))
    Xtr, Xvl = X.loc[train_index], X.loc[valid_index]
    ytr, yvl = y.loc[train_index], y.loc[valid_index]
    
    #Model
    model_lr = LogisticRegression(C = 2, max_iter = 2000)
    model_rf = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)
    model_lr.fit(Xtr, ytr)
    model_rf.fit(Xtr,ytr)
    score_lr = roc_auc_score(yvl,model_lr.predict(Xvl))
    score_rf = roc_auc_score(yvl,model_rf.predict(Xvl))
    print('ROC_AUC score model_lr: ',score_lr)
    print('ROC_AUC score model_rf: ',score_rf)
    cv_score_lr.append(score_lr)
    cv_score_rf.append(score_rf)
    i += 1
# Confusion matrix of logistic classification model
print('Confusion matrix model_lr\n',confusion_matrix(yvl,model_lr.predict(Xvl)))
# Mean of Roc_auc_score of logistic classification model
print('Mean cv score model_lr: ',np.mean(cv_score_lr))
# Confusion matrix of Random Forest model
print('Confusion matrix model_rf\n',confusion_matrix(yvl,model_rf.predict(Xvl)))
# Mean of Roc_auc_score of Random forest model
print('Mean cv score model_rf: ',np.mean(cv_score_rf))
