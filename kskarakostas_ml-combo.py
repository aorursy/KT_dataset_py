# encoding=utf8  
import sys

try:
    stdin, stdout, stderr = sys.stdin, sys.stdout, sys.stderr
    reload(sys)  
    sys.stdin, sys.stdout, sys.stderr = stdin, stdout, stderr
    sys.setdefaultencoding('utf8')
except:
    pass

random_state = 101
RS = 101
import time, os

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from nltk.stem import WordNetLemmatizer
from sklearn.utils import class_weight

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier

import pandas as pd
import numpy as np
import json, re, warnings
import pdb
warnings.filterwarnings('ignore')
wnl = WordNetLemmatizer()
#https://www.kaggle.com/ashishpatel26/multimodel-approach
def pre_processing_(recipe):
    wnl = WordNetLemmatizer()
    recipe = [str.lower(ingredient) for ingredient in recipe]
    recipe = [delete_brand_(ingredient) for ingredient in recipe]
    recipe = [delete_state_(ingredient) for ingredient in recipe]
    recipe = [delete_comma_(ingredient) for ingredient in recipe]
    recipe = [original_(ingredient) for ingredient in recipe]
    recipe = [delete_space_(ingredient) for ingredient in recipe if len(delete_space_(ingredient))>2]
    return recipe

# 2. 
def delete_brand_(ingredient):
    ingredient = re.sub("country crock|i can't believe it's not butter!|bertolli|oreo|hellmann's", '', ingredient)
    ingredient = re.sub("red gold|hidden valley|original ranch|frank's|redhot|lipton", '', ingredient)
    ingredient = re.sub("recipe secrets|eggland's best|hidden valley|best foods|knorr|land o lakes", '', ingredient)
    ingredient = re.sub("sargento|johnsonville|breyers|diamond crystal|taco bell|bacardi", '', ingredient)
    ingredient = re.sub("mccormick|crystal farms|yoplait|mazola|new york style panetini", '', ingredient)
    ingredient = re.sub("ragu|soy vay|tabasco|truvía|crescent recipe creations|spice islands", '', ingredient)
    ingredient = re.sub("wish-bone|honeysuckle white|pasta sides|fiesta sides", '', ingredient)
    ingredient = re.sub("veri veri teriyaki|artisan blends|home originals|greek yogurt|original ranch", '', ingredient)
    ingredient = re.sub("jonshonville", '', ingredient)
    ingredient = re.sub("old el paso|pillsbury|progresso|betty crocker|green giant|hellmanns|hellmannâ€", '', ingredient)
    ingredient = re.sub("oscar mayer deli fresh smoked", '', ingredient)
    return ingredient

# 3. 재료 손질, 상태를 제거하는 함수
def delete_state_(ingredient):
    ingredient = re.sub('frozen|chopped|ground|fresh|powdered', '', ingredient)
    ingredient = re.sub('sharp|crushed|grilled|roasted|sliced', '', ingredient)
    ingredient = re.sub('cooked|shredded|cracked|minced|finely', '', ingredient)        
    return ingredient

# 4. 콤마 뒤에 있는 재료손질방법을 제거하는 함수
def delete_comma_(ingredient):
    ingredient = ingredient.split(',')
    ingredient = ingredient[0]
    return ingredient

## 그외 전처리 함수 (숫자제거, 특수문자제거, 원형으로변경)
def original_(ingredient):
    # 숫자제거
    ingredient = re.sub('[0-9]', '', ingredient)
    # 특수문자 제거
    ingredient = ingredient.replace("oz.", '')
    ingredient = re.sub('[&%()®™/]', '', ingredient)
    ingredient = re.sub('[-.]', '', ingredient)
    # lemmatize를 이용하여 단어를 원형으로 변경
    ingredient = wnl.lemmatize(ingredient)

    return ingredient

# 양 끝 공백을 제거하는 함수
def delete_space_(ingredient):
    ingredient = ingredient.strip()
    return ingredient
lemmatizer = WordNetLemmatizer()

replaceC = [('  ', ' '),('  ', ' '),('ú', 'u'), ('é', 'e'), ('è', 'e'), ('î', 'i'), ('â', 'a'), ('í', 'i'), ('â€™', ''), ('€', ''), ('ç', 'c'), 
            ('half & half', 'halfandhalf'),('half half', 'halfandhalf'),  ('all purpose', ''), 
            ('seasoning mix', 'seasoning'), ('style seasoning', 'seasoning'), ('salt free', 'saltfree'), ('taco seasoning reduced sodium', 'taco seasoning'),
            ('less sodium taco seasoning', 'low sodium taco seasoning'), ('old el paso', ''), ('old bay','oldbay'), 
            ('jerk rub', 'jerk'), ('taco bell home originals', ''), ('taco bell', ''), ('low sodium', ''),
            #('cauliflorets', 'florets'), 
            ('frozen broccoli florets', 'broccoliflorets'), ('broccoli florets', 'broccoliflorets'),
            ('high gluten bread flour', 'highglutenflour'), ('high gluten flour', 'highglutenflour'), 
            
            ('chapatti', 'chapati'), ('corn flour', 'cornflour'), ('not low fat', ''), ('low fat', ''), 
            ('cauliflorets florets','cauliflorets'),  ('cauliflorets flowerets','cauliflorets'), ('cauliflower','cauliflorets'), ('cauliflower florets','cauliflorets'), ('caulifloretsets', 'cauliflorets'),
            
            ('bread flour','flour'), ('self raising flour','selfraisingflour'), ('self rising cake flour','selfraisingflour'), ('self rising flour','selfraisingflour'),
            
            
           ]
#rmC = ['®', '™', ',', '.', '!', '(', ')', '%','’','"',"'", '-']
rmC = u'®™,.;!()[]\'’"-_@#$%&`/'
p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')

def preprocess_ing(Y, doReplaces=True, rmSpaces=False):
    if len(Y)<2: return []
    if rmSpaces: Y = [_.replace('  ',' ').replace(' ','_') for _ in Y]
    ingredients_text = ' '.join(Y)
    ingredients_text = ingredients_text.lower()
    #ingredients_text = ingredients_text.replace('-', ' ').replace('™', ' ').replace('®', ' ').replace(',', ' ')
    #for rm in rmC: ingredients_text = ingredients_text.replace(rm, ' ')
    if doReplaces:
        for rm in rmC: 
            try:
                #ingredients_text = ingredients_text.encode('utf-8').replace(rm.encode('utf-8'), ' ')
                ingredients_text = ingredients_text.replace(rm, ' ')
            except  Exception as e:
                print (e)
                print(rm)
                print(ingredients_text)
        for rc in replaceC:  ingredients_text = ingredients_text.replace(rc[0], rc[1])
        for rc in replaceC:  ingredients_text = ingredients_text.replace(rc[0], rc[1])
        if rmSpaces:
            for rc in replaceC[2:]:
                ingredients_text = ingredients_text.replace(rc[0].replace(' ','_'), rc[1].replace(' ','_'))

    words = []
    if rmSpaces: split_char='_'
    else: split_char=' '
    for word in ingredients_text.split(split_char):
        word = re.sub(p0, split_char, word.strip())
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return words

def preprocess_data(X, preproMode, doReplaces=True, rmSpaces=False):
    _X = []
    for x in X:
        if preproMode=='KK': x['ingredients'] = preprocess_ing(x['ingredients'], doReplaces=doReplaces, rmSpaces=rmSpaces)
        elif preproMode=='AV': x['ingredients'] = pre_processing_(x['ingredients'])
        _X.append(x)
    return _X


def conv_secs(s):
    mins, secs = divmod(s,60)
    if mins>=60: hours, mins = divmod(mins,60)
    else: hours=0
    if hours<24: return '%02d:%02d:%02d' % (hours, mins, secs)
    else:
        days, hours = divmod(hours,24)
        return '%02d-%02d:%02d:%02d' % (days, hours, mins, secs)

def do_train_fit(model, model_name, USEOVR):
    #if 'XGB' in model_name:
    #    model._Booster.set_param({'num_class': len(lb.classes_)})
    #    xgb_param = model.get_xgb_params()
    #    xgb_param['num_class'] = 3
    t0 = time.time()
    #if len(best_params)>0:model=model(**best_params)
    if USEOVR: 
        model = OneVsRestClassifier(model, n_jobs=-1)
        model_name = 'OVR_'+model_name
    model.fit(X_train, y_train)
    #print("Test the %s on the validation data" % model_name)
    y_pred_valid = model.predict(X_valid)
    y_pred_train = model.predict(X_train)
    need_time = time.time()-t0
    print ('%s %s Acc_T: %.4f Acc_V: %.4f [%s]' % (tfidf_name, model_name, accuracy_score(y_pred_train, y_train), accuracy_score(y_pred_valid, y_valid), conv_secs(need_time)))
    return model, model_name, y_pred_valid
    
# Dataset Preparation
print ("Read Dataset ... ")
def read_dataset(path): 
    try:return json.load(open(path, encoding='utf-8')) 
    #except:return json.load(open(path), encoding='utf-8') 
    except:return json.load(open(path)) 

_HOMEFOLDER_ = os.environ['HOME']
DATA_FOLDER = '../input/'
#DATA_FOLDER = _HOMEFOLDER_+'/Dropbox/Learning/Kaggle/WhatsCooking/'

train = read_dataset(DATA_FOLDER+'train.json')
submission = read_dataset(DATA_FOLDER+'test.json')

# prepare X and y
target = [doc['cuisine'] for doc in train]
print ("Label Encode the Target Variable ... ")
lb = LabelEncoder()
y = lb.fit_transform(target)
class_weights_ = class_weight.compute_class_weight('balanced', np.unique(y), y)
class_weights = {}
for i in range(len(class_weights_)): class_weights[i]=100/class_weights_[i]

X = train

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.00010, random_state=random_state)

print(len(X_train), len(X_valid))
# Text Data Features
print ("Prepare text data  ... ")
def generate_text(data, preproMode, doReplaces=True, rmSpaces=False):
    data = preprocess_data(data, preproMode=preproMode, doReplaces=doReplaces, rmSpaces=rmSpaces)
    text_data = [" ".join(doc['ingredients']) for doc in data]
    return text_data 

doReplaces = False
rmSpaces   = False
preproMode = 'KK'
preproMode = 'AV' #https://www.kaggle.com/ashishpatel26/up-to-all-food-and-all-models

train_text = generate_text(X_train, preproMode=preproMode, doReplaces=doReplaces, rmSpaces=rmSpaces)
valid_text = generate_text(X_valid, preproMode=preproMode, doReplaces=doReplaces, rmSpaces=rmSpaces)
submission_text = generate_text(submission, preproMode=preproMode, doReplaces=doReplaces, rmSpaces=rmSpaces)

print ("Generation Done! ")
#pre_processing_(X_train[0]['ingredients'])
# Feature Engineering 
print ("TF-IDF on text data ... ")
#tfidf_name, tfidf = 'bT_SWN_xD100_nD1_sF', TfidfVectorizer(binary=True,stop_words=None,max_df=1.0, min_df=1,sublinear_tf=False)
#tfidf_name, tfidf = 'bF_SWN_xD090_nD2_sF', TfidfVectorizer(binary=False,stop_words=None,max_df=0.9, min_df=2,sublinear_tf=False)
#tfidf_name, tfidf = 'bT_SWE_xD065_nD2_sF', TfidfVectorizer(binary=True,stop_words='english', max_df=0.65, min_df=2,sublinear_tf=False)
#tfidf_name, tfidf = 'nP_bF_SWN_xD075_nD2_sT', TfidfVectorizer(binary=False,stop_words=None,max_df=0.75, min_df=2, sublinear_tf=True)
#tfidf_name, tfidf = 'bF_SWN_xD070_nD4_sT', TfidfVectorizer(binary=False,stop_words=None,max_df=0.7, min_df=4,sublinear_tf=True)
tfidf_name, tfidf = 'bF_SWN_xD060_nD1_sT_nl2', TfidfVectorizer(binary=False,stop_words=None,max_df=0.6, min_df=1,sublinear_tf=True, norm='l2')

if rmSpaces: tfidf_name = 'RmT_'+tfidf_name
else: tfidf_name = 'RmF_'+tfidf_name
    
if doReplaces: tfidf_name = 'RpT'+tfidf_name
else: tfidf_name = 'RpF'+tfidf_name

tfidf_name=preproMode+'_'+tfidf_name
    
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    # x.sort_indices()
    return x 

X_train = tfidf_features(train_text, flag="train")
X_valid = tfidf_features(valid_text, flag="valid")
X_submission = tfidf_features(submission_text, flag="submission")

print ("Feature Engineering [%s] DONE" % tfidf_name)
model_njobs = -1
model_list = [ # Must be ('Model_Name', model(wanted=params), {params:for_GSearch})
    # 0
    ('SVC_C50_g1p4_d3_c1', 
     SVC(C = 50, gamma=1.4, kernel='rbf', degree=3, coef0=1, shrinking=True, tol=0.0001, probability=True,
             cache_size=2000, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=RS), 
     {'C':[30, 50, 100, 150, 220], 'gamma': [0.1, 0.75, 1, 3, 5], 'degree': [1,2,3,4], 
      'coef0': [0, 0.5, 1], 'kernel':('linear', 'rbf', 'poly')},
    {}),
    # 1
    ('LR_C7_l2_NewCG_max200_dF', 
     LogisticRegression(penalty='l2', C=8,
                        max_iter = 200, dual = False,
                        solver='newton-cg',
                        multi_class='ovr',
                        n_jobs = model_njobs,
                        random_state=RS), 
     {'C': [0.1,0.5,1,5,10,15], 'solver': ['sag', 'newton-cg', 'lbfgs'], 
      'max_iter':[100,200,300], 'dual':[True, False], 'multi_class': ['multinomial', 'ovr']},
    {}), 
    # 2
    ('RFC_150_None', 
     RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_split=2, 
                            max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=model_njobs, 
                            random_state=RS, verbose=0, warm_start=False, class_weight=None),
    {'n_estimators':[25, 50, 100, 150, 200], 'max_features':['sqrt', 'log2', None], },
    {'n_estimators':150, 'max_features':'log2'}),
    # 3
    ('SGD_l_l2_opt_a1m5', 
     SGDClassifier(loss='log', penalty='l2', alpha=1e-05, l1_ratio=0.15, 
                   fit_intercept=True, max_iter=1000, tol=1e-03, shuffle=True, 
                   verbose=0, epsilon=0.1, n_jobs=model_njobs, random_state=RS, 
                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, 
                   warm_start=False, average=False, n_iter=None),
     {'loss': ['hinge','log','modified_huber','squared_hinge','perceptron',], 
      'alpha': [1.0e-6,1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2], 'learning_rate': ['optimal']},
    {'loss': 'log', 'learning_rate': 'optimal', 'alpha': 1e-05}),
    # 4
    ('LinSVC', LinearSVC(), {}, {}),
    
    #5
    ('ABC', 
     AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, 
                        algorithm='SAMME.R', random_state=RS), 
     {'n_estimators':[25, 50, 100, 150, 200], 'algorithm': ['SAMME.R', 'SAMME']}, {}),
    
    #6
    ('GBC',
     GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                criterion='friedman_mse', min_samples_split=2, max_depth=3, 
                                min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                                random_state=RS, max_features=None, verbose=0, max_leaf_nodes=None, 
                                warm_start=False, presort='auto', 
                                #validation_fraction=0.1, n_iter_no_change=None, tol=0.0001
                               ), 
     {'loss':['deviance','exponential'], 'n_estimators':[100, 150, 200, 300], 'max_features':['sqrt', 'log2', None]}, 
     {}),
    
    #7
    ('Perceptron_50', Perceptron(penalty=None, alpha=0.0001, fit_intercept=True, max_iter=None, tol=1e-4, shuffle=True, 
                                 verbose=0, eta0=1.0, n_jobs=model_njobs, random_state=RS, class_weight=None, 
                                 warm_start=False, n_iter=None),
     {'penalty':['l2', 'l1', 'elasticnet', None], 'max_iter':[10,100,500,1000], 'eta0':[0.5,1.0,1.5]}, {}),
    
    #8
    ('KNC_50', KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, 
                                                     p=2, metric='minkowski', metric_params=None, n_jobs=model_njobs),
     {'weights':['distance', 'uniform'], 'algorithm':['ball_tree', 'kd_tree', 'brute'], 'n_neighbors': [10,20,30,50,70], 
      'leaf_size': [10,15,20,30]},
     {}),
    
    #9
    ('LinearSVC_l2_dF', LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4, C=1.0, multi_class='ovr', 
                                  fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=RS, max_iter=2000),
     {'dual':[True, False], 'C': [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2, 3, 4]},{}),
    #10
    ('SVC_kg', SVC(C=200, kernel='rbf', degree=3,gamma=1, coef0=1, shrinking=True,tol=0.001, probability=False, 
                   cache_size=200,class_weight=None, verbose=False, max_iter=-1,decision_function_shape=None,random_state=RS), {},{}),
    #11
    ('RFC', 
     ExtraTreesClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                          #min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                          max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                          min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=model_njobs, random_state=RS, verbose=0, 
                          warm_start=False, class_weight=None), 
    {'max_features':['sqrt', 'log2', None],'n_estimators':[100, 500],'min_samples_leaf': [10,50,100,200,500]}, {}),
    #12
    ('XGB', 
     #XGBClassifier(max_depth = 9, eta = 0.003, subsample = 0.7, gamma = 7, n_jobs=model_njobs,), 
     XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=1, learning_rate=0.01, max_delta_step=0,
       max_depth=12, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=2, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.8),
     {}, {}),
]
if True:
    SEL_MODEL = 0
    model_name, model, gs_params, best_params = model_list[SEL_MODEL]
    #model, model_name, y_pred_valid = do_train_fit(model, model_name, False)
    model, model_name, y_pred_valid = do_train_fit(model, model_name, True)
#?XGBClassifier
#RpFRmF_bF_SWN_xD060_nD1_sT_nl2 OVR_XGB Acc_T: 0.8636 Acc_V: 0.7748 [00:03:50]
#dir(model)
#model._Booster.set_param({'num_class': len(lb.classes_)})

#print (model._Booster)

#RUNMODE = 'KFOLD'
#RUNMODE = 'DOSCAN_G'
#RUNMODE = 'DOSCAN_R'
RUNMODE = 'VOTE_S'
#RUNMODE = 'VOTE_H'
#RUNMODE = 'FIT'
#RUNMODE = 'FIT_BOTH'
SEL_MODEL = 0

USECLASSWEIGHTS = True
USECLASSWEIGHTS = False

USEOVR = True
#USEOVR = False

print("features: ", X_train.shape)

starttime = time.time()


model_name, model, gs_params, best_params = model_list[SEL_MODEL]

if 'class_weight' in model.__doc__ and USECLASSWEIGHTS:
    model_name = 'CW_'+model_name
    model.__setattr__('class_weight', class_weights)

if RUNMODE.split('_')[0] == 'DOSCAN':
    if RUNMODE[-1]=='R':clf = RandomizedSearchCV(model, param_distributions=gs_params, n_iter=20, verbose=10, n_jobs=-1)
    elif RUNMODE[-1]=='G':clf = GridSearchCV(model, gs_params, cv=3, verbose=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    print('best score: ', clf.best_score_)
    print("best params: ", clf.best_params_)
    print('best est: ', clf.best_estimator_)
    with open("bestparams.txt", "w") as data:
        data.write(str(clf.best_params_))
elif RUNMODE == 'KFOLD':
    #K-fold validation
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(n_splits=5, random_state=random_state).split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        training_data = X_train[train]
        training_data.sort_indices()
        model.fit(training_data, y_train[train])

        score = model.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Acc: %.3f' % (k+1, score))
        
elif RUNMODE == 'FIT':
    model, model_name, y_pred_valid = do_train_fit(model, model_name, USEOVR)
elif RUNMODE == 'FIT_BOTH':
    model, model_name, y_pred_valid = do_train_fit(model, model_name, False)
    model, model_name, y_pred_valid = do_train_fit(model, model_name, True)
elif RUNMODE.split('_')[0] == 'VOTE':
    SEL_MODELS = [0,1,3]
    WEIGHTS = [5,2,2]
    vote_est = []
    voting = 'hard'
    if RUNMODE[-1]=='S': voting = 'soft'
        
    v_model_name = 'V' + voting[0].upper()
    for _ in SEL_MODELS:
        vote_est.append((model_list[_][0], model_list[_][1]))
        v_model_name += '-'+ vote_est[-1][0]
    print('*'*66)
    print('*  Will use total %d ests' % len(vote_est))
    for x,_ in vote_est: print('* %s' % x)
    model = VotingClassifier(estimators = vote_est,
                            voting = voting,
                            weights = WEIGHTS,
                            n_jobs=-1)
    if USEOVR: model = OneVsRestClassifier(model, n_jobs=4)
    model.fit(X_train, y_train)
    print("Test the model on the validation data")
    y_pred_valid = model.predict(X_valid)
    print('Accuracy: %.4f' % accuracy_score(y_pred_valid, y_valid))
    print ('%s %s' % (tfidf_name, model_name))


print("Time: ", conv_secs(time.time()-starttime))
# Predictions 
print ("Predict on submision data ... ")
y_submission = model.predict(X_submission)
y_pred = lb.inverse_transform(y_submission)

# Submission
print ("Generate Submission File ... ")
submission_id = [doc['id'] for doc in submission]
sub = pd.DataFrame({'id': submission_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('%s_%s_%s.csv' %(tfidf_name, model_name, str(random_state) ), index=False)
if False:
    im = -1
    exclude_ids = [0,10]
    for model_name, model, gs_params, best_params in model_list:
        im+=1
        if im in exclude_ids: continue
        
        try:
            print('*'*66)
            print(im, model_name)
            #model, model_name, y_pred_valid = do_train_fit(model, model_name, False)
            try:model, model_name, y_pred_valid = do_train_fit(model, model_name, True)
            except:pass
        except Exception as e:
            print (e)






