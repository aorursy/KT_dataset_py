!pip install catboost
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,f1_score

from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import BaggingClassifier

from catboost import CatBoostClassifier

from scipy.stats import norm, skew

from scipy.special import boxcox1p

import lightgbm as lgb

from mlens.ensemble import SuperLearner



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split



%matplotlib inline
train = pd.read_csv('../input/train_LZdllcl.csv')

test = pd.read_csv('../input/test_2umaH9m.csv')
train.head()
train.describe()
train.isnull().sum()
train.dtypes
train['recruitment_channel'].value_counts()
train['education'].value_counts()
plt.matshow(train.corr())

plt.show()
f, ax = plt.subplots(figsize=(10, 8))

corr = train.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
plt.hist(train[train['KPIs_met >80%'] == 0]['previous_year_rating'])

plt.show()
plt.hist(train[train['KPIs_met >80%'] == 1]['previous_year_rating'])

plt.show()
prev_rating_Three = train[(train['previous_year_rating'].isnull())][train['KPIs_met >80%'] == 0]['employee_id']



for empId in tqdm(prev_rating_Three):

    train.loc[train['employee_id']==empId,'previous_year_rating'] = 3.0

    

prev_rating_Three = test[(test['previous_year_rating'].isnull())][test['KPIs_met >80%'] == 0]['employee_id']



for empId in tqdm(prev_rating_Three):

    test.loc[test['employee_id']==empId,'previous_year_rating'] = 3.0
prev_rating_Five = train[(train['previous_year_rating'].isnull())][train['KPIs_met >80%'] == 1]['employee_id']



for empId in tqdm(prev_rating_Five):

    train.loc[train['employee_id']==empId,'previous_year_rating'] = 5.0

    

prev_rating_Five = test[(test['previous_year_rating'].isnull())][test['KPIs_met >80%'] == 1]['employee_id']



for empId in tqdm(prev_rating_Five):

    test.loc[test['employee_id']==empId,'previous_year_rating'] = 5.0
train['education'] = train['education'].fillna('notGiven')

test['education'] = test['education'].fillna('notGiven')
X = train.drop(columns=['employee_id','is_promoted'])

y = train['is_promoted']

test.drop(columns=['employee_id'],inplace=True)
X.head()
X['no_of_trainings'].unique()
train.plot.hexbin('is_promoted','age',gridsize=15)
def binAge(x):

    age=''

    if (x<26):

        age='young'

    elif (x>=26 and x<=36):

        age='medium'

    else:

        age='old'

    return age



def binAgeTwo(x):

    age=''

    if (x<=30):

        age='young'

    else:

        age='medium'

    return age
X['age_bin_Two'] = X['age'].apply(binAgeTwo)

X['age_bin'] = X['age'].apply(binAge)



test['age_bin_Two'] = test['age'].apply(binAgeTwo)

test['age_bin'] = test['age'].apply(binAge)
X['frac_train'] = X['no_of_trainings']/10

test['frac_train'] = test['no_of_trainings']/10
X['crit_score'] = ((X['previous_year_rating']*X['KPIs_met >80%'] + X['previous_year_rating']*X['awards_won?'] + X['previous_year_rating']*X['frac_train'] ) * (20) + X['avg_training_score'])/400

test['crit_score'] = ((test['previous_year_rating']*test['KPIs_met >80%'] + test['previous_year_rating']*test['awards_won?'] + test['previous_year_rating']*test['frac_train'] ) * (20) + test['avg_training_score'])/400   
sns.boxplot(y,X['crit_score'])
contVar = set(X.columns) - set(['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two'])  

contVar = list(contVar)
flag = 1

iter = 0

while(flag!=0):

    iter = iter + 1

    if(iter > 20):

        break

    skewed_feats = X[contVar].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    #print("\nSkew in numerical features: \n")

    skewness = pd.DataFrame({'Skew' :skewed_feats})

    print(skewness[np.abs(skewness['Skew'])>0.75])

    skewnessBox = skewness[(skewness.Skew)>0.75]

    skewnessSquare = skewness[(skewness.Skew)<-0.75]

    if(skewnessBox.shape[0] == 0 and skewnessSquare.shape[0] == 0):

        flag = 0

    #print("There are {} skewed numerical features to Box Cox transform".format(skewnessBox.shape[0]))

    #print("There are {} skewed numerical features to Square transform".format(skewnessSquare.shape[0]))

    skewed_features1 = skewnessBox.index

    skewed_features2 = skewnessSquare.index

    #print(skewed_features1)

    #print(skewed_features2)

    lam = 0.15

    for feat in skewed_features1:

        X[feat] = boxcox1p(X[feat], lam)

        test[feat] = boxcox1p(test[feat], lam)

    for feat in skewed_features2:

        X[feat] = np.square(X[feat])

        test[feat] = np.square(test[feat])
X_dummies = X.copy()

X_dummies = pd.get_dummies(X,columns=['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two'])    

test_dummies = pd.get_dummies(test,columns=['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two'])    
Xstand = X_dummies.copy()

Xmin = X_dummies.copy()

Xrob = X_dummies.copy()



teststand = test_dummies.copy()

testmin = test_dummies.copy()

testrob = test_dummies.copy()
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler



standScl = StandardScaler()

minScl = MinMaxScaler()

robScl = RobustScaler()



Xstand[contVar] = standScl.fit_transform(X[contVar])

Xmin[contVar] = minScl.fit_transform(X[contVar])

Xrob[contVar] = robScl.fit_transform(X[contVar])



teststand[contVar] = standScl.transform(test[contVar])

testmin[contVar] = minScl.transform(test[contVar])

testrob[contVar] = robScl.transform(test[contVar])
def scoreOfModel(clf,X,y,flag,shuffleBool=False,nFolds=12):

    score = 0

    finalPreds = np.zeros(23490)

    #trainPreds = np.zeros(54808)

    folds = StratifiedKFold(n_splits=nFolds, shuffle=shuffleBool, random_state=42)

    train_pred = cross_val_predict(clf, X, y, cv=12,method='predict_proba')

    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X,y))):

        X_train,X_val = X.loc[trn_idx,:],X.loc[val_idx,:]

        #X_train,X_val = X[trn_idx],X[val_idx]

        y_train,y_val = y[trn_idx],y[val_idx]

        clf.fit(X_train,y_train)

        yPreds = clf.predict(X_val)

        score += f1_score(y_val,yPreds)

        if(flag==0):

            finalPreds += clf.predict(teststand)

        elif (flag==1):

            finalPreds += clf.predict(testmin)

        elif(flag==2):

            finalPreds += clf.predict(testrob)

        elif(flag==3):

            p = clf.predict_proba(test)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        print("**********"+ str(score/(1+fold_)) + "******************Iteration "+str(fold_)+" Done****************")    

    return str(score/nFolds),(train_pred),(finalPreds/nFolds)



def scoreOfModelTwo(clf,X,y,X_val,y_val):

    clf.fit(X,y)

    yPreds = clf.predict(X_val)

    score = f1_score(y_val,yPreds)

    finalPreds = clf.predict(test)

    return str(score),(finalPreds)





def scoreOfModelLGB(clfr,X,y,flag,shuffleBool=False,nFolds=12):

    score = 0

    finalPreds = np.zeros(23490)

    #trainPreds = np.zeros(54808)

    folds = StratifiedKFold(n_splits=nFolds, shuffle=shuffleBool, random_state=42)

    train_pred = cross_val_predict(clfr, X, y, cv=12,method='predict_proba')

    for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X,y))):

        X_train,X_val = X.loc[trn_idx,:],X.loc[val_idx,:]

        #X_train,X_val = X[trn_idx],X[val_idx]

        y_train,y_val = y[trn_idx],y[val_idx]

        clf = clfr.fit(X_train,y_train)

        yPreds = clf.predict(X_val)

        score += f1_score(y_val,yPreds)

        if(flag==0):

            p = clf.predict_proba(teststand)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        elif (flag==1):

            p = clf.predict_proba(testmin)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        elif(flag==2):

            p = clf.predict_proba(testrob)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        elif(flag==3):

            p = clf.predict_proba(test)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        elif(flag==4):

            p = clf.predict_proba(test_dummies)

            #q = clf.predict_proba(X)    

            for k in range(len(p)):

                finalPreds[k] += p[k][0]

            #for l in range(len(q)):

            #    trainPreds[l] += q[l][0]

        print("**********"+ str(score/(1+fold_)) + "******************Iteration "+str(fold_)+" Done****************")    

    return str(score/nFolds),(train_pred),(finalPreds/nFolds)

catClf2 = CatBoostClassifier(learning_rate = 0.0353,iterations = 1500,eval_metric='F1',cat_features=['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two'])    

clfDummies1 = lgb.LGBMClassifier(max_depth= 7, learning_rate=0.07532, n_estimators=402, num_leaves= 28, reg_alpha=2.154 , reg_lambda= 1.028)

clfDummies2 = lgb.LGBMClassifier(max_depth= 10, learning_rate=0.1, n_estimators=308, num_leaves= 30, reg_alpha=1.0 , reg_lambda= 0.1)

clfStand1 = lgb.LGBMClassifier(max_depth= 4, learning_rate=0.1, n_estimators=635, num_leaves= 30, reg_alpha=1.0 , reg_lambda= 0.1)

clfmin1 = lgb.LGBMClassifier(max_depth= 4, learning_rate=0.1, n_estimators=635, num_leaves= 30, reg_alpha=1.0 , reg_lambda= 0.1)



scr_clfDummies1,trainclfDummies1Preds,catclfDummies1Preds = scoreOfModelLGB(clfDummies1,X_dummies,y,4)

scr_clfDummies2,trainclfDummies2Preds,catclfDummies2Preds = scoreOfModelLGB(clfDummies2,X_dummies,y,4)

scr_clfStand1,trainclfStand1Preds,catclfStand1Preds = scoreOfModelLGB(clfStand1,Xstand,y,0)

scr_clfmin1,trainclfmin1Preds,catclfmin1Preds = scoreOfModelLGB(clfmin1,Xmin,y,1)

scr_catClf2,traincatClf2Preds,catClf2Preds = scoreOfModel(catClf2,X,y,3)
stackedDF = pd.DataFrame({'dummiesOne' : trainclfDummies1Preds[:,0], 'dummiesTwo' : trainclfDummies2Preds[:,0],

                          'standOne' : trainclfStand1Preds[:,0], 'minOne' : trainclfmin1Preds[:,0],

                          'catClf' : traincatClf2Preds[:,0]

                         })



stackedTest = pd.DataFrame({'dummiesOne' : catclfDummies1Preds , 'dummiesTwo' : catclfDummies2Preds,

                          'standOne' : catclfStand1Preds, 'minOne' : catclfmin1Preds,

                          'catClf' : catClf2Preds

                         })
stackedDF.head()
f, ax = plt.subplots(figsize=(10, 8))

corr = stackedDF.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax)
stackedTest.head()
# #bounds on different parameters 

# param_to_be_optimized = {'iterations':(600,1500),'learning_rate':(0.03,0.05),'depth':(3,10),

#                         'l2_leaf_reg':(2,21)}





# def param_handler_to_optimize(iterations,learning_rate,depth,l2_leaf_reg):

#     """

#     To handle integer type parameters:

    

#     """

#     thread_count=-1

#     iterations = int(iterations)

#     depth = int(depth) #int type params

#     #border_count = int(border_count)

#     #ctr_border_count = int(ctr_border_count)

    

#     param = {

#     'iterations': iterations,  # the maximum depth of each tree

#     'learning_rate': learning_rate,  # the training step for each iteration

#     'silent': True,  # logging mode - quiet

#     'depth':depth,

#     'l2_leaf_reg':l2_leaf_reg,

#     #'border_count':border_count,

#     #'thread_count':thread_count,

#     'task_type':'GPU',

#     'loss_function': 'CrossEntropy',  # error evaluation for multiclass training

#     'cat_features':['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two']  

#     }

#     return func_to_be_optimized(param)



# def func_to_be_optimized(param):

    

#     model = CatBoostClassifier(**param)    

#     score = 0

#     #finalPreds=np.zeros(23490)

#     folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=42)

#     for fold_, (trn_idx, val_idx) in tqdm(enumerate(folds.split(X,y))):

#         X_train,X_val = X.loc[trn_idx,:],X.loc[val_idx,:]

#         y_train,y_val = y[trn_idx],y[val_idx]

#         model.fit(X_train,y_train)

#         yPreds = model.predict(X_val)

#         score += f1_score(y_val,yPreds)

#     return (score/12)







# optimizer = BayesianOptimization(

#     f=param_handler_to_optimize,

#     pbounds=param_to_be_optimized,

#     random_state=1,

# )

# optimizer.maximize(

#     init_points=3,

#     n_iter=75,

# )

#------------------------------------------------------------------------------------------------

#bounds on different parameters 

# param_to_be_optimized = {'C':(0.001,20000)

#                         }



# def param_handler_to_optimize(C):

#     """

#     To handle integer type parameters:

    

#     """

   

#     param = {

#     'C':C,

#     #'max_depth':max_depth

#     #'C':C,

#     'solver':'liblinear'

#     }

#     return func_to_be_optimized(param)





# param_to_be_optimized = {'max_depth': (2, 15),'learning_rate': (0.001, 0.5),'n_estimators': (10, 1000), 

#                          'num_leaves': (2,50),'reg_alpha': (0.01, 10),'reg_lambda': (0, 3)}                                          



# def param_handler_to_optimize(max_depth,learning_rate,n_estimators,num_leaves,reg_alpha,reg_lambda):

#     """

#     To handle integer type parameters:

    

#     """

#     #thread_count=-1

#     max_depth = int(max_depth)

#     n_estimators = int(n_estimators) #int type params

#     num_leaves = int(num_leaves)  

#     #border_count = int(border_count)

#     #ctr_border_count = int(ctr_border_count)

    

#     param = {

#     'max_depth' : max_depth,  # the maximum depth of each tree

#     'learning_rate' : learning_rate,  # the training step for each iteration

#     'n_estimators' : n_estimators,  # logging mode - quiet

#     'num_leaves' : num_leaves,

#     'reg_alpha' : reg_alpha,

#     'reg_lamda' : reg_lambda,

#     #'border_count':border_count,

#     #'thread_count':thread_count,

# #     'task_type':'GPU',

# #     'loss_function': 'CrossEntropy',  # error evaluation for multiclass training

# #     'cat_features':['previous_year_rating','department','region','education','gender','recruitment_channel','KPIs_met >80%','awards_won?','age_bin','age_bin_Two']  

#     }

#     return func_to_be_optimized(param)



# def func_to_be_optimized(param):

    

#     #model = lgb.LGBMClassifier(**param)    

#     train_pred_opt = cross_val_predict(LogisticRegression(**param), stackedDFL2, y, cv=12,method='predict_proba')

#     train_pred_optTweaked = train_pred_opt[:,0]

#     thresholds = np.linspace(0.01, 0.99, 50)

#     mcc = np.array([f1_score(y, train_pred_optTweaked<thr) for thr in thresholds])

#     #best_threshold = thresholds[mcc.argmax()]

#     return (mcc.max())



# optimizer = BayesianOptimization(

#     f=param_handler_to_optimize,

#     pbounds=param_to_be_optimized,

#     random_state=1,

# )

# optimizer.maximize(

#     init_points=3,

#     n_iter=75,

# )
metaLearner = RandomForestClassifier(max_depth=5,n_estimators =116) 



metaLearner.fit(stackedDF,y)

padRF = metaLearner.predict_proba(stackedDF)

padRFTweaked = padRF[:,0]



thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([f1_score(y, padRFTweaked<thr) for thr in thresholds])

plt.plot(thresholds, mcc)

best_threshold = thresholds[mcc.argmax()]

print(mcc.max())

print(best_threshold)
# metaLearner = LogisticRegression(solver='liblinear',C=1.931)

# metaLearner.fit(stackedDF,y)

# padLR = metaLearner.predict_proba(stackedDF)

# padLRtest = metaLearner.predict_proba(stackedTest)



# padLR = cross_val_predict(metaLearner, stackedDF, y, cv=12,method='predict_proba')

# padLRTweaked = padLR[:,0]



# thresholds = np.linspace(0.01, 0.99, 50)

# mcc = np.array([f1_score(y, padTweaked<thr) for thr in thresholds])

# plt.plot(thresholds, mcc)

# best_threshold = thresholds[mcc.argmax()]

# print(mcc.max())

# print(best_threshold)



# metaLearner = lgb.LGBMClassifier(learning_rate=0.07149459085784728,

#  max_depth= 4,

#  n_estimators= 10,

#  num_leaves= 49,

#  reg_alpha= 9.71326474211533,

#  reg_lambda= 0.36594150409622384) 



# metaLearner.fit(stackedDF,y)

# padLGBM = metaLearner.predict_proba(stackedDF)

# padLGBMtest = metaLearner.predict_proba(stackedTest)



# padLGBM = cross_val_predict(metaLearner, stackedDF, y, cv=12,method='predict_proba')



# padLGBMTweaked = padLGBM[:,0]



# thresholds = np.linspace(0.01, 0.99, 50)

# mcc = np.array([f1_score(y, padLGBMTweaked<thr) for thr in thresholds])

# plt.plot(thresholds, mcc)

# best_threshold = thresholds[mcc.argmax()]

# print(mcc.max())

# print(best_threshold)



# stackedDFL2 = pd.DataFrame({'padRF' : padRF[:,0], 'padLR' : padLR[:,0],

#                           'padLGBM' : padLGBM[:,0]

#                          })



# stackedTestL2 = pd.DataFrame({'padRF' : padRFtest[:,0] , 'padLR' : padLRtest[:,0],

#                           'padLGBM' : padLGBMtest[:,0]

#                          })



#metaLearner = LogisticRegression(solver='liblinear',C=optimizer.max['params']['C'])

# metaLearner = LogisticRegression(C=0.6064701322378264) 



# metaLearner.fit(stackedDFL2,y)

# padRF2 = metaLearner.predict_proba(stackedDFL2)

# padRF2Tweaked = padRF2[:,0]

# #padRF2 = cross_val_predict(metaLearner, stackedDFL2, y, cv=12,method='predict_proba')

# #padRF2Tweaked = padRF2[:,0]



# thresholds = np.linspace(0.01, 0.99, 50)

# mcc = np.array([f1_score(y, padRF2Tweaked<thr) for thr in thresholds])

# plt.plot(thresholds, mcc)

# best_threshold = thresholds[mcc.argmax()]

# print(mcc.max())

# print(best_threshold)
sampleShuffle = pd.read_csv('../input/sample_submission_M0L0uXE.csv')
pad = metaLearner.predict_proba(stackedTest)

padTweaked = pad[:,0]



sampleShuffle['is_promoted'] = padTweaked<best_threshold

sampleShuffle['is_promoted'] = np.where(sampleShuffle['is_promoted']==False,0,1)
sampleShuffle.to_csv('submissionShuffle.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "submissionShuffle.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(sampleShuffle)