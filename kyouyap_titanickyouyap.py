# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pylab as plt

from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

from scipy.stats import mstats

from tqdm import tqdm



from sklearn import metrics, model_selection, feature_selection, ensemble, gaussian_process, linear_model, naive_bayes, neighbors, svm, tree, discriminant_analysis, model_selection

from xgboost import XGBClassifier

from imblearn import under_sampling, over_sampling





from IPython.display import Image

from io import StringIO



import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

import numpy as np

import pandas as pd
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

data = pd.concat([train, test], sort=True)

data
data.isnull().sum()
#欠損値処理

data['Sex'].replace(['male','female'],[0, 1], inplace=True)



data['Embarked'].fillna(('S'), inplace=True)

data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

# data['Mr']=data.Name.apply(lambda x:1 if 'Mr' in x else 0)

# data['Mrs']=data.Name.apply(lambda x:1 if 'Mrs' in x else 0)

# data['Miss']=data.Name.apply(lambda x:1 if 'Miss' in x else 0)

# age_avg = data['Age'].mean()



data['Age'].fillna(int(age_avg), inplace=True)



delete_columns = [ 'PassengerId', ]

data.drop(delete_columns, axis = 1, inplace = True)

data['Salutation'] = data.Name.str.extract(' ([A-Za-z]+).', expand=False)

data['Salutation'] = data['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

data['Salutation'] = data['Salutation'].replace('Mlle', 'Miss')

data['Salutation'] = data['Salutation'].replace('Ms', 'Miss')

data['Salutation'] = data['Salutation'].replace('Mme', 'Mrs')

del data['Name']

Salutation_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



data["FamilySize"] = data["SibSp"] + data["Parch"] + 1



data['Salutation'] = data['Salutation'].map(Salutation_mapping)

data['Salutation'] = data['Salutation'].fillna(0)





data['Ticket_Lett'] = data['Ticket'].apply(lambda x: str(x)[0])

data['Ticket_Lett'] = data['Ticket_Lett'].apply(lambda x: str(x))

data['Ticket_Lett'] = np.where((data['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['Ticket_Lett'], np.where((data['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

data['Ticket_Len'] = data['Ticket'].apply(lambda x: len(x))

del data['Ticket']

data['Ticket_Lett']=data['Ticket_Lett'].replace("1",1).replace("2",2).replace("3",3).replace("0",0).replace("S",3).replace("P",0).replace("C",3).replace("A",3)





data['Cabin_Lett'] = data['Cabin'].apply(lambda x: str(x)[0]) 

data['Cabin_Lett'] = data['Cabin_Lett'].apply(lambda x: str(x)) 

data['Cabin_Lett'] = np.where((data['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),data['Cabin_Lett'], np.where((data['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0'))

del data['Cabin'] 

data['Cabin_Lett']=data['Cabin_Lett'].replace("A",1).replace("B",2).replace("C",1).replace("0",0).replace("D",2).replace("E",2).replace("F",1) 



# data['IsAlone'] = 0

# data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
data
train = data[:len(train)]

test = data[len(train):]
y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)
#生存者の割合

train.Survived.sum()/train.Survived.count()

#まぁ均衡ということでいいかも、一応　データの整形を行う

from imblearn import under_sampling, over_sampling

cols = train.columns.tolist()

cols.remove('Survived')



positive_cnt = int(train['Survived'].sum())

rus = under_sampling.RandomUnderSampler(sampling_strategy={0:positive_cnt, 1:positive_cnt}, random_state=0)

data_x_sample, data_y_sample  = rus.fit_sample(train[cols], train[['Survived']])

data_x_sample
#recvを用いて特徴量選択を行う

feature_importance_models = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    tree.DecisionTreeClassifier(),

    XGBClassifier()

]

 

scoring = ['accuracy']

df_rfe_cols_cnt = pd.DataFrame(columns=['cnt'], index=cols)

df_rfe_cols_cnt['cnt'] = 0

 

for i, model in tqdm(enumerate(feature_importance_models), total=len(feature_importance_models)):

    

    rfe = feature_selection.RFECV(model, step=3)

    rfe.fit(data_x_sample, data_y_sample)

    rfe_cols = train[cols].columns.values[rfe.get_support()]

    df_rfe_cols_cnt.loc[rfe_cols, 'cnt'] += 1

    

df_rfe_cols_cnt.plot(kind='bar', figsize=(15, 5))

plt.show()

#まぁまぁどのパラメーターも効いている。家族総数のパラメータを追加してみる
data["familiynumber"]=data.Parch+data.SibSp+1
data
#Familiynumberを設けてrecvを行う



train = data[:len(train)]

test = data[len(train):]

y_train = train['Survived']

X_train = train.drop('Survived', axis = 1)

X_test = test.drop('Survived', axis = 1)



cols = train.columns.tolist()

cols.remove('Survived')



positive_cnt = int(train['Survived'].sum())

rus = under_sampling.RandomUnderSampler(sampling_strategy={0:positive_cnt, 1:positive_cnt}, random_state=0)

data_x_sample, data_y_sample  = rus.fit_sample(train[cols], train[['Survived']])

#recvを用いて特徴量選択を行う

feature_importance_models = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    tree.DecisionTreeClassifier(),

    XGBClassifier()

]

 

scoring = ['accuracy']

df_rfe_cols_cnt = pd.DataFrame(columns=['cnt'], index=cols)

df_rfe_cols_cnt['cnt'] = 0

 

for i, model in tqdm(enumerate(feature_importance_models), total=len(feature_importance_models)):

    

    rfe = feature_selection.RFECV(model, step=3)

    rfe.fit(data_x_sample, data_y_sample)

    rfe_cols = train[cols].columns.values[rfe.get_support()]

    df_rfe_cols_cnt.loc[rfe_cols, 'cnt'] += 1

    

df_rfe_cols_cnt.plot(kind='bar', figsize=(15, 5))

plt.show()

#家族の数を追加したところ、Parch,SibSpの重要度が減少した。
# #Embarked,Parch,SibSpを除いてモデル選択を行う場合はこちらのコメントアウトを外す。今回は多い特徴量で検討

# x_cols = df_rfe_cols_cnt[df_rfe_cols_cnt['cnt'] >= 4].index

# x_cols
#Embarked,Parch,SibSpを除かずモデル選択を行う

x_cols = df_rfe_cols_cnt.index

x_cols
positive_cnt = int(train.Survived.sum())

rus = under_sampling.RandomUnderSampler(sampling_strategy={0:positive_cnt, 1:positive_cnt}, random_state=0)

data_x_sample, data_y_sample = rus.fit_sample(train[x_cols], train[['Survived']])



len(data_x_sample), len(data_y_sample), data.Survived.sum()
# 特徴量を選択して、複数のモデルで精度を調査する

import lightgbm as lgb

 

models = [

 

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

 

    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),

 

    #xgboost

    XGBClassifier(),

    lgb.LGBMClassifier()    

]

 

df_compare = pd.DataFrame(columns=['name', 'train_accuracy', 'valid_accuracy', 'time'])

scoring = ['accuracy']

 

for model in tqdm(models):

    

    name = model.__class__.__name__

    

    cv_rlts = model_selection.cross_validate(model, data_x_sample, data_y_sample, scoring=scoring, cv=10, return_train_score=True)

 

    for i in range(10):

        s = pd.Series([name, cv_rlts['train_accuracy'][i], cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name=name+str(i))

        df_compare = df_compare.append(s)

        

plt.figure(figsize=(12,8))

sns.boxplot(data=df_compare, y='name', x='valid_accuracy', orient='h', linewidth=0.5, width=0.5)

plt.grid()

plt.show()

# 精度の良いモデルを選んで、投票モデルを学習

 

vote_models = [

 

    #Ensemble Methods

    ('abc', ensemble.AdaBoostClassifier()),

    ('bc', ensemble.BaggingClassifier()),

    ('etsc', ensemble.ExtraTreesClassifier()),

    ('gbc', ensemble.GradientBoostingClassifier()),

    ('rfc', ensemble.RandomForestClassifier()),

 

    #Gaussian Processes

    #('gpc', gaussian_process.GaussianProcessClassifier()),

    

    #GLM

    ('lrcv', linear_model.LogisticRegressionCV()),

    #('rccv', linear_model.RidgeClassifierCV()), # unable soft voting

    

    #Navies Bayes

    #('bnb', naive_bayes.BernoulliNB()),

    #('gnb', naive_bayes.GaussianNB()),

    

    #Nearest Neighbor

    #('knc', neighbors.KNeighborsClassifier()),

    

    #Trees    

    #('dtc', tree.DecisionTreeClassifier()),

    #('etc', tree.ExtraTreeClassifier()),

    

    #Discriminant Analysis

    #('lda', discriminant_analysis.LinearDiscriminantAnalysis()),

    #('qda', discriminant_analysis.QuadraticDiscriminantAnalysis()),

 

    #xgboost

    ('xgbc', XGBClassifier()),

    

    #lightgbm

    ('lgbm',lgb.LGBMClassifier())

    

]

 

df_compare = pd.DataFrame(columns=['name', 'valid_accuracy', 'time'])

scoring = ['accuracy']

 

vote_hard_model = ensemble.VotingClassifier(estimators=vote_models, voting='hard')

cv_rlts = model_selection.cross_validate(vote_hard_model, data_x_sample, data_y_sample, cv=10, scoring=scoring)

for i in range(10):

    s = pd.Series(['hard', cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name='hard'+str(i))

    df_compare = df_compare.append(s)

    

vote_soft_model = ensemble.VotingClassifier(estimators=vote_models , voting='soft')

cv_rlts = model_selection.cross_validate(vote_soft_model, data_x_sample, data_y_sample, cv=10, scoring=scoring)

for i in range(10):

    s = pd.Series(['soft', cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name='soft'+str(i))

    df_compare = df_compare.append(s)

    

plt.figure(figsize=(12,3))

sns.boxplot(data=df_compare, y='name', x='valid_accuracy', orient='h', linewidth=0.5, width=0.5)

plt.grid()

plt.show()


# 各モデルのハイパーパラメータをグリッドサーチ



grid_n_estimator = [10, 50, 100, 300]

grid_ratio = [.1, .25, .5, .75, 1.0]

grid_learn = [.01, .03, .05, .1, .25]

grid_max_depth = [2, 4, 6, 8, 10, None]

grid_min_samples = [5, 10, .03, .05, .10]

grid_criterion = ['gini', 'entropy']

grid_bool = [True, False]

grid_seed = [0]



grid_param = [

    

    #AdaBoostClassifier

    [{ 

        'n_estimators': grid_n_estimator, #default=50

        'learning_rate': grid_learn, #default=1

        #'algorithm': ['SAMME', 'SAMME.R'], #default=’SAMME.R

        'random_state': grid_seed

    }],

    

    #BaggingClassifier

    [{

        'n_estimators': grid_n_estimator, #default=10

        'max_samples': grid_ratio, #default=1.0

        'random_state': grid_seed

     }],



    #ExtraTreesClassifier

    [{

        'n_estimators': grid_n_estimator, #default=10

        'criterion': grid_criterion, #default=”gini”

        'max_depth': grid_max_depth, #default=None

        'random_state': grid_seed

     }],



    #GradientBoostingClassifier

    [{

        #'loss': ['deviance', 'exponential'], #default=’deviance’

        'learning_rate': [.05], #default=0.1

        'n_estimators': [300], #default=100

        #'criterion': ['friedman_mse', 'mse', 'mae'], #default=”friedman_mse”

        'max_depth': grid_max_depth, #default=3   

        'random_state': grid_seed

     }],



    #RandomForestClassifier

    [{

        'n_estimators': grid_n_estimator, #default=10

        'criterion': grid_criterion, #default=”gini”

        'max_depth': grid_max_depth, #default=None

        'oob_score': [True], #default=False

        'random_state': grid_seed

     }],

    

    #LogisticRegressionCV

    [{

        'fit_intercept': grid_bool, #default: True

        #'penalty': ['l1','l2'],

        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], #default: lbfgs

        'random_state': grid_seed

     }],

    

    # ExtraTreeClassifier

    [{}],

    

    # LinearDiscriminantAnalysis

    [{}],

    

    #XGBClassifier

    [{

        'learning_rate': grid_learn, #default: .3

        'max_depth': [1,2,4,6,8,10], #default 2

        'n_estimators': grid_n_estimator, 

        'seed': grid_seed  

     }],

    #LGBMClassifier

    [{

        'learning_rate':grid_learn,

        'n_estimators':grid_n_estimator,

        'max_depth':[1,2,4,6,8,10],

        'min_child_weight':[0.5,1,2],

        'min_child_samples':[5,10,20],

        'subsample':[0.8],

        'colsample_bytree':[0.8],

        'verbose':[-1],

        'num_leaves':[80]}]

    

]



for model, param in tqdm(zip(vote_models, grid_param), total=len(vote_models)):

    

    best_search = model_selection.GridSearchCV(estimator=model[1], param_grid=param, scoring='roc_auc')

    best_search.fit(data_x_sample, data_y_sample)



    best_param = best_search.best_params_

    model[1].set_params(**best_param)
#　投票モデルの作成、softモデルとhardモデルの作成と検証

df_compare = pd.DataFrame(columns=['name', 'valid_accuracy', 'time'])

scoring = ['accuracy']

 

vote_hard_model = ensemble.VotingClassifier(estimators=vote_models, voting='hard')

cv_rlts = model_selection.cross_validate(vote_hard_model, data_x_sample, data_y_sample, cv=10, scoring=scoring)

for i in range(10):

    s = pd.Series(['hard',  cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name='hard'+str(i))

    df_compare = df_compare.append(s)

    

vote_soft_model= ensemble.VotingClassifier(estimators=vote_models , voting='soft')

cv_rlts = model_selection.cross_validate(vote_soft_model, data_x_sample, data_y_sample, cv=10, scoring=scoring)

for i in range(10):

    s = pd.Series(['soft',  cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name='soft'+str(i))

    df_compare = df_compare.append(s)

    

plt.figure(figsize=(12,3))

sns.boxplot(data=df_compare, y='name', x='valid_accuracy', orient='h',  linewidth=0.5, width=0.5)

plt.grid()

plt.show()
df_compare.groupby('name').mean().sort_values(by='valid_accuracy', ascending=False)

from sklearn.model_selection import KFold, train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(data_x_sample, data_y_sample, test_size=0.3, random_state=0)



vote_soft_model.fit(train_x, train_y)



pred = vote_soft_model.predict(valid_x)



fig, axs = plt.subplots(ncols=2,figsize=(15,5))



sns.heatmap(metrics.confusion_matrix(valid_y, pred), vmin=0, annot=True, fmt='d', ax=axs[0])

axs[0].set_xlabel('Predict')

axs[0].set_ylabel('Ground Truth')

axs[0].set_title('Accuracy: {}'.format(metrics.accuracy_score(valid_y, pred)))

fpr, tpr, thresholds = metrics.roc_curve(valid_y, pred)

axs[1].plot(fpr, tpr)

axs[1].set_title('ROC curve')

axs[1].set_xlabel('False Positive Rate')

axs[1].set_ylabel('True Positive Rate')

axs[1].grid(True)

plt.show()
vote_hard_model.fit(train_x, train_y)



pred = vote_hard_model.predict(valid_x)



fig, axs = plt.subplots(ncols=2,figsize=(15,5))



sns.heatmap(metrics.confusion_matrix(valid_y, pred), vmin=0, annot=True, fmt='d', ax=axs[0])

axs[0].set_xlabel('Predict')

axs[0].set_ylabel('Ground Truth')

axs[0].set_title('Accuracy: {}'.format(metrics.accuracy_score(valid_y, pred)))

fpr, tpr, thresholds = metrics.roc_curve(valid_y, pred)

axs[1].plot(fpr, tpr)

axs[1].set_title('ROC curve')

axs[1].set_xlabel('False Positive Rate')

axs[1].set_ylabel('True Positive Rate')

axs[1].grid(True)

plt.show()


# 動かすパラメータを明示的に表示

params = {"learning_rate":[0.1,0.3,0.5],

        "max_depth": [2,3,5,10],

         "subsample":[0.5,0.8,0.9,1],

         "colsample_bytree": [0.5,1.0],

         }

# モデルにインスタンス生成

mod = XGBClassifier()

# ハイパーパラメータ探索

gv =  model_selection.GridSearchCV(mod, params, cv = 10, scoring= 'roc_auc', n_jobs =-1)



#　trainデータとtestデータに分割

train_x, valid_x, train_y, valid_y = train_test_split(data_x_sample, data_y_sample, test_size=0.3, random_state=0)





# 予測モデルを作成

gv.fit(train_x,train_y)



gv =  model_selection.GridSearchCV(mod, params, cv = 10, scoring= 'roc_auc', n_jobs =-1)

cv_rlts = model_selection.cross_validate(gv, data_x_sample, data_y_sample, cv=10, scoring=scoring)

for i in range(10):

    s = pd.Series(['xgboost',  cv_rlts['test_accuracy'][i], cv_rlts['fit_time'][i]], index=df_compare.columns, name='xgb'+str(i))

    df_compare = df_compare.append(s)

    

plt.figure(figsize=(12,3))

sns.boxplot(data=df_compare, y='name', x='valid_accuracy', orient='h',  linewidth=0.5, width=0.5)

plt.grid()

plt.show()

train_x, valid_x, train_y, valid_y = train_test_split(data_x_sample, data_y_sample, test_size=0.3, random_state=0)



gv.fit(train_x, train_y)



pred = gv.predict(valid_x)



fig, axs = plt.subplots(ncols=2,figsize=(15,5))



sns.heatmap(metrics.confusion_matrix(valid_y, pred), vmin=0, annot=True, fmt='d', ax=axs[0])

axs[0].set_xlabel('Predict')

axs[0].set_ylabel('Ground Truth')

axs[0].set_title('Accuracy: {}'.format(metrics.accuracy_score(valid_y, pred)))

fpr, tpr, thresholds = metrics.roc_curve(valid_y, pred)

axs[1].plot(fpr, tpr)

axs[1].set_title('ROC curve')

axs[1].set_xlabel('False Positive Rate')

axs[1].set_ylabel('True Positive Rate')

axs[1].grid(True)

plt.show()
X_test.isnull().sum()
#最初はモデル評価のために均衡データの作成を行い、そのデータを分割しモデルの評価を行なった

#提出するモデルは訓練データを全て使い学習する。データ数多い方が精度良くなりそうだし。

from sklearn.model_selection import KFold, train_test_split

# train_x, valid_x, train_y, valid_y = train_test_split(data_x_sample, data_y_sample, test_size=0.3, random_state=0)



vote_soft_model.fit(X_train, y_train)



pred = vote_soft_model.predict(X_test)



sub = pd.DataFrame(pd.read_csv("/kaggle/input/titanic/test.csv")['PassengerId'])

sub['Survived'] = list(map(int, pred))

sub.to_csv("submission.csv", index = False)