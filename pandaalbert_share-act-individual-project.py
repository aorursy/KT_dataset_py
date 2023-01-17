import pandas as pd 

import numpy as np 

import os

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.impute import KNNImputer

from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit

import seaborn as sns 

import matplotlib as mpl 

import matplotlib.pyplot as plt

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import cross_validate

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
train=pd.read_csv('../input/bank_marketing_train.csv')

train['y'].value_counts()/len(train)

### distribution of y is No:0.88, Yes:0.11

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(train, train["y"]):

    strat_train_set = train.loc[train_index]

    strat_test_set = train.loc[test_index]

print('shape of strat train',strat_train_set.shape)

print('shape of strat test',strat_test_set.shape)

strat_train_set.to_csv('./strat_train_set.csv',index=False)

strat_test_set.to_csv('./strat_test_set.csv',index=False)
view=pd.read_csv('./strat_train_set.csv',index_col=False)

view.y=view.y.replace({'yes':1,'no':0})

view=view.replace({'unknown':np.nan})

view.head()
num_list=[]

for col in list(view.columns):

    if view[col].dtype != object:

        num_list.append(col)

view_num=view[num_list]

view_obj=view.drop(num_list,axis=1)
view_obj.isnull().sum()/view.shape[0]
view_num.isnull().sum()/view.shape[0]
view.hist(figsize=(20,20))
class campaign_age_unknown_trans(BaseEstimator,TransformerMixin):

    def fit(self,df,y=None):

        return self

    def transform(self,df):

        df=df.replace(to_replace={"unknown":np.nan})

        df.loc[(df['age']<0)|(df['age']>100),'age'] =  np.nan

        df['marital'] = df['marital'].replace(to_replace={"sungle":"single"})

        df.loc[(df['pdays'] ==999) & (df['poutcome'] !='nonexistent'),'pdays'] = np.nan

        q1,q3 = df['campaign'].quantile([0.25,0.75])

        lower = q1 - 3*(q3-q1)

        upper = q3 + 3*(q3-q1)

        df.loc[(df['campaign']<lower)|(df['campaign']>upper),'campaign'] = np.nan

        df.set_index('previous',inplace=True)

        df['campaign'] = df['campaign'].interpolate('linear')

        df.reset_index(inplace=True)

        df = df.assign(contacts_daily=(df['campaign']/(df['pdays']+1)).values)

        df[(df['age']>=60)&(pd.isnull(df.job))]['job']='retired'

        return df



class assign_educ_job_marital(BaseEstimator,TransformerMixin):

    def fit(self,df,y=None):

        return self 

    def transform(self,df):

        imp=SimpleImputer(strategy='most_frequent')

        df[['job']]=imp.fit_transform(df[['job']])

        df[['education']]=imp.fit_transform(df[['education']])

        df[['loan']]=imp.fit_transform(df[['loan']])

        df[['housing']]=imp.fit_transform(df[['housing']])

        df[['marital']]=imp.fit_transform(df[['marital']])

        return df



class fix_imbalance(BaseEstimator,TransformerMixin):

    def fit(self,df,y=None):

        return self 

    def transform(self,df):

        self.class_priors_pos = (df['y']  == 'yes').sum()

        self.class_priors_neg = (df['y']  == 'no').sum()

        self.df_pos = df[df['y'] == 'yes']

        self.df_neg = df[df['y']  == 'no']

        self.df_pos_over = self.df_pos.sample(int(0.5*self.class_priors_neg), replace=True)

        df = pd.concat([self.df_pos_over,self.df_neg])

        return df
handle_pipeline = Pipeline([

        ('step1', campaign_age_unknown_trans()),

        ('step2', assign_educ_job_marital()),

        ('step3', fix_imbalance()),

    ])

handle_pipeline_test = Pipeline([

        ('step1', campaign_age_unknown_trans()),

        ('step2', assign_educ_job_marital()),

    ])



strat_train_set=pd.read_csv('./strat_train_set.csv',index_col=False)

strat_test_set=pd.read_csv('./strat_test_set.csv',index_col=False)



train=handle_pipeline.fit_transform(strat_train_set)

test=handle_pipeline_test.fit_transform(strat_test_set)
# train['age']=pd.cut(train['age'],bins=np.linspace(15,100,num=18))

# test['age']=pd.cut(test['age'],bins=np.linspace(15,100,num=18))

num_list=['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contacts_daily']

train_num_group=train[num_list]

test_num_group=test[num_list]
train['y']=train['y'].replace({'yes':True,'no':False})

test['y']=test['y'].replace({'yes':True,'no':False})
train=train.drop(num_list,axis=1)

test=test.drop(num_list,axis=1)
num_imp=SimpleImputer(strategy='mean')

train_num_group=num_imp.fit_transform(train_num_group)

test_num_group=num_imp.fit_transform(test_num_group)

std_=StandardScaler()

train_num_group=std_.fit_transform(train_num_group)

test_num_group=std_.fit_transform(test_num_group)
i=1

plt.figure(figsize=(20,30))

for col in list(train.columns)[:-1]:

    tmp=train[[col,'y']].groupby(col).sum().reset_index()

    tmp['y']=tmp['y']/train.shape[0]

    plt.subplot(5,2,i)

    i+=1

    plt.bar(tmp[col],tmp['y'])

    plt.xticks(rotation=25)

    plt.title(col+' plot')
one_hot_groupname=['housing','day_of_week']

one_hot_train=train[one_hot_groupname]

one_hot_test=test[one_hot_groupname]

train=train.drop(one_hot_groupname,axis=1)

test=test.drop(one_hot_groupname,axis=1)

imp=SimpleImputer(strategy='most_frequent')

one_hot_train=imp.fit_transform(one_hot_train)

one_hot_test=imp.transform(one_hot_test)

one_hot=OneHotEncoder(handle_unknown='ignore')

one_hot_train=one_hot.fit_transform(one_hot_train)

one_hot_train=one_hot_train.toarray()

one_hot_test=one_hot.transform(one_hot_test)

one_hot_test=one_hot_test.toarray()
encoder_columns=list(train.columns)

encoder_columns=encoder_columns[:-1]

for col_name in encoder_columns:

    df_change=train[[col_name,'y']]

    df_change=df_change.groupby(col_name).mean().sort_values('y').reset_index()

    num=1

    match_dict=dict()

    for i in df_change.iloc[:,0]:

        match_dict[i]=num

        num+=1

        train[col_name]=train[col_name].replace(match_dict)

        test[col_name]=test[col_name].replace(match_dict)
train_label=train['y']

test_label=test['y']

train=train.drop('y',axis=1)

test=test.drop('y',axis=1)

train_data=np.concatenate([train_num_group,one_hot_train,train.values],axis=1)

test_data=np.concatenate([test_num_group,one_hot_test,test.values],axis=1)

knn_imp=KNNImputer(n_neighbors=200)

train_data=knn_imp.fit_transform(train_data)

test_data=knn_imp.transform(test_data)
clf = RandomForestClassifier(random_state = 50, n_jobs = -1)

best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,

                        param_grid={'n_estimators': [200,340,500]})

best_clf.fit(train_data,train_label)

print("Select best RandomForest model with n_estimators = {} with best_score={}".format(

    best_clf.best_params_['n_estimators'],

    best_clf.best_score_))
test_clf=RandomForestClassifier(random_state = 1, n_jobs = -1,n_estimators=best_clf.best_params_['n_estimators'])

test_clf.fit(train_data,train_label)

print("The AUC_ROC of the best random forest on test data is",

      roc_auc_score(test_label.tolist(),test_clf.predict(test_data).tolist()))
clf = LogisticRegression()

best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,

                        param_grid={'C': [0.001,0.01,0.1,1,10,100]})

best_clf.fit(train_data,train_label)

print("Select best Logistic Regression model with C = {} with best_score={}".format(

    best_clf.best_params_['C'],

    best_clf.best_score_))
for c in  [0.001,0.01,0.1,1,10,100]:

    test_clf=LogisticRegression(C=c,max_iter=500)

    test_clf.fit(train_data,train_label)

    print("The AUC_ROC of the logistic regression with {} on test data is".format(c),

        roc_auc_score(test_label.tolist(),test_clf.predict(test_data).tolist()))
clf=MLPClassifier(hidden_layer_sizes=(30,10),random_state=42,activation='logistic')

best_clf = GridSearchCV(clf,scoring='roc_auc',cv=5,n_jobs=-1,

                        param_grid={'alpha': [0.05,0.08,0.1,0.12]})

best_clf.fit(train_data,train_label)

print("Select best neural network model with alpha = {} with best_score={}".format(

    best_clf.best_params_['alpha'],

    best_clf.best_score_))
for lay in [(40,20),(50,20),(45,15)]:

    for a in [0.001,0.05,0.08,0.1,0.12,0.5,1]:

        nn_clf=MLPClassifier(random_state = 42,activation='logistic',alpha=a,hidden_layer_sizes=lay,max_iter=500)

        nn_clf.fit(train_data,train_label)

        print("The AUC_ROC of the nn model with lay={}, alpha={} on test data is".format(lay,a),

            roc_auc_score(test_label.tolist(),nn_clf.predict(test_data).tolist()))
nn_clf_1=MLPClassifier(random_state = 42,activation='logistic',alpha=0.1,hidden_layer_sizes=(50,20))

nn_clf_2=MLPClassifier(random_state = 42,activation='logistic',alpha=0.1,hidden_layer_sizes=(40,20))

nn_clf_3=MLPClassifier(random_state = 42,activation='logistic',alpha=0.08,hidden_layer_sizes=(40,20))

voting_clf=VotingClassifier(estimators=[('nn1',nn_clf_1),('nn2',nn_clf_2),('nn3',nn_clf_3)],voting='soft',weights=[0.33,0.33,0.34])

cv_res=cross_validate(voting_clf,train_data,train_label,scoring='roc_auc',cv=5)
sc=cv_res['test_score']

print(np.mean(sc))

print(np.std(sc))