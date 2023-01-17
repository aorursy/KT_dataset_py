#install the fastai library version 0.7 as used in the fastai course

!pip install fastai==0.7.0 --no-deps
#import libraries, this code follows the imports from the fastai course

import os

%matplotlib inline

from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier

from IPython.display import display



from sklearn import metrics

np.random.seed(20190810)
#load data, we only need the transactions dataset for now

#train_id = pd.read_csv('/kaggle/input/train_identity.csv')

train_trans = pd.read_csv('/kaggle/input/train_transaction.csv')

#test_id = pd.read_csv('/kaggle/input/test_identity.csv')

test_trans = pd.read_csv('/kaggle/input/test_transaction.csv')

#sample_submission = pd.read_csv('/kaggle/input/sample_submission.csv')
#the proc_df installed in kaggle was not working properly so copied the definition from my pc

def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):



    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res
#testing for some sort of time order in the data set

experiment_set = train_trans.tail(100000)

last_20k = experiment_set.tail(20000)

first_80k = experiment_set.head(80000)

random_20k = first_80k.sample(20000)

experiment_train = first_80k[~first_80k.TransactionID.isin(random_20k.TransactionID)]

def print_score(m,X_train,y_train,X_valid,y_valid):

    res = {'auc_train' : metrics.roc_auc_score(y_train,m.predict_proba(X_train)[:,1]),

           'auc_valid' : metrics.roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1])}

    if hasattr(m, 'oob_score_'): res["oob"] = m.oob_score_

    print(res)
train_cats(experiment_train)

apply_cats(df=random_20k, trn=experiment_train)



X, y , nas = proc_df(experiment_train, 'isFraud')

random_valid, random_valid_y, _ = proc_df(random_20k,'isFraud', na_dict=nas)





m = RandomForestClassifier(n_jobs=-1, n_estimators=10)



m.fit(X, y)



print_score(m,X,y,random_valid,random_valid_y)
train_cats(experiment_train)

apply_cats(df=last_20k, trn=experiment_train)



X, y , nas = proc_df(experiment_train, 'isFraud')



last_valid, last_valid_y, _= proc_df(last_20k,'isFraud', na_dict=nas)





m = RandomForestClassifier(n_jobs=-1, n_estimators=10)



m.fit(X, y)



print_score(m,X,y, last_valid, last_valid_y)
#create a data frame from the TransactionDT columns of the train and test set and add an index to make plotting easy.

df1 = pd.DataFrame(train_trans.TransactionDT)

df1['datasource'] = "train"

df2 = pd.DataFrame(test_trans.TransactionDT)

df2['datasource'] = "test"

df = pd.concat([df1,df2])

df = df.reset_index()
#create a sampled scatterplot showing the distribution

sns.scatterplot(x='TransactionDT', y='index',hue='datasource', data=df.sample(100000),edgecolors='none',marker='.')

plt.axvline(df1.TransactionDT.max())

plt.axvline(df2.TransactionDT.min())

plt.title("Ranges of the TransactionDT values in the two data sets with lines showing max and min of the two datasets")
#need to delete objects from memory in order to free up memory for model training.

del[df1,df2,df,last_valid,last_valid_y,random_valid,random_valid_y,experiment_set,last_20k,first_80k,random_20k,experiment_train,m]
tt_train = train_trans.head(500000)

tt_valid = train_trans.tail(90540)



train_cats(tt_train)

apply_cats(df=tt_valid, trn=tt_train)



X, y , nas = proc_df(tt_train, 'isFraud')



valid, valid_y, _= proc_df(tt_valid,'isFraud', na_dict=nas)



apply_cats(df=test_trans, trn=tt_train)

X_test, _ , _= proc_df(test_trans,na_dict=nas)
del([test_trans,train_trans,tt_train,tt_valid])
mrf1 = RandomForestClassifier(n_jobs=-1, n_estimators=10)

mrf1.fit(X, y)



#output predictions and make submission files

mrf1_sub = pd.DataFrame()

mrf1_sub['TransactionID']=X_test['TransactionID']

mrf1_sub['isFraud']=mrf1.predict_proba(X_test)[:,1]

mrf1_sub.to_csv('mrf1_submission.csv', index=False)

print_score(mrf1,X,y, valid, valid_y)
del([mrf1,mrf1_sub])
mrf2 = RandomForestClassifier(n_jobs=-1, n_estimators=10,min_samples_leaf=1000)

mrf2.fit(X, y)



mrf2_sub = pd.DataFrame()

mrf2_sub['TransactionID']=X_test['TransactionID']

mrf2_sub['isFraud']=mrf2.predict_proba(X_test)[:,1]

mrf2_sub.to_csv('mrf2_submission.csv', index=False)

print_score(mrf2,X,y, valid, valid_y)
del([mrf2,mrf2_sub])
mgb1 = GradientBoostingClassifier(n_estimators=10)

mgb1.fit(X, y)



mgb1_sub = pd.DataFrame()

mgb1_sub['TransactionID']=X_test['TransactionID']

mgb1_sub['isFraud']=mgb1.predict_proba(X_test)[:,1]

mgb1_sub.to_csv('mgb1_submission.csv', index=False)

print_score(mgb1,X,y, valid, valid_y)
del([mgb1,mgb1_sub])
mgb2 = GradientBoostingClassifier(n_estimators=10,min_samples_leaf=1000)

mgb2.fit(X, y)



mgb2_sub = pd.DataFrame()

mgb2_sub['TransactionID']=X_test['TransactionID']

mgb2_sub['isFraud']=mgb2.predict_proba(X_test)[:,1]

mgb2_sub.to_csv('mgb2_submission.csv', index=False)

print_score(mgb2,X,y, valid, valid_y)
del([mgb2,mgb2_sub])
outcome = pd.DataFrame({'valid':[0.8273,0.8599,0.8044,0.8105],

          'public':[0.8506,0.8826, 0.8386,0.8424]})

outcome.plot.scatter(x='valid',y='public')

plt.title('Performance of different models on public leaderboard and validation set')