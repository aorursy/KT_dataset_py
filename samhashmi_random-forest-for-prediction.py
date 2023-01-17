### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as pl

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def joinData(read_file, read_train_data, read_test_data):

    read_test_data['outcome'] = np.nan

    combine=pd.concat([read_train_data, read_test_data], axis=0)

    combine=combine[['people_id', 'activity_id', 'outcome', 'date']]

    merge_ppl_act=pd.merge(left=combine,right=read_file, how='right', on='people_id')

    return merge_ppl_act;
def featureEncoding(merge_ppl_act):

    

    for i in merge_ppl_act.columns:

        if(i.find('char')==0):

            category={}

            for p,j in enumerate(merge_ppl_act[i].unique()):

                category[j]=p

            merge_ppl_act[i]=merge_ppl_act[i].map(category)

    group={}

    for i,j in enumerate(merge_ppl_act.group_1.unique()):

        group[j]=i

    merge_ppl_act['group_1']=merge_ppl_act.group_1.map(group)

    return merge_ppl_act;
def change_date_to_Feature(merge_ppl_act):

    merge_ppl_act['year_x']=merge_ppl_act.date_x.dt.year

    merge_ppl_act['month_x']=merge_ppl_act.date_x.dt.month

    merge_ppl_act['day_x']=merge_ppl_act.date_x.dt.day

    merge_ppl_act['year_y']=merge_ppl_act.date_y.dt.year

    merge_ppl_act['month_y']=merge_ppl_act.date_y.dt.month

    merge_ppl_act['day_y']=merge_ppl_act.date_y.dt.day

    del merge_ppl_act['date_x']

    del merge_ppl_act['date_y']

    return merge_ppl_act;
def remove_extra_Features(merge_ppl_act):

    del merge_ppl_act['activity_id']

    del merge_ppl_act['people_id']

    return merge_ppl_act;
def training_testing(x_train, x_test, y_train, y_test, testing):

    cv=KFold(len(y_train), n_folds=8)

    fold=1

    for train, test in cv:

        training=RandomForestClassifier(n_estimators=20).fit(x_train.iloc[train], y_train.iloc[train])

        prediction=training.predict_proba(x_train.iloc[test])

        fpr, tpr, thresholds = roc_curve(y_train.iloc[test], prediction[:, 1])

        roc_auc = auc(fpr, tpr)

        pl.plot(fpr, tpr, label='ROC curve (fold %1i) (area = %0.4f)' % (fold, roc_auc))

        print(fold, roc_auc)

        print (training.feature_importances_)

        fold+=1



    final=training.predict_proba(x_test)

    fpr,tpr, thresholds=roc_curve(y_test, final[:,1])

    roc_auc=auc(fpr, tpr)

    print(roc_auc)

    pl.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % (roc_auc))

    return training.predict(testing)
x=training.predict(x_test)

len(y_test[x==y_test]), len(x)
#if __name__=="__main__":

read_file=pd.read_csv('../input/people.csv',  sep=",", parse_dates=['date'])

read_train_data=pd.read_csv('../input/act_train.csv', sep=",", dtype={'outcome':np.int8}, parse_dates=['date'])

read_test_data=pd.read_csv('../input/act_test.csv', sep=",", parse_dates=['date'])
combined_frame=joinData(read_file, read_train_data, read_test_data)

encoded=featureEncoding(combined_frame)

date_features=change_date_to_Feature(encoded)

feature_engineering=remove_extra_Features(date_features)

testing_data=feature_engineering[feature_engineering.outcome.isnull()]

feature_engineering=feature_engineering.dropna(axis=0)

feature_engineering.outcome=feature_engineering.outcome.astype(np.int64)

outcome=feature_engineering.outcome

del feature_engineering['outcome']
x_train, x_test, y_train, y_test=train_test_split(feature_engineering, outcome, test_size=0.20, random_state=42)

prediction=training_testing(x_train, x_test, y_train, y_test, testing_data.drop(['outcome'], axis=1))
prediction