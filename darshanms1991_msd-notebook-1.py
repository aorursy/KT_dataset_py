# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
sub_df = pd.read_csv('../input/sample_submission.csv')
sub_df.head()
train_df = pd.read_csv('../input/act_train.csv')
train_df['activity_category'] = train_df['activity_category'].astype('category').cat.codes

columns = ['char_'+str(i) for i in range(1,11)]

train_df[columns] = train_df[columns].apply(lambda x: x.astype('category').cat.codes)

train_df['date'] = pd.to_datetime(train_df['date'])

train_df['day'] = train_df['date'].apply(lambda x:x.day)

train_df['year'] = train_df['date'].apply(lambda x:x.year)

train_df['month'] = train_df['date'].apply(lambda x:x.month)

train_df = train_df.drop(['date'],axis = 1)
train_df.corr().outcome
people_df = pd.read_csv('../input/people.csv')

people_df.group_1.unique().shape
people_df = pd.read_csv('../input/people.csv')

columns = ['char_'+str(i) for i in range(1,10)]

people_df[columns] = people_df[columns].apply(lambda x: x.astype('category').cat.codes)

people_df['group_1'] = people_df['group_1'].astype('category').cat.codes

people_df['date'] = pd.to_datetime(people_df['date'])

people_df['day'] = people_df['date'].apply(lambda x:x.day)

people_df['year'] = people_df['date'].apply(lambda x:x.year)

people_df['month'] = people_df['date'].apply(lambda x:x.month)

people_df = people_df.drop(['date'],axis = 1)

people_df = people_df.set_index(people_df['people_id'])

people_df.head()
train_X = train_df.join(people_df,on = 'people_id', rsuffix='_people')
Y = train_X['outcome']

X = train_X.drop(['outcome','people_id','people_id_people','activity_id'],axis = 1)
#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(n_estimators = 10)

#clf = clf.fit(X, Y)

import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X, Y)
test_df = pd.read_csv('../input/act_test.csv')

test_df['activity_category'] = test_df['activity_category'].astype('category').cat.codes

columns = ['char_'+str(i) for i in range(1,11)]

test_df[columns] = test_df[columns].apply(lambda x: x.astype('category').cat.codes)

test_df['date'] = pd.to_datetime(test_df['date'])

test_df['day'] = test_df['date'].apply(lambda x:x.day)

test_df['year'] = test_df['date'].apply(lambda x:x.year)

test_df['month'] = test_df['date'].apply(lambda x:x.month)

test_df = test_df.drop(['date'],axis = 1)
test_X = test_df.join(people_df,on = 'people_id', rsuffix='_people')

X = test_X.drop(['people_id','people_id_people','activity_id'],axis = 1)

#output = clf.predict(X)

output = gbm.predict(X)
test_df['outcome'] = output

test_df.to_csv('submission.csv',columns = ['activity_id','outcome'],index = False)
sum(output)