

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt



from sklearn.model_selection import cross_val_score, KFold



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/training_variants')

train_df.shape
train_df.head()
test_df = pd.read_csv('../input/test_variants')

test_df.shape
train_text_df = pd.DataFrame(columns=['ID', 'Text'])

test_text_df = pd.DataFrame(columns=['ID', 'Text'])
import sys

import csv

maxInt = sys.maxsize

decrement = True



while decrement:

    # decrease the maxInt value by factor 10 

    # as long as the OverflowError occurs.



    decrement = False

    try:

        csv.field_size_limit(maxInt)

    except OverflowError:

        maxInt = int(maxInt/10)

        decrement = True



with open('../input/training_text', 'r') as training_text:

    train_text = csv.reader((line.replace('||', '|') for line in training_text), delimiter='|')

    i=0

    for row in train_text:

        if i > 0:

            train_text_df.loc[len(train_text_df)] = [row[0], row[1]]

        i = i+1

        

with open('../input/test_text', 'r') as test_text:

    test_text = csv.reader((line.replace('||', '|') for line in test_text), delimiter='|')

    i=0

    for row in test_text:

        if i > 0:

            test_text_df.loc[len(test_text_df)] = [row[0], row[1]]

        i = i+1
train_text_df.head()
train_y = train_df['Class']
train_y_df = train_y.reset_index()

cnt_y = train_y_df.Class.value_counts().reset_index()

cnt_y.columns = ['Class', 'Count']

total_y = cnt_y.Count.sum()

cnt_y["Percentage"] =(cnt_y.Count/total_y)*100



cnt_y.plot(x='Class', y='Percentage', kind='bar', figsize=(16, 10))



for idx, c in cnt_y.iterrows():

    print('Class ', c.Class,': ', c.Percentage, '%')
for i in range(1, 9):

    classs = train_df.loc[train_df['Class'] == i]



    cnt_genes = classs.Gene.value_counts().reset_index()

    cnt_genes.columns = ['Gene', 'Count']

    

    total = cnt_genes.Count.sum()

    

    cnt_genes['Percentage'] = (cnt_genes.Count/total)*100



    cnt_genes.loc[cnt_genes['Count'] > 2].plot(x='Gene', y='Percentage', kind='bar', figsize=(12, 8))
s = train_df.Gene.value_counts().reset_index()

s.columns=['Gene', 'Count']
cnt = s.Count.value_counts().reset_index()

cnt.columns = ['Occurence', 'Count']

total = cnt.Count.sum()



for idx, c in cnt.iterrows():

    print((c.Count/total)*100, "% having ", c.Occurence)
s.loc[s['Count'] > 25].plot(x='Gene', y='Count', kind='bar', figsize=(16, 10))
brca1 = train_df.loc[train_df['Gene']=='BRCA1']



val = brca1.Class.value_counts().reset_index()

val.columns=['Class', 'Count']



total = val.Count.sum()

val['percentage'] = (val.Count/total)*100



val.plot(x='Class', y='percentage', kind='bar', figsize=(16, 10))



for idx, c in val.iterrows():

    print('Class ', c.Class,': ', c.percentage, '%')
brca2 = train_df.loc[train_df['Gene']=='BRCA2']



val = brca2.Class.value_counts().reset_index()

val.columns=['Class', 'Count']



total = val.Count.sum()

val['percentage'] = (val.Count/total)*100



val.plot(x='Class', y='percentage', kind='bar', figsize=(16, 10))



for idx, c in val.iterrows():

    print('Class ', c.Class,': ', c.percentage, '%')
tp53 = train_df.loc[train_df['Gene']=='TP53']



val = tp53.Class.value_counts().reset_index()

val.columns=['Class', 'Count']



total = val.Count.sum()

val['percentage'] = (val.Count/total)*100



val.plot(x='Class', y='percentage', kind='bar', figsize=(16, 10))



for idx, c in val.iterrows():

    print('Class ', c.Class,': ', c.percentage, '%')
pten = train_df.loc[train_df['Gene']=='PTEN']



val = pten.Class.value_counts().reset_index()

val.columns=['Class', 'Count']



total = val.Count.sum()

val['percentage'] = (val.Count/total)*100



val.plot(x='Class', y='percentage', kind='bar', figsize=(16, 10))



for idx, c in val.iterrows():

    print('Class ', c.Class,': ', c.percentage, '%')
vart = train_df.Variation.value_counts().reset_index()

vart.columns=['Variation', 'Count']
train_df.Variation.head()
cnt_vart = vart.Count.value_counts().reset_index()

cnt_vart.columns = ['Occurence', 'Count']

total_vart = cnt_vart.Count.sum()



for idx, c in cnt_vart.iterrows():

    print((c.Count/total_vart)*100, "% having ", c.Occurence, ' occurence')
vart.loc[vart['Count'] > 25].plot(x='Variation', y='Count', kind='bar', figsize=(16, 10))
genes_25 = s.loc[s['Count'] > 25]



train1 =  train_df[train_df['Gene'].isin(genes_25.Gene)]

test1 = test_df[test_df['Gene'].isin(genes_25.Gene)]



train1.head()
train1 = train1.drop('ID', 1)

train1 = train1.drop('Variation', 1)



test1 = test1.drop('ID', 1)

test1 = test1.drop('Variation', 1)
y = train1['Class']

train1 = train1.drop('Class', 1)



num_train = len(train1)



df_all = pd.concat([train1, test1])

df_all = pd.get_dummies(df_all, drop_first=True)



new_train = df_all[:num_train]

new_test = df_all[num_train:]
from sklearn.ensemble import RandomForestClassifier



model_rfr = RandomForestClassifier(n_estimators=500, max_depth=3)



importances = model_rfr.fit(new_train, y).feature_importances_

features = pd.DataFrame()

features['feature'] = new_train.columns

features['importance'] = importances



features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)



features[features.size-100:].plot(kind='barh', figsize=(12,24))
todrop = features.loc[features['importance'] == 0].index

new_train_df = new_train.drop(todrop, 1)

new_test_df = new_test.drop(todrop, 1)



print('Train shape: ', new_train_df.shape)

print('Test shape: ', new_test_df.shape)
import xgboost as xgb



model_xgb = xgb.XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.01, min_child_weight=5, subsample=1.0, colsample_bytree=1.0, reg_alpha=0.01)



results = cross_val_score(model_xgb, new_train_df, y, cv=10, scoring='accuracy')

print("XGB score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))

print(results)
fold = KFold(n_splits=10)



i=1



for train_index, test_index in fold.split(np.array(new_train_df)):



    if i==7:

          print(np.array(new_train_df)[test_index])

    i = i + 1

    