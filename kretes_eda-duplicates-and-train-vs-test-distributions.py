import pandas as pd

import numpy as np

import seaborn as sns

baseDir = "../input/"

people = pd.read_csv('{0}people.csv'.format(baseDir)).drop_duplicates()

act_train = pd.read_csv('{0}act_train.csv'.format(baseDir)).drop_duplicates()

act_test = pd.read_csv('{0}act_test.csv'.format(baseDir)).drop_duplicates()
print(people.shape)

print(act_train.shape)

print(act_test.shape)
print("{0} duplicate people rows".format(people.drop('people_id',axis=1).duplicated().sum()))

print("{0} duplicate people ids".format(people['people_id'].duplicated().sum()))

print("{0} duplicate train rows".format(act_train.drop('activity_id',axis=1).duplicated().sum()))

print("{0} duplicate train rows with different outcome".format(

        act_train.drop(['activity_id'],axis=1).drop_duplicates().drop('outcome',axis=1).duplicated().sum()))

print("{0} duplicate train activity id".format(act_train['activity_id'].duplicated().sum()))

print("{0} duplicate test rows".format(act_test.drop('activity_id',axis=1).duplicated().sum()))

print("{0} duplicate test activity id".format(act_test['activity_id'].duplicated().sum()))
unqTrain = act_train.drop(['activity_id','outcome'],axis=1).drop_duplicates()

unqTest = act_test.drop(['activity_id'],axis=1).drop_duplicates()

total = pd.concat([unqTrain,unqTest],axis=0)

print("{0} rows duplicated between train and test".format(len(total) - len(total.drop_duplicates())))

print("{0} columns diff between train and test".format([c for c in act_train.columns if c not in act_test.columns and c!='outcome']))
def addPrefix(df,suffix, exclude):

    for c in df.columns:

        if c not in exclude:

            df.rename(columns={c:suffix+c},inplace=True)
addPrefix(people,'ppl_',['people_id'])

addPrefix(act_train,'act_',['people_id','activity_id','activity_category','outcome'])

addPrefix(act_test,'act_',['people_id','activity_id','activity_category'])
train = pd.merge(act_train,people, on='people_id', how='left')

print(train.shape)

test = pd.merge(act_test,people, on='people_id', how='left')

print(test.shape)

print(train.columns)
trainUnique = train[~train.drop(['people_id','activity_id'],axis=1).duplicated()]

print(trainUnique.shape)

testUnique = test[~test.drop(['people_id','activity_id'],axis=1).duplicated()]

print(testUnique.shape)
nonCategoricalColumns = ['people_id','activity_id','outcome','ppl_char_38','ppl_date','act_date']

valCounts = {}

def calcCountSuffix(df,exclude):

    for c in df.columns:

        if c not in exclude:

            cnt = len(df[c].value_counts())

            valCounts[c] = cnt

calcCountSuffix(trainUnique,nonCategoricalColumns)
def addCountSuffix(df,exclude):

    for c in df.columns:

        if c not in exclude:

            cnt = valCounts[c]

            df.rename(columns={c:c+"_cnt_"+str(cnt)},inplace=True)

addCountSuffix(train,nonCategoricalColumns)

addCountSuffix(test,nonCategoricalColumns)

addCountSuffix(trainUnique,nonCategoricalColumns)

addCountSuffix(testUnique,nonCategoricalColumns)
def getColumnsBySuffix(df,minValue,maxValue,exclude):

    return [c for c in df.columns if c not in exclude if int(c.split("_")[-1])>=minValue and int(c.split("_")[-1])<=maxValue]

def drawViolin(df, minCnt,maxCnt,indexFrom, indexTo, size=3.5):

    g = sns.PairGrid(df,

                 x_vars=getColumnsBySuffix(train,minCnt,maxCnt,nonCategoricalColumns)[indexFrom:indexTo],

                 y_vars=["outcome"],

                 aspect=.75, size=size)

    g.map(sns.violinplot, palette="pastel");
sam10k = trainUnique.sample(10000)

sam100k = trainUnique.sample(100000)

sam500k = trainUnique.sample(500000)
drawViolin(sam10k,2,2,0,6)
drawViolin(sam10k,2,2,6,11)
drawViolin(sam10k,2,2,11,16)
drawViolin(sam10k,2,2,16,21)
drawViolin(sam10k,2,2,21,26)
drawViolin(sam10k,2,2,26,31)
drawViolin(sam100k,3,6,0,6,8.0)
drawViolin(sam100k,6,7,0,5,5.0)
drawViolin(sam100k,8,8,0,5,8.0)
drawViolin(sam100k,9,9,0,5,8.0)
getColumnsBySuffix(trainUnique,10,10000000,exclude=nonCategoricalColumns)
def createDataForDistributionsPlot():

    train['set'] = 'train'

    trainUnique['set'] = 'trainUnique'

    test['set'] = 'test'

    return pd.concat([train,trainUnique,test],axis=0)
trainAndTest = createDataForDistributionsPlot()
def drawDistributions(column):

    gb = trainAndTest.groupby([c,'set'],as_index=False).count()[[c,'set','activity_id']]

    gb['c_freq'] = gb['activity_id'] / np.where(gb['set'] == 'train',len(train),np.where(gb['set'] == 'trainUnique',len(trainUnique),len(test)))

    sns.barplot(x=c, y='c_freq', hue='set', hue_order=['train','trainUnique','test'], data=gb)    
import matplotlib.pyplot as plt

for c in getColumnsBySuffix(trainAndTest,2,2,exclude=nonCategoricalColumns + ['set']):

    drawDistributions(c)

    plt.show()
import matplotlib.pyplot as plt

for c in getColumnsBySuffix(trainAndTest,3,52,exclude=nonCategoricalColumns + ['set']):

    drawDistributions(c)

    plt.show()