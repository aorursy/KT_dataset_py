import math

import numpy as np

import pandas as pd
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test.insert(1,'target',0)

print(train.shape)

print(test.shape)
x = pd.concat([train,test])

x = x.reset_index(drop=True)

unwanted = x.columns[x.columns.str.startswith('ps_calc_')]

x.drop(unwanted,inplace=True,axis=1)
features = x.columns[2:]

categories = []

for c in features:

    trainno = len(x.loc[:train.shape[0],c].unique())

    testno = len(x.loc[train.shape[0]:,c].unique())

    print(c,trainno,testno)
x.loc[:,'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50,labels=False)

x.loc[:,'ps_car_12'] = pd.cut(x['ps_car_12'], 50,labels=False)

x.loc[:,'ps_car_13'] = pd.cut(x['ps_car_13'], 50,labels=False)

x.loc[:,'ps_car_14'] =  pd.cut(x['ps_car_14'], 50,labels=False)

x.loc[:,'ps_car_15'] =  pd.cut(x['ps_car_15'], 50,labels=False)
test = x.loc[train.shape[0]:].copy()

train = x.loc[:train.shape[0]].copy()
#Always good to shuffle for SGD type optimizers

train = train.sample(frac=1).reset_index(drop=True)
train.drop('id',inplace=True,axis=1)

test.drop('id',inplace=True,axis=1)
train.head()
test.head()
categories = train.columns[1:]

numerics = []
currentcode = len(numerics)

catdict = {}

catcodes = {}

for x in numerics:

    catdict[x] = 0

for x in categories:

    catdict[x] = 1



noofrows = train.shape[0]

noofcolumns = len(features)

with open("alltrainffm.txt", "w") as text_file:

    for n, r in enumerate(range(noofrows)):

        if((n%100000)==0):

            print('Row',n)

        datastring = ""

        datarow = train.iloc[r].to_dict()

        datastring += str(int(datarow['target']))





        for i, x in enumerate(catdict.keys()):

            if(catdict[x]==0):

                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])

            else:

                if(x not in catcodes):

                    catcodes[x] = {}

                    currentcode +=1

                    catcodes[x][datarow[x]] = currentcode

                elif(datarow[x] not in catcodes[x]):

                    currentcode +=1

                    catcodes[x][datarow[x]] = currentcode



                code = catcodes[x][datarow[x]]

                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

        datastring += '\n'

        text_file.write(datastring)

        

noofrows = test.shape[0]

noofcolumns = len(features)

with open("alltestffm.txt", "w") as text_file:

    for n, r in enumerate(range(noofrows)):

        if((n%100000)==0):

            print('Row',n)

        datastring = ""

        datarow = test.iloc[r].to_dict()

        datastring += str(int(datarow['target']))





        for i, x in enumerate(catdict.keys()):

            if(catdict[x]==0):

                datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])

            else:

                if(x not in catcodes):

                    catcodes[x] = {}

                    currentcode +=1

                    catcodes[x][datarow[x]] = currentcode

                elif(datarow[x] not in catcodes[x]):

                    currentcode +=1

                    catcodes[x][datarow[x]] = currentcode



                code = catcodes[x][datarow[x]]

                datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

        datastring += '\n'

        text_file.write(datastring)

# sub = pd.read_csv('../input/sample_submission.csv')

# outputs = pd.read_csv('output.txt',header=None)

# outputs.columns = ['target']

# sub.target = outputs.target.ravel()

# sub.to_csv('libffmsubmission.csv',index=False)