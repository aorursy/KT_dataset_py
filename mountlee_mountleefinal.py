from __future__ import division
import string
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import pandas as pd
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
%matplotlib inline

        
file = open("../input/train.csv")
fout = open('subset_datatest.csv','w')
n = 0
for line in file:
    if n == 0:
        fout.write(line)
    if n <400000*5:
        n +=1
    elif 400000*5<=n <400000*10:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()
file = open("../input/train.csv")
fout = open('subset_datatrain.csv','w')
n = 0
for line in file:
    if n <400000*5:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()


featurelist = ['user_id','user_location_city','orig_destination_distance','srch_destination_id','hotel_market','srch_ci']
whlist = ['user_id','user_location_city','orig_destination_distance','srch_destination_id','hotel_market','srch_ci','hotel_cluster']


trainpart = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = whlist)
traindata = trainpart[featurelist].fillna(0).values
trainpara = {'data':traindata,'feature_names':featurelist,'target':trainpart['hotel_cluster'].values,
'target_names':np.arange(100)}

testpart = pd.read_csv('subset_datatest.csv',na_values=['--  '],usecols = featurelist)
testdata = testpart[featurelist].fillna(0).values



trainerror = []

def geterror(clf, test, truth):
    precluster = clf.predict(test)
    return (sum(precluster!=truth))/len(test)




featurelist = ['user_location_city','orig_destination_distance','srch_destination_id','hotel_market']
whlist = ['user_location_city','orig_destination_distance','srch_destination_id','hotel_market','hotel_cluster']

trainpart = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = whlist)
traindata = trainpart[featurelist].fillna(0).values
trainpara = {'data':traindata,'feature_names':featurelist,'target':trainpart['hotel_cluster'].values,
'target_names':np.arange(100)}

testpart = pd.read_csv('subset_datatest.csv',na_values=['--  '],usecols = whlist)
testdata = testpart[featurelist].fillna(0).values
testerror = []
from sklearn import tree
DTclf = tree.DecisionTreeClassifier()
EDTclf = tree.ExtraTreeClassifier()
DTclf = DTclf.fit(trainpara['data'], trainpara['target'],)
EDTclf = EDTclf.fit(trainpara['data'], trainpara['target'],)


precluster = DTclf.predict(traindata)
err  = (sum(precluster!=trainpart['hotel_cluster'].values))/len(traindata)
trainerror.append(err)
print("DTclf train error: {}".format(err))
precluster = DTclf.predict(testdata)
err  = (sum(precluster!=testpart['hotel_cluster'].values))/len(testdata)
testerror.append(err)
print("DTclf test error: {}".format(err))


precluster = EDTclf.predict(traindata)
err  = (sum(precluster!=trainpart['hotel_cluster'].values))/len(traindata)
trainerror.append(err)
print("EDTclf train error: {}".format(err))
precluster = EDTclf.predict(testdata)
err  = (sum(precluster!=testpart['hotel_cluster'].values))/len(testdata)
testerror.append(err)
print("EDTclf test error: {}".format(err))
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(n_estimators=30,
    max_depth=20, random_state=0).fit(traindata, trainpara['target'])
print('RFclf OK!')
testerror = []

testerror = []
mn = divmod(len(traindata),40000)
m = mn[0]
n = mn[1]
err1 = 0
err2 = 0
for i in range(m+1):
    if i<m:
        a = RFclf.predict(testdata[(i*40000):(i+1)*40000,:])
        b = RFclf.predict(traindata[(i*40000):(i+1)*40000,:])
        err1 += (sum(a!=testpart['hotel_cluster'].values[(i*40000):(i+1)*40000]))/len(testdata)
        err2 += (sum(b!=trainpart['hotel_cluster'].values[(i*40000):(i+1)*40000]))/len(traindata)
    else:
        a = RFclf.predict(testdata[(i*40000):len(testdata),:])
        b = RFclf.predict(traindata[(i*40000):len(traindata),:])
        err1 += (sum(a!=testpart['hotel_cluster'].values[(i*40000):len(testdata)]))/len(testdata)
        err2 += (sum(b!=trainpart['hotel_cluster'].values[(i*40000):len(traindata)]))/len(traindata)

trainerror.append(err1)
print("RFclf test error: {}".format(err1))

testerror.append(err2)
print("RFclf test error: {}".format(err2))
mn = divmod(len(traindata),40000)
m = mn[0]
n = mn[1]
testpre = np.array([])
trainpre = np.array([])
for i in range(m+1):
    if i<m:
        testpre = np.hstack( (testpre,RFclf.predict(testdata[(i*40000):(i+1)*40000,:])) )
        trainpre = np.hstack( (trainpre,RFclf.predict(traindata[(i*40000):(i+1)*40000,:])) )
    else:
        testpre = np.hstack( (testpre,RFclf.predict(testdata[(i*40000):len(testdata),:])) )
        trainpre = np.hstack( (trainpre,RFclf.predict(traindata[(i*40000):len(traindata),:])) )

testerr = []
clusnuma= []
trainerr = []
clusnumb = []
for i in range(100):
    if i%10 == 0:
        print("cluster:{}".format(i))
    indexa = testpart['hotel_cluster'].values == i
    tmpre = testpre[indexa]
    clusnuma.append(len(tmpre))
    testerr.append(sum(tmpre!=i)/ len(tmpre)) 
    indexb = trainpart['hotel_cluster'].values == i
    tmpre = trainpre[indexb]
    clusnumb.append(len(tmpre))
    trainerr.append(sum(tmpre!=i)/ len(tmpre)) 
testerr = Series(testerr)
clusnuma= Series(clusnuma)
trainerr = Series(trainerr)
clusnumb = Series(clusnumb)
fig=plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,1,1)
ax.plot(testerr,'bo--')
ax.plot(trainerr,'ro--')
fig=plt.figure(figsize=(20,10))
ax = fig.add_subplot(1,1,1)
ax.plot(testerr,'bo--')
ax.plot(trainerr,'ro--')

mn = divmod(len(testdata),40000)
m = mn[0]
n = mn[1]
eventid = 0
for i in range(m+1):
    clus = []
    if i<m:
        a = RFclf.predict(testdata[(i*40000):(i+1)*40000,:])
        b=np.argsort(a)[:,-5:]
        for ind in b:
            clus = []
            for ix in ind:
                clus.append(str(ix))
            out.write(str(eventid)+","+" ".join(clus)+"\n")
            eventid += 1
    else:
        a = RFclf.predict(testdata[(i*40000):len(testdata),:])
        b=np.argsort(a)[:,-5:]
        for ind in b:
            clus = []
            for ix in ind:
                clus.append(str(ix))
            out.write(str(eventid)+","+" ".join(clus)+"\n")
            eventid += 1

precluster = RFclf.predict(traindata)
err  = (sum(precluster!=trainpart['hotel_cluster'].values))/len(traindata)
trainerror.append(err)
print("RFclf test error: {}".format(err))

precluster = RFclf.predict(testdata)
err  = (sum(precluster!=testpart['hotel_cluster'].values))/len(testdata)
testerror.append(err)
print("RFclf test error: {}".format(err))
from sklearn.ensemble import GradientBoostingClassifier
def bicluster(i,j):
    tix = np.array(trainpart['hotel_cluster'].values==i)+np.array(trainpart['hotel_cluster'].values==j)
    tGBtraintarget = (trainpart['hotel_cluster'].values==i)*1
    tGBpara = {'data':GBdata,'feature_names':featurelist,'target':tGBtraintarget,
    'target_names':np.arange(100)}
    tmp = tGBpara['target'][tix]
    if sum(tmp==0)==0:
        tmp[-1] = 0
    tclf = GradientBoostingClassifier(n_estimators=20, learning_rate=1,
    max_depth=2, random_state=0).fit(GBdata[tix], tmp)
    return tclf

def oneclus(n):
    if n<99:
        return bicluster(n,n+1)
    else:
        return bicluster(n,0)
    
def getvoter():
    voterlist = []
    for i in range(100):
        accuracy = []
        clflist = []
        clf  = oneclus(i)
        clflist.append(clf)
        for j in range(100):
            tix = np.array(testpart1['hotel_cluster'].values==i)+np.array(testpart1['hotel_cluster'].values==j)
            accuracy.append( clf.score(testdata1[tix], 1*(testpart1['hotel_cluster'][tix].values==i)) )  
            #must use a testdata that contains true clusters
        accuracy = DataFrame([accuracy],index = ['accuracy']).T
        clusix = accuracy.sort_values( by ='accuracy',ascending = True).index[:2]
        tclf = clf
        for ind in clusix:
            tclf = bicluster(i,ind)    
            clflist.append(tclf)
        voterlist.append(clflist)
    return voterlist
def GBvote(testdata,voterlist):
    clusprob = []
    now = datetime.datetime.now()
    path = 'submission_GB_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    out.write("id,hotel_cluster\n")
    m = len(voterlist[0])
    for i in range(100):
#        print('1----'+str(i))
        clflist = voterlist[i]
        tmp = np.zeros([len(testdata),2])
        for j in range(m):    #compute the probability given by evevey clf
            clf = clflist[j]
            tmp = tmp + clf.predict_proba(testdata)
        tmp = tmp/m    
        tmp = (tmp[:,1]>0.5)*tmp[:,1]    #total probability for belonging to cluster i
        clusprob.append(tmp)
    clusprob = np.array(clusprob)
    for i in range(len(testdata)):
#        if i%20000 == 0:
#            print('2----'+str(i))
        clus = []
        a = clusprob[:,i]
        b=np.argsort(a)[-5:]
        #clusprob.drop(i,axis = 1)
        for ind in b:
            clus.append(str(ind))
        out.write(str(i)+","+"\t".join(clus)+"\n")