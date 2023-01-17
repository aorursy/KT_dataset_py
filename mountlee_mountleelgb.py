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
import random
import datetime
from sklearn.ensemble import GradientBoostingClassifier
#%matplotlib inline

        
# file = open("../input/train.csv")
# fout = open('subset_datatrain.csv','w')
# n = 0
# fout.write(file.readline())
# for line in file:
#     arr = line.strip().split(',')
#     is_book = int(arr[-6])
#     if is_book == 1:
#         fout.write(line)
# fout.close()
# file.close()



# dategroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
# chingroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep','2016Jan','2016May','2016Sep']
# dateix = [[] for i in range(12)]
# chinix = [[] for i in range(12)]

def datedeal(date):
    n = len(date)
    for i in range(n):
        a = date[i]
        if type(a) == type(0.1):
            a = '2015-01-01'
        if int(a[1])>0 or int(a[2])>1 or int(a[2])<1 or int(a[3])>6 or int(a[3])<3:   #大于2016或小于2013的年份全部换成2015
            date[i] = '2015-01-01'
    return pd.to_datetime(date)

def frameDateDeal(frame, datename):
    frame[datename]=frame[datename].fillna('2015-01-01')
    dateix = [[] for i in range(12)]
    datevalue = datedeal(frame[datename].values)
    datevalue = Series(np.arange(len(datevalue)),index = datevalue)
    for i in range(48):
        y = divmod(i,12)[0]
        r = divmod(i,12)[1]
        n = divmod(i,4)[0]
        if r<9:
            dateix[n].extend(datevalue['201'+str(3+y)+'-0'+str(r+1)].values)
        else:
            dateix[n].extend(datevalue['201'+str(3+y)+'-'+str(r+1)].values)
    for i in range(12):
        frame[datename].values[dateix[i]] = i
    return frame

# coding: utf-8

import datetime
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict

def run_solution():
    print('Preparing arrays...')
    f = open("../input/train.csv", "r")
    f.readline()
    best_hotels_od_ulc = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest = defaultdict(lambda: defaultdict(int))
    best_hotels_search_dest1 = defaultdict(lambda: defaultdict(int))
    best_hotel_country = defaultdict(lambda: defaultdict(int))
    popular_hotel_cluster = defaultdict(int)
    total = 0

    # Calc counts
    while 1:
        line = f.readline().strip()
        total += 1

        if total % 10000000 == 0:
            print('Read {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        book_year = int(arr[0][:4])
        user_location_city = arr[5]
        orig_destination_distance = arr[6]
        srch_destination_id = arr[16]
        is_booking = int(arr[18])
        hotel_country = arr[21]
        hotel_market = arr[22]
        hotel_cluster = arr[23]

        append_1 = 3 + 17*is_booking
        append_2 = 1 + 5*is_booking

        if user_location_city != '' and orig_destination_distance != '':
            best_hotels_od_ulc[(user_location_city, orig_destination_distance)][hotel_cluster] += 1

        if srch_destination_id != '' and hotel_country != '' and hotel_market != '' and book_year == 2014:
            best_hotels_search_dest[(srch_destination_id, hotel_country, hotel_market)][hotel_cluster] += append_1
        
        if srch_destination_id != '':
            best_hotels_search_dest1[srch_destination_id][hotel_cluster] += append_1
        
        if hotel_country != '':
            best_hotel_country[hotel_country][hotel_cluster] += append_2
        
        popular_hotel_cluster[hotel_cluster] += 1
    
    f.close()

    print('Generate submission...')
    now = datetime.datetime.now()
    #path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    path = 'predict_test.csv'
    out = open(path, "w")
    f = open("../input/test.csv", "r")
    f.readline()
    total = 0
    #out.write("id,hotel_cluster\n")
    topclasters = nlargest(10, sorted(popular_hotel_cluster.items()), key=itemgetter(1))

    while 1:
        line = f.readline().strip()
        total += 1

        if total % 1000000 == 0:
            print('Write {} lines...'.format(total))

        if line == '':
            break

        arr = line.split(",")
        id = arr[0]
        user_location_city = arr[6]
        orig_destination_distance = arr[7]
        srch_destination_id = arr[17]
        hotel_country = arr[20]
        hotel_market = arr[21]
        # id = arr[0]
        # user_location_city = arr[5]
        # orig_destination_distance = arr[6]
        # srch_destination_id = arr[16]
        # hotel_country = arr[21]
        # hotel_market = arr[22]


        out.write(str(id) + ',')
        filled = []

        out.write(str(id) + ',')
        filled = []

        s1 = (user_location_city, orig_destination_distance)
        if s1 in best_hotels_od_ulc:
            d = best_hotels_od_ulc[s1]
            topitems = nlargest(10, sorted(d.items()), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 10:
                    break
                out.write(',' + topitems[i][0])
                filled.append(topitems[i][0])

        s2 = (srch_destination_id, hotel_country, hotel_market)
        if s2 in best_hotels_search_dest:
            d = best_hotels_search_dest[s2]
            topitems = nlargest(10, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 10:
                    break
                out.write(',' + topitems[i][0])
                filled.append(topitems[i][0])
        elif srch_destination_id in best_hotels_search_dest1:
            d = best_hotels_search_dest1[srch_destination_id]
            topitems = nlargest(10, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 10:
                    break
                out.write(',' + topitems[i][0])
                filled.append(topitems[i][0])

        if hotel_country in best_hotel_country:
            d = best_hotel_country[hotel_country]
            topitems = nlargest(10, d.items(), key=itemgetter(1))
            for i in range(len(topitems)):
                if topitems[i][0] in filled:
                    continue
                if len(filled) == 10:
                    break
                out.write(',' + topitems[i][0])
                filled.append(topitems[i][0])

        for i in range(len(topclasters)):
            if topclasters[i][0] in filled:
                continue
            if len(filled) == 10:
                break
            out.write(',' + topclasters[i][0])
            filled.append(topclasters[i][0])

        out.write("\n")
    out.close()
    print('Completed!')


def bicluster(i,j):
    tix = np.array(trainpart['hotel_cluster'].values==i)+np.array(trainpart['hotel_cluster'].values==j)
    tGBtraintarget = (trainpart['hotel_cluster'].values==i)*1
    tGBpara = {'data':GBdata,'feature_names':featurelist,'target':tGBtraintarget,
    'target_names':np.arange(100)}
    tmp = tGBpara['target'][tix]
    if sum(tmp==0)==0:
        tmp[-1] = 0
    tclf = GradientBoostingClassifier(n_estimators=20, learning_rate=1,
    max_depth=4, random_state=0).fit(GBdata[tix], tmp)
    return tclf

def oneclus(n):
    a = list(range(100))
    a.remove(n)
    return bicluster(n,random.sample( a ,1)[0])
    
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
 


file = open("../input/train.csv")
fout = open('subset_datatest.csv','w')
n = 0
for line in file:
    if n == 0:
        fout.write(line)
    if n <200000*5:
        n +=1
    elif 200000*5<=n <200000*10:
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
    if n <200000*5:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()



featurelist = ['user_id','user_location_city','srch_destination_id','hotel_market','srch_ci']
whlist = ['user_id','user_location_city','srch_destination_id','hotel_market','srch_ci','hotel_cluster']
trainpart = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = whlist)

#to be used in function getvoter()
trainpart = frameDateDeal(trainpart,'srch_ci')
LGBdata = trainpart[featurelist].values
LGBpara = {'data':LGBdata,'feature_names':featurelist,'target':trainpart['hotel_cluster'].values,
'target_names':np.arange(100)}

#to be used in final selection
testpart = pd.read_csv('../input/test.csv',na_values=['--  '],usecols = featurelist)
testpart = frameDateDeal(testpart,'srch_ci')
testdata = testpart[featurelist].values

#to be used in funcion getvoter()
testpart1 = pd.read_csv('subset_datatest.csv',na_values=['--  '],usecols = whlist)
testpart1 = frameDateDeal(testpart1,'srch_ci')
testdata1 = testpart1[featurelist].values

os.remove('subset_datatest.csv')

run_solution()




def oneclus(n):
    a = list(range(100))
    a.remove(n)
    return bicluster(n,random.sample( a ,1)[0])

def bicluster(i,j):
    tix = np.array(trainpart['hotel_cluster'].values==i)+np.array(trainpart['hotel_cluster'].values==j)
    tLGBtraintarget = (trainpart['hotel_cluster'].values==i)*1
    tLGBpara = {'data':LGBdata,'feature_names':featurelist,'target':tLGBtraintarget,
    'target_names':np.arange(100)}
    tmp = tLGBpara['target'][tix]
    if sum(tmp==0)==0:
        tmp[-1] = 0
    tclf = GradientBoostingClassifier(n_estimators=20, learning_rate=1,
    max_depth=4, random_state=0).fit(LGBdata[tix], tmp)
    return tclf

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

voterlist = getvoter()
now = datetime.datetime.now()
path = 'submission_LGB_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")   

file = open('predict_test.csv')
for idd in range(len(testdata)):
    if idd%200000 == 0:
        print('Write {} lines...'.format(idd))
    line = file.readline().strip().split(',')
    cluslist = line[-10:]
    clusprob = []
    m = len(voterlist[0])
    for i in cluslist:   # the label of the cluster suggested by the leaskage solution
        i = int(i)
        clflist = voterlist[i]
        for j in range(m):    #compute the probability given by evevey clf
            clf = clflist[j]
            tmp = tmp + clf.predict_proba(testdata[idd,:].reshape(1,-1))
        tmp = tmp/m    
        tmp = (tmp[:,1]>0.5)*tmp[:,1]    #total probability for belonging to cluster i
        clusprob.append(tmp)
    clusprob = np.array(clusprob)
    b=np.argsort(clusprob)[-5:]
    for ind in b:
        clus.append(str(ind))
    out.write(str(idd)+","+" ".join(clus)+"\n")
    file.close()
    out.close()
1


