# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
inputPath = '/kaggle/input/tj19data/'

outputPath = ''



txn_data = pd.read_csv(inputPath+"txn.csv")
def keywordFromTxn(txn_data):

    startTime = time.time()

    keyword_list = []

    kw_les3 = []

    print('len txn',len(txn_data))

    length = len(txn_data)

    for i in range(length):

        t00 = txn_data.loc[i,'t0']

        if i%100000 == 1:

            lapse = time.time() - startTime

            pc= i/length*1.

            print (i,'lapse %.2f' % (lapse/60),'mins, (%.2f' % (pc*100.),'%%) need more %.2f' % (lapse*(1.-pc)/pc),'sec, or %.2f' % (lapse/60.*(1.-pc)/pc),'mins')

        try:

            float(t00)

        except :

            if len(t00)>3:

                keyword_list.append(t00)

            else:

                kw_les3.append(t00)

            #print('not number',extd)

    keyword_list.sort()

    return keyword_list,kw_les3

keyword_list,kw_les3 = keywordFromTxn(txn_data);

print('len keyword_list',len(keyword_list))
#keyword_list

kw_les3
def extractKw(kw_list):

    keyword_list2 = []

    error_len_list =[]

    oldD = ''

    for j in range(len(kw_list)):

        keyword =text = kw_list[j]

        for i in range(len(text)):

            if not((text[i] >= 'A' and text[i] <= 'Z')or text[i] == ' '):

                keyword = text[:i]

                break

        

        if len(keyword)<3:

            #print('error length',text,j)

            rspace = text.rfind(' ')

            if rspace > 3:

                keyword = text[:rspace]

            else:

                #error_len_list.append(text)

                if oldD != text[:4]:

                    #print('difficult',text[:4],'full',text)

                    oldD = text[:4]

                keyword = text[:4]

        else:

            rspace = keyword.rfind(' ')

            #print(rspace)

            if rspace>len(keyword)/2:

                keyword = keyword[:rspace]

            

            

        keyword_list2.append(keyword.lower())

    return keyword_list2,error_len_list

    

keyword_extracted,error_len_list = extractKw(keyword_list)

print('done extractKw',len(keyword_extracted))
#not used

missing = ['adidas', 'Aldo', 'Aliexpress', 'airpay','Agoda','Amazon']

missing = []

for line in error_len_list:

    isFound = False

    for line2 in missing:

        if line.lower().find(line2.lower())!=-1:

            isFound = True

            break;

    if not isFound:

        rf =line.rfind(' ')

        if rf>3:

            print (line[:rf])
def cleanKeyword(keyword_list):

    if len(keyword_list)==1:

        return keyword_list

    

    keyword_list.sort();

    for i in range(1,len(keyword_list)):

        if len(keyword_list[i])>0 and keyword_list[i].find(keyword_list[i-1])==0:

            keyword_list[i] = keyword_list[i-1].strip()

    

    return keyword_list





keyword_extracted_clean = cleanKeyword(keyword_extracted)

print('done cleanKeyword',len(keyword_extracted_clean))
def keywordHistogram(keyword_list):

    list_hist = {}

    for i in range(1,len(keyword_list)):

        if keyword_list[i] in list_hist:

            list_hist[keyword_list[i]]+=1

        else:

            list_hist[keyword_list[i]] = 1;

    

    list_hist2 = {}

    for i in list_hist.keys():

        if list_hist[i]!=1:

            list_hist2[i]=list_hist[i]

        

    return list_hist2

keyword_extracted_hist = keywordHistogram(keyword_extracted_clean)

#keyword_extracted_hist_sort = sorted(keyword_extracted_hist.items(), key = lambda kv: -kv[1])

print('done histogram',len(keyword_extracted_hist))
keyword_extracted_hist_sort = sorted(keyword_extracted_hist.items(), key = lambda kv: -kv[1])

keyword_extracted_hist_sort[0][0]
txn_data.id.unique()
def digestTxnToUser(txn_data,kw_hist,n_total=100,txn_count=20000000):

    #get hist

    kw_hist_filtered = {}

    if len(kw_hist)>n_total:

        keyword_extracted_hist_sort = sorted(kw_hist.items(), key = lambda kv: -kv[1])

        for i in range(n_total):

            kw_hist_filtered[keyword_extracted_hist_sort[i][0]]=keyword_extracted_hist_sort[i][1];

    else:

        n_total = len(kw_hist)

        kw_hist_filtered = kw_hist

        

    #print (len(kw_hist_filtered),kw_hist_filtered)

    user_feature = {}

    startTime = time.time()

    length = min(len(txn_data),txn_count)

    

    for i in range(length):

        if i%10000 == 1:

            lapse = time.time() - startTime

            pc= i/length*1.

            print (i,'lapse %.2f' % (lapse/60),'mins, (%.2f' % (pc*100.),'%%) need more %.2f' % (lapse*(1.-pc)/pc),'sec, or %.2f' % (lapse/60.*(1.-pc)/pc),'mins')

  

        keys = list(kw_hist_filtered.keys())

        for j in range(n_total):

            #print(kw_hist_filtered[j])

            if txn_data.loc[i,'t0'].lower().find(keys[j])!=-1:

               #found

                if txn_data.loc[i,'id'] not in user_feature:

                    user_feature[txn_data.loc[i,'id']] = [0]*n_total

                user_feature[txn_data.loc[i,'id']][j] +=1

                

    

    return user_feature

               

feature = digestTxnToUser(txn_data,keyword_extracted_hist,4,1000000)

sweep = {}

for j in []:

    feature = digestTxnToUser(txn_data,keyword_extracted_hist,j,1000000)

    #print(feature)

    result = [np.sum(feature[i]) for i in feature]

    print (j,'zeros',len(result)-np.count_nonzero(result),'zero %',100*(len(result)-np.count_nonzero(result))/len(result))

    sweep[j]= 100*(len(result)-np.count_nonzero(result))/len(result)

    
feature_sort = sorted(feature.items(), key = lambda kv: kv[0])



feature_list = []

j=0

for i in feature_sort:

    #print(i)

    feature_list.append([i[0]])

    feature_list[j].extend(i[1])

    j+=1
feature_sort
df = pd.DataFrame(feature_list)

df.to_csv(path_or_buf="10f_1mil.csv",index=False)