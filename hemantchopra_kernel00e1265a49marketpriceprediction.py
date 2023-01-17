# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from kaggle.competitions import twosigmanews
#import os
from sklearn.linear_model import LogisticRegression
#from sklearn import datasets
import nltk
import re
from stop_words import get_stop_words
from datetime import datetime
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#print(os.listdir("../input"))
import datetime,time
env = twosigmanews.make_env()
%matplotlib inline
# Any results you write to the current directory are saved as output.
(market_train_df, news_train_df) = env.get_training_data()
news_train_df.head()
market_train_df.head()
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))
def headlines(value,asset):
    #print('processing headlines for ', asset)
    #get company name
    assetsplit=str(asset).lower().split()
    #print(assetsplit[0])
    if len(assetsplit[0])==1 & len(assetsplit) > 2:
        companyname=assetsplit[0]+assetsplit[1]
    else:
        companyname=assetsplit[0]
    print(companyname)
    nameposition=[]
    print(len(value))
    for i in range(len(value)):
        score=0
        sentence=value[i]
        #print(sentence)
        tokens = nltk.word_tokenize(sentence)
        #print(tokens,'\n')
        #print(tokens)
        #check polarity score for news (postive, negtive, neutral)
        for word in tokens:
            if str(word).lower()==companyname:
                nameposition.append(i)
                break
    print(len(nameposition))
def text_process(title,text_values):
    #print('Processing Text for counts for ',title)
    regex = r'\w+'
    text=[]
    text_count=[]
    for i in range(len(text_values)):
        list1=re.findall(regex, text_values[i])
        #count of related topics
        text.append(list1)
        text_count.append(len(list1))    
    return text_count
def replace_txt_w_num(values):
    values_temp=[]
    
    for (i, item) in enumerate(values):
        if item == True:
            values_temp.append(1.0)
        else:
            values_temp.append(0.0)
    return values_temp
def sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,values,max_time_buckets, title):
    #print('Syncing date to values for ',title)
    
    upd_max_time_buckets= [datetime.datetime.utcfromtimestamp(i/(1e+9)).date() for i in max_time_buckets]
    #print(upd_max_time_buckets)
    start_row=row_insert
    indexes_v=[]
    checks=0
    #print(time_values)
    for i in time_values:
        if i>upd_max_time_buckets[len(upd_max_time_buckets)-1]:
            break
        if i in upd_max_time_buckets:
            asset_matrix[start_row,upd_max_time_buckets.index(i)+checks]=values[time_values.index(i)]
            indexes_v.append(upd_max_time_buckets.index(i)+checks)
        else:
            x=1
            word='damn'
            
            while word=='damn':
                date=str(i)
                
                date2=date[0:4]+date[5:7]+date[8:10]
                
                if date2 in set(added_dates):
                    asset_matrix[start_row,np.where(asset_matrix == int(date2))[1]]=values[time_values.index(i)]
                    #print('Done new', date2, '   ',values[upd_time_values.index(i)] )
                    word='yay'
                if date2 not in set(added_dates):
                    #print(date2)
                    if i+datetime.timedelta(days=x) in upd_max_time_buckets:       
                        asset_matrix=np.insert(asset_matrix,upd_max_time_buckets.index(i+datetime.timedelta(days=x))+checks,int(date2),axis=1)
                        asset_matrix[1:,upd_max_time_buckets.index(i+datetime.timedelta(days=x))+checks]=0.0
                        asset_matrix[start_row,upd_max_time_buckets.index(i+datetime.timedelta(days=x))+checks]=values[time_values.index(i)]
                        indexes_v.append(upd_max_time_buckets.index(i+datetime.timedelta(days=x))+checks)
                        added_dates.append(date2)
                        checks+=1
                        word='yay'
                    x+=1  
                                 
    #print('Nett new Date Columns added = ',checks)
    
    return asset_matrix, added_dates
def category_replace(list_of_values, title):
    #print ('Replacing with category count for ',title)
    new_list=[]
    try:
        x = list(set(list_of_values.categories))
    except:
        x = list(set(list_of_values))
    dic = dict(zip(x, list(range(1,len(x)+1))))
    for y in list_of_values:
        new_list.append(dic[y])
    return new_list
        
def normalize(input_array):
    input_array[np.isnan(input_array)]=0
    divisor=np.amax(input_array, axis=0)
    #print(divisor)
    divisor[divisor==0]=1
    #print(divisor)
    input_array2=input_array/divisor
    return input_array2
def find_traded_assets(market_train_df,news_train_df):    
    traded_assets=market_train_df[market_train_df.universe==float(1.0)]['assetName'].unique().astype(str)
    news_asset=news_train_df['assetName'].unique().astype(str)
    trade_w_news=intersection(traded_assets, news_asset)
    
    return trade_w_news
def values_date_avg(news_sorted,field):
    j=0
    dict_values={}
    for i in range(len(news_sorted)-1):
        if news_sorted['time'][i].date() != news_sorted['time'][i+1].date():
            mean=news_sorted[field][j:i].mean()
            j=i
            dict_values[news_sorted['time'][i].date()]=mean
    #print(dict_values)
    return dict_values
def values_date_avg_mod(news_sorted,list_of_values,field):
    j=0
    dict_values={}
    for i in range(len(news_sorted)-1):
        if news_sorted['time'][i].date() != news_sorted['time'][i+1].date():
            try:
                mean=sum(list_of_values[j:i])/(i-j)
            except:
                mean=sum(list_of_values[j:i])/1
            j=i
            dict_values[news_sorted['time'][i].date()]=mean
    #print(dict_values)
    return dict_values
def build_matrix(market_train_df, news_train_df,traded_asset):
    
    asset=traded_asset
    print(asset)
    market_sorted = market_train_df[(market_train_df.assetName == str(asset))]['time'].values.tolist()
    print('Asset time bucket count',len(market_sorted))
    asset_matrix=np.zeros(shape=(41,len(market_sorted)))
    max_time_buckets_upd=[datetime.datetime.utcfromtimestamp(i/(1e+9)).date() for i in market_sorted]
    date_list=[]
    for i in max_time_buckets_upd:
        date=str(i)
        date2=date[0:4]+date[5:7]+date[8:10]
        date_list.append(int(date2))
    asset_matrix[0,:]=date_list
    #print(asset_matrix)
    column_list=market_train_df.columns.values
    #print(asset_matrix.shape)

    #Build feature matrix for a random asset (except asset name and asset code in the matrix) for market information
    
    for j in range(3, 16):
        #print("Processing Column ",column_list[j])
        #quick way of creating max_length list
        column_values=market_train_df[(market_train_df.assetName == str(asset))][column_list[j]].values.tolist()
    
        if len(column_values) < len(market_sorted):
            #create_empty list
            empty_column_list=[0.0]*len(max_time_buckets)
            #print('Check1 - less than max time buckets')
            #update values from actual into empty list
            i=0
            while i<len(positions):
                #print('Replacing at position ',positions[i], 'with value ', column_values[i])
                empty_column_list[int(positions[i])]=column_values[i]
                i+=1
        else:
            empty_column_list=column_values
        
        asset_matrix[(j-2),:]=empty_column_list
        #print('Check 2 - Deleting list')
        del column_values[:],empty_column_list[:]
    asset_matrix[np.isnan(asset_matrix)] = 0
    #print(asset_matrix)
    ## Add news to the matrix
    #Asset data till row 13
    
    row_insert=14
    added_dates=[]
    news_sorted = news_train_df[(news_train_df.assetName == str(asset))].sort_values(by='time')
    print('Total rows count ',len(news_sorted))
    #time_values=news_train_df[(news_train_df.assetName == str(asset))]['time'].values.tolist()
    #4 headline - name appears in headline, positive, negative, factual, earnings
    '''headline_value=news_train_df[(news_train_df.assetName == str(asset))]['headline'].values.tolist()
    '''
    
    #Name occurence, positive, negative, neutral, earnings report

    #5 urgency - as is
    urgency_values=values_date_avg(news_sorted,'urgency')
    time_values=[*urgency_values.keys()]
    urgency_values=[*urgency_values.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,urgency_values,market_sorted,'urgency')
    row_insert+=1

    '''#6 takesequence - as - is ignore
    takeSequence_values=news_train_df[(news_train_df.assetName == str(asset))]['takeSequence'].values
    '''
    #7 provider
    provider_values=news_train_df[(news_train_df.assetName == str(asset))]['provider'].values
    provider_values_replace=category_replace(provider_values,'provider')  
    provider_values=values_date_avg_mod(news_sorted,provider_values_replace,'provider')
    provider_values=[*provider_values.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,provider_values,market_sorted,'provider')
    print('Reduced row Count', len(provider_values))
    row_insert+=1

    #8 subjects
    subject_values=news_train_df[(news_train_df.assetName == str(asset))]['subjects'].values
    subject_count=text_process('subjects',subject_values)
    subject_values=values_date_avg_mod(news_sorted,subject_count,'subjects')
    subject_values=[*subject_values.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,subject_values,market_sorted,'subjects')
    #print(len(subject_values))
    row_insert+=1

    #9 'audiences'
    audiences_value=news_train_df[(news_train_df.assetName == str(asset))]['audiences'].values
    audiences_count=text_process('audiences',audiences_value)
    audiences_value=values_date_avg_mod(news_sorted,audiences_count,'audiences')
    audiences_value=[*audiences_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,audiences_value,market_sorted,'audiences')
    #print(len(audiences_value))
    row_insert+=1


    #10 'bodySize' - create intervals and map
    bodySize_value=values_date_avg(news_sorted,'bodySize')
    bodySize_value=[*bodySize_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,bodySize_value,market_sorted,'bodySize')
    row_insert+=1

    #11 'companyCount' 
    companyCount_value=values_date_avg(news_sorted,'companyCount')
    companyCount_value=[*companyCount_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,companyCount_value,market_sorted,'companyCount')
    #print(len(companyCount_value))
    row_insert+=1

    #12 'headlineTag'- assign numbers
    headlineTag_value=news_train_df[(news_train_df.assetName == str(asset))]['headlineTag'].values
    headlineTag_value_replace=category_replace(provider_values,'headlineTag')  
    headlineTag_value=values_date_avg_mod(news_sorted,headlineTag_value_replace,'headlineTag')
    headlineTag_value=[*headlineTag_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,headlineTag_value,market_sorted,'headlineTag')
    #print(len(headlineTag_value))
    #print(len(provider_values))


    #13 'marketCommentary' -0/1
    marketCommentary_value=news_train_df[(news_train_df.assetName == str(asset))]['marketCommentary'].values
    marketCommentrary_repl_value=replace_txt_w_num(marketCommentary_value)
    marketCommentary_value=values_date_avg_mod(news_sorted,marketCommentrary_repl_value,'marketCommentary')
    marketCommentary_value=[*marketCommentary_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,marketCommentary_value,market_sorted,'marketCommentary')
    #print(len(marketCommentary_value))
    #print(len(provider_values))

    #14 'sentenceCount' - create ranges
    sentenceCount_value=values_date_avg(news_sorted,'sentenceCount')
    sentenceCount_value=[*sentenceCount_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentenceCount_value,market_sorted,'sentenceCount')
    #print(len(sentenceCount_value))
    row_insert+=1
    
    #15 'wordCount' - create ranges
    wordCount_value=values_date_avg(news_sorted,'wordCount')
    wordCount_value=[*wordCount_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,wordCount_value,market_sorted,'wordCount')
    #print(len(wordCount_value))
    row_insert+=1
    
    #16 'assetCodes' - blank
    #17 'assetName' - blank

    #18 'firstMentionSentence' -create ranges
    firstMentionSentence_value=values_date_avg(news_sorted,'firstMentionSentence')
    firstMentionSentence_value=[*firstMentionSentence_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,firstMentionSentence_value,market_sorted,'firstMentionSentence')
    #print(len(firstMentionSentence_value))
    row_insert+=1
    
    #19 'relevance'- score between 0 and 1
    relevance_value=values_date_avg(news_sorted,'relevance')
    relevance_value=[*relevance_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,relevance_value,market_sorted,'relevance')
    #print(len(relevance_value))
    row_insert+=1
    
    #20 'sentimentClass' - 1,-1,0
    sentimentClass_value=values_date_avg(news_sorted,'sentimentClass')
    sentimentClass_value=[*sentimentClass_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentimentClass_value,market_sorted,'sentimentClass')
    #print(len(sentimentClass_value))
    row_insert+=1
    
    #21 'sentimentNegative' -  score between 0 and 1
    sentimentNegative_value=values_date_avg(news_sorted,'sentimentNegative')
    sentimentNegative_value=[*sentimentNegative_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentimentNegative_value,market_sorted,'sentimentNegative')
    #print(len(sentimentNegative_value))
    row_insert+=1
    
    #22 'sentimentNeutral' - score between 0 and 1
    sentimentNeutral_value=values_date_avg(news_sorted,'sentimentNeutral')
    sentimentNeutral_value=[*sentimentNeutral_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentimentNeutral_value,market_sorted,'sentimentNeutral')
    #print(len(sentimentNeutral_value))
    row_insert+=1
    
    #23 'sentimentPositive' - score between 0 and 1
    sentimentPositive_value=values_date_avg(news_sorted,'sentimentPositive')
    sentimentPositive_value=[*sentimentPositive_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentimentPositive_value,market_sorted,'sentimentPositive')
    #print(len(sentimentPositive_value))
    row_insert+=1
    
    #24 'sentimentWordCount' - create ranges
    sentimentWordCount_value=values_date_avg(news_sorted,'sentimentWordCount')
    sentimentWordCount_value=[*sentimentWordCount_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,sentimentWordCount_value,market_sorted,'sentimentWordCount')
    #print(len(sentimentWordCount_value))
    row_insert+=1


    #25 'noveltyCount12H' - 0 to 6
    noveltyCount12H_value=values_date_avg(news_sorted,'noveltyCount12H')
    noveltyCount12H_value=[*noveltyCount12H_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,noveltyCount12H_value,market_sorted,'noveltyCount12H')
    #print(len(noveltyCount12H_value))
    row_insert+=1
    
    #26 'noveltyCount24H'- 0 to 6
    noveltyCount24H_value=values_date_avg(news_sorted,'noveltyCount24H')
    noveltyCount24H_value=[*noveltyCount24H_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,noveltyCount24H_value,market_sorted,'noveltyCount24H')
    #print(len(noveltyCount24H_value))
    row_insert+=1
    
    #27 'noveltyCount3D' - 0 to 6
    noveltyCount3D_value=values_date_avg(news_sorted,'noveltyCount3D')
    noveltyCount3D_value=[*noveltyCount3D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,noveltyCount3D_value,market_sorted,'noveltyCount3D')
    #print(len(noveltyCount3D_value))
    row_insert+=1
    
    #28 'noveltyCount5D' - 0 to 6
    noveltyCount5D_value=values_date_avg(news_sorted,'noveltyCount5D')
    noveltyCount5D_value=[*noveltyCount5D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,noveltyCount5D_value,market_sorted,'noveltyCount5D')
    #print(len(noveltyCount5D_value))
    row_insert+=1
    
    #29 'noveltyCount7D' - 0 to 6
    noveltyCount7D_value=values_date_avg(news_sorted,'noveltyCount7D')
    noveltyCount7D_value=[*noveltyCount7D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,noveltyCount7D_value,market_sorted,'noveltyCount7D')
    #print(len(noveltyCount7D_value))
    row_insert+=1
    
    #30 'volumeCounts12H' - range till 21
    volumeCounts12H_value=values_date_avg(news_sorted,'volumeCounts12H')
    volumeCounts12H_value=[*volumeCounts12H_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,volumeCounts12H_value,market_sorted,'volumeCounts12H')
    #print(len(volumeCounts12H_value))
    row_insert+=1
    
    #31 'volumeCounts24H' - range till 21
    volumeCounts24H_value=values_date_avg(news_sorted,'volumeCounts24H')
    volumeCounts24H_value=[*volumeCounts24H_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,volumeCounts24H_value,market_sorted,'volumeCounts24H')
    #print(len(volumeCounts24H_value))
    row_insert+=1
    
    #32 'volumeCounts3D' - range till 21
    volumeCounts3D_value=values_date_avg(news_sorted,'volumeCounts3D')
    volumeCounts3D_value=[*volumeCounts3D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,volumeCounts3D_value,market_sorted,'volumeCounts3D')
    #print(len(volumeCounts3D_value))
    row_insert+=1
    
    #33 'volumeCounts5D' - range till 27
    volumeCounts5D_value=values_date_avg(news_sorted,'volumeCounts5D')
    volumeCounts5D_value=[*volumeCounts5D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,volumeCounts5D_value,market_sorted,'volumeCounts5D')
    #print(len(volumeCounts5D_value))
    row_insert+=1
    
    #34 'volumeCounts7D' - range till 37
    volumeCounts7D_value=values_date_avg(news_sorted,'volumeCounts7D')
    volumeCounts7D_value=[*volumeCounts7D_value.values()]
    asset_matrix,added_dates=sync_to_date_range(added_dates,row_insert,asset_matrix,time_values,volumeCounts7D_value,market_sorted,'volumeCounts7D')
    #print(len(volumeCounts7D_value))
    row_insert+=1
    
    return asset_matrix
   
def prep_data(asset_matrix):
    cause2=np.delete(asset_matrix,0,0)
    cause=np.delete(cause2,[1,2,3,4,5,6,7,8,9,10],0)
    
    effect=asset_matrix[12,:]

    for i in range(asset_matrix.shape[1]):
        if cause[8,i]==0 and cause[2,i]==0:
            np.delete(cause,i,1)
            np.delete(effect,i,0)
    cause=cause.transpose()
    cause=normalize(cause)
    effect=effect.transpose()
    effect=np.delete(effect,0)
    effectout=np.append(effect,[0,])
    effectout[effectout>0]=1
    effectout[effectout<0]=-1
    #consider a column only if news exists and universe = 1
    #print('Classes ',np.unique(effectout))
    X=cause
    y=effectout
    
    return X,y
def train_my_model(X,y):
    clf = LogisticRegression(random_state=0, solver='liblinear',multi_class='ovr',verbose=1).fit(X, y)
    return clf
def append_matrix(asset,asset_matrix,market_obs_df,news_obs_df):
    #print(asset_matrix.shape)
    asset_matrix=np.insert(asset_matrix, asset_matrix.shape[1], 0, axis=1)
    #print(asset_matrix.shape)
    volume=market_obs_df[(market_obs_df.assetName == str(asset))]['volume'].values.tolist()
    #print(volume)
    asset_matrix[1,asset_matrix.shape[1]-1]= volume[0]
    
    returnsClosePrevRaw1=market_obs_df[(market_obs_df.assetName == str(asset))]['returnsClosePrevRaw1'].values.tolist()
    asset_matrix[11,asset_matrix.shape[1]-1]= returnsClosePrevRaw1[0]
    
    time=market_obs_df[(market_obs_df.assetName == str(asset))]['time'].values.tolist()
    time_upd=[datetime.datetime.utcfromtimestamp(i/(1e+9)).date() for i in time]
    if time_upd[0].weekday()<5:
        asset_matrix[13,asset_matrix.shape[1]-1]= 1
    else:
        asset_matrix[13,asset_matrix.shape[1]-1]= 0
    
    urgency=news_obs_df[(news_obs_df.assetName == str(asset))]['urgency']
    asset_matrix[14,asset_matrix.shape[1]-1]= urgency[:].mean()

    provider=news_obs_df[(news_obs_df.assetName == str(asset))]['provider']
    provider_values_replace=category_replace(provider,'provider')  
    asset_matrix[15,asset_matrix.shape[1]-1]= sum(provider_values_replace[:])/len(provider_values_replace)
    
    subjects=news_obs_df[(news_obs_df.assetName == str(asset))]['subjects']
    subject_count=text_process('subjects',subjects)
    asset_matrix[16,asset_matrix.shape[1]-1]= sum(subject_count[:])/len(subject_count)
    
    audiences=news_obs_df[(news_obs_df.assetName == str(asset))]['audiences']
    audiences_count=text_process('audiences',audiences)
    asset_matrix[17,asset_matrix.shape[1]-1]= sum(audiences_count[:])/len(audiences_count)
    
    bodySize=news_obs_df[(news_obs_df.assetName == str(asset))]['bodySize']
    asset_matrix[18,asset_matrix.shape[1]-1]= bodySize[:].mean()
    
    companyCount=news_obs_df[(news_obs_df.assetName == str(asset))]['companyCount']
    asset_matrix[19,asset_matrix.shape[1]-1]= companyCount[:].mean()
    
    headlineTag=news_obs_df[(news_obs_df.assetName == str(asset))]['headlineTag']
    headlineTag_value_replace=category_replace(headlineTag,'headlineTag')  
    asset_matrix[20,asset_matrix.shape[1]-1]= sum(headlineTag_value_replace[:])/len(headlineTag_value_replace)
    
    marketCommentary=news_obs_df[(news_obs_df.assetName == str(asset))]['marketCommentary']
    marketCommentrary_repl_value=replace_txt_w_num(marketCommentary)
    asset_matrix[21,asset_matrix.shape[1]-1]= sum(headlineTag_value_replace[:])/len(headlineTag_value_replace)
    
    sentenceCount=news_obs_df[(news_obs_df.assetName == str(asset))]['sentenceCount']
    asset_matrix[22,asset_matrix.shape[1]-1]= sentenceCount[:].mean()
    
    wordCount=news_obs_df[(news_obs_df.assetName == str(asset))]['wordCount']
    asset_matrix[23,asset_matrix.shape[1]-1]= wordCount[:].mean()
    
    firstMentionSentence=news_obs_df[(news_obs_df.assetName == str(asset))]['firstMentionSentence']
    asset_matrix[24,asset_matrix.shape[1]-1]= firstMentionSentence[:].mean()
    
    relevance=news_obs_df[(news_obs_df.assetName == str(asset))]['relevance']
    asset_matrix[25,asset_matrix.shape[1]-1]= relevance[:].mean()
    
    sentimentClass=news_obs_df[(news_obs_df.assetName == str(asset))]['sentimentClass']
    asset_matrix[26,asset_matrix.shape[1]-1]= sentimentClass[:].mean()
    
    sentimentNegative=news_obs_df[(news_obs_df.assetName == str(asset))]['sentimentNegative']
    asset_matrix[27,asset_matrix.shape[1]-1]= sentimentNegative[:].mean()
    
    sentimentNeutral=news_obs_df[(news_obs_df.assetName == str(asset))]['sentimentNeutral']
    asset_matrix[28,asset_matrix.shape[1]-1]= sentimentNeutral[:].mean()
    
    sentimentPositive=news_obs_df[(news_obs_df.assetName == str(asset))]['sentimentPositive']
    asset_matrix[29,asset_matrix.shape[1]-1]= sentimentPositive[:].mean()
    
    sentimentWordCount=news_obs_df[(news_obs_df.assetName == str(asset))]['sentimentWordCount']
    asset_matrix[30,asset_matrix.shape[1]-1]= sentimentWordCount[:].mean()
    
    noveltyCount12H=news_obs_df[(news_obs_df.assetName == str(asset))]['noveltyCount12H']
    asset_matrix[31,asset_matrix.shape[1]-1]= noveltyCount12H[:].mean()
    
    noveltyCount24H=news_obs_df[(news_obs_df.assetName == str(asset))]['noveltyCount24H']
    asset_matrix[32,asset_matrix.shape[1]-1]= noveltyCount24H[:].mean()
    
    noveltyCount3D=news_obs_df[(news_obs_df.assetName == str(asset))]['noveltyCount3D']
    asset_matrix[33,asset_matrix.shape[1]-1]= noveltyCount3D[:].mean()
    
    noveltyCount5D=news_obs_df[(news_obs_df.assetName == str(asset))]['noveltyCount5D']
    asset_matrix[34,asset_matrix.shape[1]-1]= noveltyCount5D[:].mean()
    
    noveltyCount7D=news_obs_df[(news_obs_df.assetName == str(asset))]['noveltyCount7D']
    asset_matrix[35,asset_matrix.shape[1]-1]= noveltyCount7D[:].mean()
    
    volumeCounts12H=news_obs_df[(news_obs_df.assetName == str(asset))]['volumeCounts12H']
    asset_matrix[36,asset_matrix.shape[1]-1]= volumeCounts12H[:].mean()
    
    volumeCounts24H=news_obs_df[(news_obs_df.assetName == str(asset))]['volumeCounts24H']
    asset_matrix[37,asset_matrix.shape[1]-1]= volumeCounts24H[:].mean()
    
    volumeCounts3D=news_obs_df[(news_obs_df.assetName == str(asset))]['volumeCounts3D']
    asset_matrix[38,asset_matrix.shape[1]-1]= volumeCounts3D[:].mean()
    
    volumeCounts5D=news_obs_df[(news_obs_df.assetName == str(asset))]['volumeCounts5D']
    asset_matrix[39,asset_matrix.shape[1]-1]= volumeCounts5D[:].mean()
    
    volumeCounts7D=news_obs_df[(news_obs_df.assetName == str(asset))]['volumeCounts7D']
    asset_matrix[40,asset_matrix.shape[1]-1]= volumeCounts7D[:].mean()
    
    X,y=prep_data(asset_matrix)
    #print(X[-1,:])
    return X
dict_assets={}
traded_w_news=find_traded_assets(market_train_df,news_train_df)
count=1
for i in traded_w_news:
    if i == 'Agilent Technologies Inc':
        asset_matrix=build_matrix(market_train_df, news_train_df,i)
        X,y=prep_data(asset_matrix)
        model=train_my_model(X,y)
        dict_assets[i]=[model,X,asset_matrix]
        print ('\nModel done for ',i, 'which is ',count,'/',len(traded_w_news),'\n')
        count+=1
days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    predictions_df=predictions_template_df.copy()
    for i in range(len(predictions_template_df)):
        asset=market_obs_df['assetName'][i]
        assetcode=market_obs_df['assetCode'][i]
        print(asset)
        if asset in traded_w_news:
            model=dict_assets[asset][0]
            matrix=dict_assets[asset][2]
            X_upd=append_matrix(asset,matrix,market_obs_df, news_obs_df)
            X1=X_upd[X_upd[:,0]>0]
            X2=X1[0,:]
            Xm=np.vstack([X2, X_upd[-1,:]])
            prediction1=model.predict_proba(Xm)
            prediction2=model.predict(Xm)
            prediction=np.amax(prediction1[1,:])*prediction2[1]
            predictions_df.confidenceValue[i] = prediction
        else:
            predictions_df.confidenceValue[i] = 0
    env.predict(predictions_df)
print('Done!')