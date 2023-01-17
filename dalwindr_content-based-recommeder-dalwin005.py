# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/RecommSystrain.txt",sep=",",skiprows=232,chunksize=200000)
#print(train.shape)
for df in train:
    df2=df.copy()
    break
df2.columns=['id', 'publish_date', 'headline_category', 'headline_text']
df2.head(5)
### Following are the functions created

##### def getTopArticles(article):
##### def getSimilarityBetweenTwoArticles(article1, article2):
##### def getCosineSimilarity(bow1, bow2):
##### def getBow(text):
##### def getNorm(vec):
'''
This module provides functions to find matching articles to a given article
'''

import csv
from pprint import pprint
import time
import json
#from textMatching import getCosineSimilarity, getBow
#from getSimilarityBetweenTexts import TextSimilarity
# Create the text similarity object
#ts = TextSimilarity()


    
from collections import defaultdict
import math

def getBow(text):
    text = text.lower()
    
    bowDict = defaultdict(int) # word->freq mapping dict
    wordList = text.split(' ')
    
    # print 'wordList', wordList
    
    for word in wordList:
        bowDict[word] += 1
        
    result = dict(bowDict)
    return result

def getNorm(vec):
    sum_sq = 0
    
    for x in vec:
        sum_sq += x*x
        
    norm = math.sqrt(sum_sq)
    
    return norm

def getCosineSimilarity(bow1, bow2):
    norm1 = getNorm(bow1.values())
    norm2 = getNorm(bow2.values())
    
    dot_product = 0
    
    for keyword1, freq1 in bow1.items():
        if keyword1 in bow2:
            freq2 = bow2[keyword1]
            
            dot_product += freq1*freq2
            
    cosine = float(dot_product)/(1 + float(norm1*norm2))
    
    return cosine

def getTopMatchingArticles(article,simScoreThreshhold=0.25,No_ofTopScoreArticle=5 ):
    all_similarArticles = []
    for articleid2, article2 in articleIdToArticleMapping.items():
        # s = getSimilarityBetweenTwoArticleIds(articleid1, articleid2, articleIdToArticleMapping)
        
        if articleid == articleid2:
            continue

        simScore = getSimilarityBetweenTwoArticles(article, article2)
        
        if simScore > simScoreThreshhold: #min cosine
            row = [articleid, articleid2, simScore]
        
            # print row
            all_similarArticles.append(row)
        
    
    
    #Get topN
    topN = No_ofTopScoreArticle
    sorted_all_SimilarArticles = sorted(all_similarArticles, key = lambda x: x[2], reverse = True)
    topN_SimilarArticles = sorted_all_SimilarArticles[0:topN]
    # print 'topNExcelSheet:', topNExcelSheet
    
    Ids_of_topN_Articles = []
    for (article_orig, article_id2, similarity) in topN_SimilarArticles:
        Ids_of_topN_Articles.append(article_id2)

    # pprint(article2bsent)
    
    return Ids_of_topN_Articles

def getSimilarityBetweenTwoArticles(article1, article2):
    bow1 = getBow(article1)
    bow2 = getBow(article2)
    #text1 = article1['headline_text']
    #text2 = article2['headline_text']
    
    cosineSimilarity_Score = getCosineSimilarity(bow1,bow2)
    return cosineSimilarity_Score

if __name__ == '__main__':
    print ('Text matching module')
    
    text1 = 'I am data scientist'
    text2 = 'very different sales and scientist'
    
    print (text1)
    print (text2)
    
    '''
    {'I':1,...}
    '''
    bow1 = getBow(text1)
    print ('bow1', bow1)

    bow2 = getBow(text2)
    print ('bow2', bow2  )  
    
    '''
    Vocabulary
    
    i | am | data| scientist| is| a |great| job
    
    text1 = 'I am data scientist'
    text2 = 'Data scientist is a great job'
    
    text1 = [1,1,1,1,0,0,0,0]
    text2 = [0,0,1,1,1,1,1,1] - 8D
    
    cos = a.b/norm(a)*norm(b)
    '''
    
    # Find the cosine similarity between bow1 and bow2
    # s = getCosineSimilarity(bow1, bow2)
    
    print (getCosineSimilarity(bow1, bow2))

if __name__ == '__main__':
    print ('Compute Article Similarity and Save Model')
    
    # Load the dataset
        
#     f = open('data.csv', 'rb')
#     fieldnames = ['id', 'publish_date', 'headline_category', 'headline_text']
#     reader = csv.DictReader(f, fieldnames=fieldnames)
    
    
        
    
    i = 0
    articleIdToArticleMapping = {} #id->article
    st_time = time.time()
    articleIdToArticleMapping = dict(df2[['id','headline_text']].head(1000).values)
    
    end_time = time.time()
    print("time taken",end_time-st_time, "seconds")

    #pprint({'articlesToBeSentMapping': articlesToBeSentMapping})


    
    articleToArticle_matched = {} #article-> which article to send
    for articleid, article in articleIdToArticleMapping.items(): 
        topN_Matching_article_ids = getTopMatchingArticles(article)
        articleToArticle_matched[articleid] = topN_Matching_article_ids
        print("-----\nFor given article id =",articleid,"total No of matching article found:",len(articleToArticle_matched[articleid]))
        if len(articleToArticle_matched[articleid]) > 0:
            print ('  Original Article:', "(id",articleid,")", articleIdToArticleMapping[articleid])

        print ('\t Matching Articles:')                  
        for Matched_articleid in articleToArticle_matched[articleid]:
                print ('\t \t', "(Matched id",Matched_articleid,")",  articleIdToArticleMapping[Matched_articleid])

    

    
    # Save the model to a json
    g = open('articlesToBeSentMapping.json', 'wb')
    json.dump(articleToArticle_matched, g)
    g.close()


    # Save the article id to article mapping to a json
    g = open('articleIdToArticleMapping.json', 'wb')
    json.dump(articleIdToArticleMapping, g)
    g.close()    
from numba import cuda
cuda.select_device(0)
cuda.close()
