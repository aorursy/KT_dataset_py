# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



bookDF=pd.read_csv('../input/books.csv')

bookDF=bookDF.drop(['image_url','small_image_url','title','best_book_id','isbn','isbn13'],axis=1)

ratingsDF = pd.read_csv('../input/ratings.csv')



# Any results you write to the current directory are saved as output.
bookDF.head(5)
ratingsDF.head(5)
listOfDictonaries=[]

indexMap = {}

reverseIndexMap = {}

ptr=0;

testdf = ratingsDF

testdf=testdf[['user_id','rating']].groupby(testdf['book_id'])

for groupKey in testdf.groups.keys():

    tempDict={}



    groupDF = testdf.get_group(groupKey)

    for i in range(0,len(groupDF)):

        tempDict[groupDF.iloc[i,0]]=groupDF.iloc[i,1]

    indexMap[ptr]=groupKey

    reverseIndexMap[groupKey] = ptr

    ptr=ptr+1

    listOfDictonaries.append(tempDict)
from sklearn.feature_extraction import DictVectorizer

dictVectorizer = DictVectorizer(sparse=True)

vector = dictVectorizer.fit_transform(listOfDictonaries)
from sklearn.metrics.pairwise import cosine_similarity

pairwiseSimilarity = cosine_similarity(vector)
def printBookDetails(bookID):

    print("Title:", bookDF[bookDF['id']==bookID]['original_title'].values[0])

    print("Author:",bookDF[bookDF['id']==bookID]['authors'].values[0])

    print("Printing Book-ID:",bookID)

    print("=================++++++++++++++=========================")





def getTopRecommandations(bookID):

    row = reverseIndexMap[bookID]

    print("------INPUT BOOK--------")

    printBookDetails(bookID)

    print("-------RECOMMENDATIONS----------")

    similarBookIDs = [printBookDetails(indexMap[i]) for i in np.argsort(pairwiseSimilarity[row])[-7:-2][::-1]]
getTopRecommandations(1245)
getTopRecommandations(4536)
getTopRecommandations(99)