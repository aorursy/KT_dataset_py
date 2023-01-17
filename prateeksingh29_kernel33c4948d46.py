# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
dataset =pd.read_csv("../input/movie.csv")

dataset.head()

item =pd.read_csv("../input/item.csv",encoding = "ISO-8859-1", index_col = False)

data = pd.merge(dataset, item,left_on='itemID',right_on='itemID')
userIds=data.userID # a Pandas series object

userIds2=data[['userID']]
userIds.head()

userIds2.head()
type(userIds)
data

data.loc[0:10,['userID']]
toyStoryUsers=data[data.title=="Toy Story (1995)"]



toyStoryUsers.head()
data=pd.DataFrame.sort_values(data,['userID','itemID'],ascending=[0,1])
numUsers=max(data.userID)

numMovies=max(data.itemID)
moviesPerUser=data.userID.value_counts()

usersPerMovie=data.title.value_counts()

usersPerMovie
def favoriteMovies(activeUser,N):



    topMovies=pd.DataFrame.sort_values(

        data[data.userID==activeUser],['rating'],ascending=[0])[:N] 

    return list(topMovies.title)



print(favoriteMovies(5,4))
userItemRatingMatrix=pd.pivot_table(data, values='rating',

                                    index=['userID'], columns=['itemID'])
userItemRatingMatrix.head()
from scipy.spatial.distance import correlation 

def similarity(user1,user2):

    user1=np.array(user1)-np.nanmean(user1) 

    user2=np.array(user2)-np.nanmean(user2)

    commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]



    if len(commonItemIds)==0:

    

        return 0

    else:

        user1=np.array([user1[i] for i in commonItemIds])

        user2=np.array([user2[i] for i in commonItemIds])

        return correlation(user1,user2)
def nearestNeighbourRatings(activeUser,K):



    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,

                                  columns=['Similarity'])



    for i in userItemRatingMatrix.index:

        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],

                                          userItemRatingMatrix.loc[i])



    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,

                                              ['Similarity'],ascending=[0])



    nearestNeighbours=similarityMatrix[:K]

  

    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]



    predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])



    for i in userItemRatingMatrix.columns:

      

        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])

        for j in neighbourItemRatings.index:

      

            if userItemRatingMatrix.loc[j,i]>0:

              

                predictedRating += (userItemRatingMatrix.loc[j,i]

                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']



        predictItemRating.loc[i,'Rating']=predictedRating

    return predictItemRating

def nearestNeighbourRatings(activeUser,K):



    similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,

                                  columns=['Similarity'])



    for i in userItemRatingMatrix.index:

        similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],

                                          userItemRatingMatrix.loc[i])



    similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,

                                              ['Similarity'],ascending=[0])



    nearestNeighbours=similarityMatrix[:K]

 

    neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]



    predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])



    for i in userItemRatingMatrix.columns:

        

        predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])

        

        for j in neighbourItemRatings.index:

         

            if userItemRatingMatrix.loc[j,i]>0:

           

                predictedRating += (userItemRatingMatrix.loc[j,i]

                                    -np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']

        

        predictItemRating.loc[i,'Rating']=predictedRating

    return predictItemRating

def topNRecommendations(activeUser,N):

    predictItemRating=nearestNeighbourRatings(activeUser,10)

    

    moviesAlreadyWatched=list(userItemRatingMatrix.loc[activeUser]

                              .loc[userItemRatingMatrix.loc[activeUser]>0].index)



    predictItemRating=predictItemRating.drop(moviesAlreadyWatched)

    topRecommendations=pd.DataFrame.sort_values(predictItemRating,

                                                ['Rating'],ascending=[0])[:N]

 

    topRecommendationTitles=(item.loc[item.itemID.isin(topRecommendations.index)])

    return list(topRecommendationTitles.title)
activeUser=5

print(favoriteMovies(activeUser,5),"\n",topNRecommendations(activeUser,3))
def matrixFactorization(R, K, steps=10, gamma=0.001,lamda=0.02):

 

    N=len(R.index)# Number of users

    M=len(R.columns) # Number of items 

    P=pd.DataFrame(np.random.rand(N,K),index=R.index)



    Q=pd.DataFrame(np.random.rand(M,K),index=R.columns)



    for step in range(steps):

 

        for i in R.index:

            for j in R.columns:

                if R.loc[i,j]>0:

                   

                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])

                  

                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])

          

                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])

      

        e=0

        for i in R.index:

            for j in R.columns:

                if R.loc[i,j]>0:

                

                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))

        if e<0.001:

            break

        print(step)

    return P,Q



(P,Q)=matrixFactorization(userItemRatingMatrix.iloc[:100,:100],K=2,gamma=0.001,lamda=0.02, steps=100)
activeUser=1

predictItemRating=pd.DataFrame(np.dot(P.loc[activeUser],Q.T),index=Q.index,columns=['Rating'])

topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:3]

topRecommendationTitles=item.loc[item.itemID.isin(topRecommendations.index)]

print(list(topRecommendationTitles.title))

import itertools 



allitems=[]



def support(itemset):

    userList=userItemRatingMatrix.index

    nUsers=len(userList)

    ratingMatrix=userItemRatingMatrix

    for item in itemset:

        ratingMatrix=ratingMatrix.loc[ratingMatrix.loc[:,item]>0]

        userList=ratingMatrix.index

    return float(len(userList))/float(nUsers)
minsupport=0.3

for item in list(userItemRatingMatrix.columns):

    itemset=[item]

    if support(itemset)>minsupport:

        allitems.append(item)
len(allitems)
minconfidence=0.1

assocRules=[]

i=2

for rule in itertools.permutations(allitems,2):

    from_item=[rule[0]]

    to_item=rule

    confidence=support(to_item)/support(from_item)

    if confidence>minconfidence and support(to_item)>minsupport:

        assocRules.append(rule)

assocRules
