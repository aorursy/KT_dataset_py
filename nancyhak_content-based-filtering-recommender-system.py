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
import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.metrics.pairwise import cosine_similarity
df=pd.read_excel('../input/exerciseCB.xlsx',sheet_name='CB - Simply Unary',header=1,index_col=0)
#Subsetting the initial dataframe

q_data = df.iloc[:20,:10]

user_data=df.iloc[:20,13:17].fillna(0)

answer_data=df.iloc[:20,18:22].fillna(0)

copy_1=q_data.copy()

copy_2=q_data.copy()
#I am going to iterate through users and follow the steps of 

#1) multiplying questions with user like/dislikes, 

#2) creating a user profile by summing the columns from step 1,

#3) multiplying this user profile in step 2 with initial table of questions

n=1

results=pd.DataFrame()

for i in user_data.columns:

    for col in q_data.columns:

        copy_2[col]=np.where(q_data.loc[:,col]==0,q_data.loc[:,col],user_data.loc[:,i]*q_data.loc[:,col])

        user_profile=pd.DataFrame(copy_2.sum(axis=0)).transpose()

        product=cosine_similarity(copy_1, user_profile).sum(axis=1)

    #Storing the results and then initiating empty lists again to have a fresh start for every user

    results[n]=product

    user_profile=[]

    product=[]

    n=n+1

results.index=user_data.index 

top = pd.DataFrame()

for r in results.columns:

    ranking = results.loc[:,r].sort_values(ascending = False)

    top[r]=ranking.index

top.columns=user_data.columns

#For User 4 I am recommending random questions because we have no data on the preference and feedback, we can also exclude that user but I think it is better to recommend smth then understand what the user likes

print(top.head(5))
#Exactly the same process with weights calculated row-wise to leverage on the number of topics in one question

weights=1/copy_1.sum(axis=1)

n=1

results=pd.DataFrame()

copy_2=q_data.copy()

for i in user_data.columns:

    for col in q_data.columns:

        copy_2[col]=np.where(q_data.loc[:,col]==0,q_data.loc[:,col],user_data.loc[:,i]*q_data.loc[:,col]*weights)

        user_profile=pd.DataFrame(copy_2.sum(axis=0)).transpose()

        product=cosine_similarity(copy_1, user_profile).sum(axis=1)

    results[n]=product

    user_profile=[]

    product=[]

    n=n+1

#Simply sorting values and making it look clean by having same column names

results.index=user_data.index 

top = pd.DataFrame()

for r in results.columns:

    ranking = results.loc[:,r].sort_values(ascending = False)

    top[r]=ranking.index

top.columns=user_data.columns

#For User 4 I am recommending random questions because we have no data on the preference and feedback, we can also exclude that user but I think it is better to recommend smth then understand what the user likes

print(top.head(5))
#I am going to iterate through users and follow the steps of 

#1) multiplying questions with user like/dislikes, 

#2) getting weights based on number of topics in a question

#3) getting IDF from frequency of occurance of a topic in all questions

#4) to get the user profile I used IDF, weights and question data

#5) multiplying this user profile in step 4 with initial table of questions

list2=[]

list3=[]

#Getting IDF

for i in q_data.index:

    list1=1/sum(q_data.loc[i,:])

    list2.append(list1)

for i in q_data.columns:

    list1=np.log10(20/sum(q_data.loc[:,i]))

    list3.append(list1)

list3=pd.DataFrame(list3).transpose()

list3.columns=q_data.columns

copy_2=q_data.copy()

n=1

results=pd.DataFrame()

#Getting user_profile by using questions data IDF and weights

for i in user_data.columns:

    for col in q_data.columns:

        copy_2[col]=np.where(q_data.loc[:,col]==0,q_data.loc[:,col],user_data.loc[:,i]*q_data.loc[:,col]*list2*list3.loc[0,col])

        user_profile=pd.DataFrame(copy_2.sum(axis=0)).transpose()

        product=cosine_similarity(copy_1, user_profile).sum(axis=1)

    results[n]=product

    user_profile=[]

    product=[]

    n=n+1

#Simply sorting values and making it look clean by having same column names

results.index=user_data.index 

top = pd.DataFrame()

for r in results.columns:

    rank = results.loc[:,r].sort_values(ascending = False)

    top[r]=rank.index

top.columns=user_data.columns

#For User 4 I am recommending random questions because we have no data on the preference and feedback, we can also exclude that user but I think it is better to recommend smth then understand what the user likes

print(top.head(5))

#Initialising empty lists

list2=[]

list3=[]

copy_1=q_data.copy()

#Obtaining IDF by duplicating the same steps described in the previous method

for i in q_data.index:

    list1=1/sum(q_data.loc[i,:])

    list2.append(list1)

for i in q_data.columns:

    list1=np.log10(20/sum(q_data.loc[:,i]))

    list3.append(list1)

list3=pd.DataFrame(list3).transpose()

list3.columns=q_data.columns

n=1

results=pd.DataFrame()

copy_2=q_data.copy()

#Resolving cold start issue by returning the average of other users for the new user

for i in user_data.columns:

    if ((user_data.loc[:,i].mean())==0 and (user_data.loc[:,i].std())==0):

        results.loc[:,n]=results.loc[:,results.columns!=n].mean(axis=1)

    else:

        copy_2=q_data.copy()

        #Product of User Feedback, IDF, Weights and topic column 

        for col in q_data.columns:

            copy_2[col]=np.where(q_data.loc[:,col]==0,q_data.loc[:,col],user_data.loc[:,i]*q_data.loc[:,col]*list2*list3.loc[0,col])

            user_profile=pd.DataFrame(copy_2.sum(axis=0)).transpose()

            product=cosine_similarity(copy_1, user_profile).sum(axis=1)

        results[n]=product

        user_profile=[]

        product=[]

        n=n+1

#Simply sorting values and making it look clean by having same column names

results.index=user_data.index   

top=pd.DataFrame()

for r in results.columns:

    ranking=results.loc[:,r].sort_values(ascending = False)

    top[r]=ranking.index

top.columns=user_data.columns

print(top.head(5))
#In the Notebook I will explain the steps but the rationale can be found in the PDF

#Rescaling the data in User Answers to logically merge it with User Feedback

copy_3=q_data.copy()

answer_data.columns=user_data.columns

V1=pd.DataFrame(preprocessing.scale(answer_data))

a_copy=answer_data.copy()



#After scaling some data is distorted in particluar the zero values, hence I take two steps to recover them

#1) I create a copy table which holds duplicates

#2) I multiply values with that table to get values back because they hold important info to us

for a in a_copy.columns:

    for c in a_copy.index:

        if (a_copy.loc[c,a]!=0):

            a_copy.loc[c,a]=1

m_product=np.multiply(V1,a_copy)

m_product.columns=answer_data.columns

m_product.index=answer_data.index

a_data=m_product.copy()

#I am following the logic from previous methods and multiplying the user data with answer data which was molded from user feedback and answers

for n in user_data.columns:

    for p in user_data.index:

        if (user_data.loc[p,n]==1):

            if (a_data.loc[p,n]!=0):

                user_data.loc[p,n]=user_data.loc[p,n]*a_data.loc[p,n]

                #Here I have a condition for negative values because if I don't treat them specially I will receive a positive based on two negatives, which does not correctly mirror the User Profile

            elif(user_data.loc[p,n]== -1):

                if (a_data.loc[p,n]<0):

                    user_data.loc[p,n]=(-1)*user_data.loc[p,n]*a_data.loc[p,n]

                if(answer_data.loc[p,n]>0):

                    user_data.loc[p,n]=(-1)*user_data.loc[p,n]*a_data.loc[p,n]

                #Here I have one more condition for zeros because we want to cover for cases when the person did not like/dislike but still gave a good answer, I wanted to count this in 

            elif(user_data.loc[p,n]== 0):

                if (a_data.loc[p,n]!=0):

                    user_data.loc[p,n]=a_data.loc[p,n]  

#Initializing empty lists

list2=[]

list3=[]

#Assigning weights based on frequency of occurence of a topic

for i in q_data.index:

    list1=1/sum(q_data.loc[i,:])

    list2.append(list1)

#Computing IDF

for i in q_data.columns:

    list1=np.log10(20/sum(q_data.loc[:,i]))

    list3.append(list1)

list3=pd.DataFrame(list3).transpose()

list3.columns=q_data.columns

n=1

results=pd.DataFrame()

#Here I make sure that for users who have a cold start, hence a st.dev of zero average is taken from other users' predictions

for i in user_data.columns:

    if ((user_data.loc[:,i].std())==0):

        results.loc[:,n]=results.loc[:,results.columns!=n].mean(axis=1)

    else:

        copy_2=q_data.copy()

        #Same methodology as in previous steps of using product of user profile(in this case a new one), weights and question data

        for col in q_data.columns:

            copy_2[col]=np.where(q_data.loc[:,col]==0,q_data.loc[:,col],user_data.loc[:,i]*q_data.loc[:,col]*list2*list3.loc[0,col])

            user_profile=pd.DataFrame(copy_2.sum(axis=0)).transpose()

            product=cosine_similarity(copy_3, user_profile).sum(axis=1)

        results[n]=product

        user_profile=[]

        product=[]

        n=n+1

results.index=user_data.index 

#Simply sorting values and making it look clean by having same column names

top=pd.DataFrame()

for r in results.columns:

    ranking=results.loc[:,r].sort_values(ascending = False)

    top[r]=ranking.index

top.columns=user_data.columns

print(top.head(5))