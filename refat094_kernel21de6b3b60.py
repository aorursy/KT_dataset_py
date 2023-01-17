# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



#input genome data GRCh38 standerd human genome sample in csv format

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
#two genomw sample one having 10samples having 20 sequence

#second one have genome sequence over 100K

df = pd.read_csv('../input/data-final/final.csv', header=None)
#data sample 

df.head()
#data storing in variable y

y = df.iloc[:,0].values
y[0]
for i in df:print(df)
#initialization of required variables globally  

point=0

xmax=0

ymax=0

result=0

result1=0

result2=0

pt=[0,0,0,0,0,0,0,0,0,0,0]
def check(s , t , m , n , gap , match , mis_match):

    

    #edit = [[0 for i in range(2000)] for j in range(2000)]

    

    store = [[0 for i in range(30000)] for j in range(3)]

    

    

    print("Length of input genome sequences are")

    print(m-1,n-1)

    



    for x in range (0,m,1):

        store[0][x] = -x * gap

        #print(store[0][x])

    

    #using only 3 rows recursively,using memory efficiently   

    for i in range (0,n,1):

        for j in range (0,m,1):

            k = i % 2

            if(j == 0):

                store[k][j] = -i * gap

                #print(store[k][j])

                continue

            x=0

            if(s[i]==t[j]):

                x=match

                store[k][j]=max(max(store[1-k][j]-gap,store[k][j-1]-gap), store[1-k][j-1]+x)

                #print(store[k][j])

                ymax=max(store[1-k][j],store[k][j])

                xmax=max(store[k][j-1],store[k][j])

                #result1=store[k][j]

                

            else:

                x=mis_match

                store[k][j]=max(max(store[1-k][j]-gap,store[k][j-1]-gap), store[1-k][j-1]+x)

                #print(store[k][j])

                ymax=max(store[1-k][j],store[k][j]) 

                xmax=max(store[k][j-1],store[k][j])

                #result2=store[k][j]

                #print(result)

                

                

    result=max(xmax,ymax)          

    #result=store[k][j]

    #result=store[2][2]

    

    print("Match Score")

    print(result)

    

    if(m>n):

        point=(result/((m-1)*5))*100

    else:

        point=(result/((n-1)*5))*100

        

    print("Match percentage")

    print(point,"%")

    

    for x in range(0,9,1):

        pt[x]=point

    

    

    
def check2(s , t , m , n , gap , match , mis_match):

    

    #edit = [[0 for i in range(2000)] for j in range(2000)]

    

    store = [[0 for i in range(30000)] for j in range(3)]

    

    

    print("Length of input genome sequences are")

    print(m-1,n-1)

    



    for x in range (0,m,1):

        store[0][x] = -x * 0

        #print(store[0][x])

    

    #using only 3 rows recursively,using memory efficiently   

    for i in range (0,n,1):

        for j in range (0,m,1):

            k = i % 2

            if(j == 0):

                store[k][j] = -i * 0

                #print(store[k][j])

                continue

            x=0

            if(s[i]==t[j]):

                x=match

                store[k][j]=max(max(store[1-k][j]-gap,store[k][j-1]-gap), store[1-k][j-1]+x)

                #print(store[k][j])

                result1=store[k][j]

                

            else:

                x=mis_match

                store[k][j]=max(max(store[1-k][j]-gap,store[k][j-1]-gap), store[1-k][j-1]+x)

                #print(store[k][j])

                result2=store[k][j]

                #print(result)

                

                

    result=max(result1,result2)          

    #result=store[k][j]

    #result=store[2][2]

    

    print("Match Score")

    print(result)

    

    if(m>n):

        point=(result/((m-1)*5))*100

    else:

        point=(result/((n-1)*5))*100

        

    print("Match percentage")

    print(point,"%")

    

    for x in range(0,9,1):

        pt[x]=point

    
gap = 2

match = 5

mis_match = -1



#dest="TGCAGAGCTA"

#t = input("enter ur data: ")



#taking a sample genome data as target sequence

t = y[2]

t = "0" + t

m = len(t)

#print(m,t)



i = 0

while i<10:

    #print(y[i])

    #taking sample genome data as source sequence

    s = "0" + y[i]

    n = len(s)

    #print(n,s)

    #print(len(dest))

    

    #normal function considering gap value strictly 

    check(s , t , n , m , gap , match , mis_match)

    

    #function compatible when there are unmatched sequences at the beginning or end

    check2(s , t , n , m , gap , match , mis_match)

    

    i =i+1