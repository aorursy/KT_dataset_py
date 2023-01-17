# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


#Question 1:

#Solution:1

a=[]

a = [x-(x-1) for x in range(1,25)]



print(a)



#Solution:2



b=[]

for i in range(25):

    b.append(1)



print (b)



#Solution3:

a = list(range(25))

a = [1 for x in a]

a



#Question 2:



#Solution:1



alphabet=[]

for i in range(65,91):

    alphabet.append(chr(i))

    

print (alphabet)



#Solution:2



alphabet2 = [chr(i) for i in range(65,91)]

print (alphabet2)





#Question 3:

#Solution:1

reverse=[]

#sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]

sequence =[x for x in range(1,10)]

#print (inp)

print("Sequence is:")

print(sequence)



reverse = sequence[::-1]

print("Reverse of sequence is:")

print(reverse)



#Question 4:

inputArray = [5,9,2,10,7,4,8,1,6,3]

#Please clear question 4, not able to understand





#Question4: Need more understanding





#Question5:



#Solution:1 

numericList= [5,9,2,10,7,4,8,1,6,3]

strList= ['FIve','Nine','Two','Ten','Seven','Four','Eight','One','Six','Three']

print("before Sorting numericList:")

print(numericList)

print("before Sorting strList:")

print(strList)



#mapped = zip(numericList,strList)

#mapped = set(mapped)

print("zip function map indivisual list elements")

#print(mapped)



#dicSortedbyValue =[values for key,values in sorted(dict(zip(numericList,strList)).items())]

dicSortedbyValue ={values for key,values in sorted(zip(numericList,strList)).values}

#dicSortedbyKey = sorted(dict(mapped)[])

print("Post Sorting numericList:")

numericList.sort()

print (numericList)

print("Post Sorting  strList:")

print(dicSortedbyValue)

##mapping of dictionary value back to y1)

#y1 = [i for i in dicSortedbyKey.keys()]

#print(y1)



#Program for Quick sort






