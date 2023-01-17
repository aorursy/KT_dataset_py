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
def knapSack(W, wt, val, n):

    # Base Case

    if n == 0 or W == 0:

        return 0

    # Jika weight dari item ke-n lebih besar daripada kapasitas Knapsack W,

    # maka item tersebut tidak dapat dimasukan kedalam optimal solution

    if (wt[n-1] > W) :

        return knapSack(W, wt, val, n-1)

    

    # return nilai maksimum dari dua kemungkinan:

    # item ke-n dimasukan

    # tidak dimasukan

    else:

        return max(val[n-1] + knapSack(W-wt[n-1] , wt , val , n-1), 

                   knapSack(W , wt , val , n-1)) 

def knapSack(W, wt, val, n): 

    K = [[0 for x in range(W+1)] for x in range(n+1)] 

  

    for i in range(n+1): 

        for w in range(W+1): 

            if i==0 or w==0: 

                K[i][w] = 0

            elif wt[i-1] <= w: 

                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]) 

            else: 

                K[i][w] = K[i-1][w] 

        print(K[i])

  

    return K[n][W] 



print(knapSack(10, [3,10,5,25], [2,32,4,35], 4))
# Python 3 Program to find length of the  

# Longest Common Increasing Subsequence (LCIS) 

  

# Returns the length and the LCIS of two 

# arrays arr1[0..n-1] and arr2[0..m-1] 

def LCIS(arr1, n, arr2, m): 

  

    # table[j] is going to store length of LCIS 

    # ending with arr2[j]. We initialize it as 0, 

    table = [[]] * m 

    for j in range(m): 

        table[j] = []

  

    # Traverse all elements of arr1[] 

    for i in range(n): 

      

        # Initialize current length of LCIS 

#         current = 0

        current = []

  

        # For each element of arr1[],  

        # traverse all elements of arr2[]. 

        for j in range(m): 

              

            # If both the array have same elements. 

            # Note that we don't break the loop here. 

            if (arr1[i] == arr2[j]): 

                if (len(current) + 1 > len(table[j])): 

                    temp = current[:]

                    temp.append(arr2[j])

                    table[j] = temp

#                     table[j] = current + 1

  

            # Now seek for previous smaller common 

            # element for current element of arr1  

            if (arr1[i] > arr2[j]):

                if (len(table[j]) > len(current)): 

#                     current = table[j] 

                    current = table[j]

  

    # The maximum value in table[]  

    # is out result 

    return max([(len(x), x) for x in table])[1]

#     return [len(x) for x in table]



L1 = [2,3,4,1,41,22,12,5,59,23]

L2 = [3,4,2,1,2,34,41,56,63,59]



print(LCIS(L2, len(L2), L1, len(L1)))
def LongestPal(str):

    n = len(str)

    pali = ""

    reverse = ""

    palLen = 0

    

    for i in range(n):

        for j in range(i+1,n+1):

            substr = str[i:j]

            rev = substr[::-1]

            substrLen = len(substr)

        

            if rev in str:

                if str.find(rev,j) != -1:

                    if substrLen > palLen:

                        palLen = substrLen

                        pal = substr

                        reverse = rev

    

    return [pal, reverse]



LongestPal("REVERSE")
