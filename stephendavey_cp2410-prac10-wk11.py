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
def merger(s1, s2, s):

    i = 0

    j = 0

    final_length = len(s1) + len(s2)

    while i+j < final_length:

        if j == len(s2) or (i < len(s1) and s1[i] < s2[j]):

            s[i+j] = s1[i]

            i += 1

        elif i == len(s1):  # incremented i is out of bounds

            s[i+j] = s2[j]

            j += 1

        elif s1[i] == s2[j]: # test for duplicates

            s[i+j] = s1[i]

            i += 1

            s2.pop(j)

            final_length -= 1

        else:

            s[i+j] = s2[j]

            j += 1



    return s[0:final_length]

            
s1 = [1,2,6,8]

s2 = [2,3,5,6,8]

s = [0] * (len(s1)+len(s2))

merger(s1, s2, s)

def insertion_sort(d):

    """Sort array into non-decreasing order of keys

    d: array of (key, value)"""

    

    for i in range(1, len(d)):  # Increment through array of keys

        current_key = d[i][0]

        j = i

        while j > 0 and d[j-1][0] > current_key:

            d[j][0] = d[j-1][0]  # swap items

            j -= 1

        d[j] = d[i]
def count_runs(s1, s2):

    """Count function for sorted subsequences s1, s2

    All votes should be sorted in sub-sequences, but a bit convoluted lagorithm to count """

    i = j = 0

    final_length = len(s1) + len(s2)

    

    if s1[0] > s2[0]:  # Swaps seqs so lowest starting int is s1

        s1, s2 = s2, s1

    curr = (0, 0)  # vote, count 

    lead = (0, 0)

    while i+j < final_length:

        # S1 always next lowest int; check s2[j] and s1[i+1]

        if i < len(s1)-1 and s1[i] == s2[j]:   

            # count another s1 el

            curr[1] += 2

            if s1[i] == s1[i+1]:

                curr[1] += 1

                i+= 1

            i += 1

            j += 1

        elif j == len(s2) or (i < len(s1)-1 and s[i+1] < s2[j]):

            curr = (s1[i+1], 1)

            i += 1

        elif i == len(s1)-1 and s[i] < s2[j]:

            curr[1] += 1

            i += 1

        else:  # swap to next lowest vote to count and continue

            s1, s2 = s2, s1

            i, j = j, i

            curr = (s1[i], 1)

            i += 1            

        if curr[1] > lead[1]:

            lead = curr

    return lead



def merge_sort(seq):

    """Recursive sorting algorithm that divides and conquers"""

    n = len(seq)

    

    if n < 2:  # Base case

        return

   

    mid = n // 2   # Divide

    lower = seq[:mid]

    higher = seq[mid:]

    

    merge_sort(lower)  # Conquer

    merge_sort(higher)

    

    count_runs(lower, higher)
s1 = [1,3,4,2,6,5,5,6,4,7]

s2 = [5,6,6,4,5,4,5,3,2,7]



low = [s1[0] if s1[0] < s2[0] else s2[0]][0]

low
def count_votes(s):

    """Returns tuple of candidate number, and number of votes recieved"""

    m = {}

    lead = [0,0]

    for vote in s:

        if vote not in m:

            m[vote] = 1

        else:

            m[vote] += 1

        if m[vote] > lead[1]:

            lead[0], lead[1] = vote, m[vote]

            

    return lead
s = [1,1,4,3,5,3,1]

count_votes(s)