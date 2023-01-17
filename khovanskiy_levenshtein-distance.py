# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



def distance(a, b):

    "Calculates the Levenshtein distance between a and b."

    n, m = len(a), len(b)

    if n > m:

        # Make sure n <= m, to use O(min(n,m)) space

        a, b = b, a

        n, m = m, n



    current_row = range(n+1) # Keep current and previous row, not entire matrix

    for i in range(1, m+1):

        previous_row, current_row = current_row, [i]+[0]*n

        for j in range(1,n+1):

            add, delete, change = previous_row[j]+1, current_row[j-1]+1, previous_row[j-1]

            if a[j-1] != b[i-1]:

                change += 1

            current_row[j] = min(add, delete, change)



    return current_row[n]



df = pd.read_csv('../input/train.csv')

for row in df.itertuples():

    if row[6] != 0:

        d = distance(row[4], row[5])

        ml = max(len(row[4]), len(row[5]))

     

        print(row[0], row[4], '|', row[5], row[6], 1-d/ml)

# Any results you write to the current directory are saved as output.