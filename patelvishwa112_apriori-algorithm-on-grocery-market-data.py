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
#pip install apyori
# Import Data from CSV file 

dataset = pd.read_csv('/kaggle/input/groceries/groceries - groceries.csv',sep=',')
# View the dataset

dataset.head(10)
# Apriori algorithm takes the list of items that were bought together as input. Hence, we need to get each row 

# as list (except 1st column and 'NAN' in the columns).

# Create a list of trasactions

transactions = []



# Add all the items from each row in one list( Neglect the 1st columns where all the items are in number (0-9))

for i in range(0, 9835):

    transactions.append([str(dataset.values[i,u]) for u in range(1, 33)])

    

# Training the Apriori Algorithms

from apyori import apriori

rules = apriori(transcations, min_support=0.0022, min_confidence=0.20, min_lift=3, min_length = 2)



# Min_support  = 3(3 times a day) * 7 (7 days a week) / 9835 = 0.0022

# Min_confidence = set it lower to get more relations between products (weak relations), if we set it high then 

# we might miss some. I have selected confidence of 0.20

# Min_lift = In order to get some relevant rules, I am setting min_lift to 3.
# Store rules in result variable

results = list(rules)



# See the items that were bought together with their support

results_list = []

for i in range(0, len(results)):

    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
#Print results to see the common things bought together at market