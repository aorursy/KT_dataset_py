# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!pip install apyori

from apyori import apriori



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



#loading the dataset

store_data = pd.read_csv("/kaggle/input/market-based-optimization/Market_Basket_Optimisation.csv")



# check the dataset

store_data

print("Size of data:",store_data.shape)

#Run test on 500 records (7500 records total)

records = []

for i in range(0,500):

    records.append([str(store_data.values[i,j]) for j in range(0,20)])



for i,j in enumerate(records):

    while 'nan' in records[i]: records[i].remove('nan')
#build the firsh apriori mode

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, mim_lift=3, min_length=2)

association_results = list(association_rules)



print("Total: ",len(association_results))



print(association_results[0])
#print(association_result)



for item in association_results:

    

    # first index of the inner list

    # Contains base item and add item

    pair = item[0] 

    items = [x for x in pair]       

    print("Rule: " + str(list(item.ordered_statistics[0].items_base)) + " -> " + str(list(item.ordered_statistics[0].items_add)))



    #second index of the inner list

    print("Support: " + str(item[1]))



    #third index of the list located at 0th

    #of the third index of the inner list



    print("Confidence: " + str(item[2][0][2]))

    print("Lift: " + str(item[2][0][3]))

    print("=====================================")