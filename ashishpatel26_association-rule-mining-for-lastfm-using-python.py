!pip install apyori

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from apyori import apriori

%matplotlib inline

import os

print(os.listdir("../input"))
lastfm1 = pd.read_csv("https://www.biz.uiowa.edu/faculty/jledolter/DataMining/lastfm.csv")
lastfm = lastfm1.copy()

lastfm.shape
lastfm = lastfm[['user','artist']]
lastfm = lastfm.drop_duplicates()

lastfm.shape
records = []

for i in lastfm['user'].unique():

    records.append(list(lastfm[lastfm['user'] == i]['artist'].values))
print(type(records))
association_rules = apriori(records, min_support=0.01, min_confidence=0.4, min_lift=3, min_length=2)

association_results = list(association_rules)
print("There are {} Relation derived.".format(len(association_results)))
for i in range(0, len(association_results)):

    print(association_results[i][0])
for item in association_results:

    # first index of the inner list

    # Contains base item and add item

    pair = item[0]

    items = [x for x in pair]

    print("Rule: With " + items[0] + " you can also listen " + items[1])



    # second index of the inner list

    print("Support: " + str(item[1]))



    # third index of the list located at 0th

    # of the third index of the inner list



    print("Confidence: " + str(item[2][0][2]))

    print("Lift: " + str(item[2][0][3]))

    print("=====================================")