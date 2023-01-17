
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
%matplotlib inline
import os



#Big shout out to Sebastian Raschka. 
#Thank you for the beautiful Library and for getting me into Data Science -Python Machine Learning

#Load data and transform for analysis

df = pd.read_csv('../input/BreadBasket_DMS.csv')

transactions=[]

#Might look at this later 
item_sets = {}

for t,g in df.groupby('Transaction')['Item']:
    transactions.append(g.tolist())
    item_sets[t] = g.tolist()
    
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
ap = pd.DataFrame(te_ary, columns=te.columns_)
#Quick look at the top 25 items

ap.sum().to_frame('Frequency').sort_values('Frequency',ascending=False)[:25].plot(kind='bar',
                                                                                  figsize=(12,8),
                                                                                  title="Frequent Items")
plt.show()
#Take a look at various support levels and confidences and determine a good threshold.

ap_0_5 = {}
ap_1 = {}
ap_5 = {}
ap_1_0 = {}

confidence = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def gen_rules(df,confidence,support):
    ap = {}
    for i in confidence:
        ap_i =apriori(df,support,True)
        rule= association_rules(ap_i,min_threshold=i)
        ap[i] = len(rule.antecedents)
    return pd.Series(ap).to_frame("Support: %s"%support)

confs = []
for i in [0.005,0.01,0.05,0.1]:
    ap_i = gen_rules(ap,confidence=confidence,support=i)
    confs.append(ap_i)

all_conf = pd.concat(confs,axis=1)

all_conf.plot(figsize=(8,8),grid=True)
plt.ylabel('Rules')
plt.xlabel('Confidence')
plt.show()
#Look at Support: 0.01 and Confidence = 0.3

ap_final =  apriori(ap,0.01,True)
rules_final = association_rules(ap_final,min_threshold=.03,support_only=False)

rules_final[rules_final['confidence'] > 0.5]