



import numpy as np 

import pandas as pd





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from mlxtend.frequent_patterns import apriori,fpgrowth

from mlxtend.preprocessing import TransactionEncoder

df= pd.read_csv("../input/AllElectronics.csv") 





nb_unique_item=len(df.axes[1])

print(nb_unique_item)

nb_transaction=len(df.axes[0])

print(nb_transaction)

items=[]

print(df)

for i in range(nb_transaction):

    items.append([str(df.values[i,j])for j in range(nb_unique_item)])

for i in range(len(items)):

    while('nan' in (items[i])):

        items[i].remove("nan")

    

te = TransactionEncoder()

te_ary = te.fit(items).transform(items)

df = pd.DataFrame(te_ary, columns=te.columns_)

df



association_rule_ap=apriori(df,min_support=0.2,use_colnames=True)

association_rule_fp=fpgrowth(df, min_support=0.2,use_colnames=True)





    

    

print("Frequent pattern by using Apriori Algorithm")

print(association_rule_ap)

print("Frequent pattern by using FpGrowth Algorithm")

print(association_rule_fp)
import time

run_time_ap=[]

run_time_fp=[]

min_sup=[1,2,3,4,5,6,7,8,9]

for i in range(nb_transaction):

    st_time1=time.time()

    association_rule_ap=apriori(df,min_support=min_sup[i]*0.1,use_colnames=True)

    end_time1=time.time()

    run_time1=end_time1-st_time1

    run_time_ap.append(run_time1)

    st_time2=time.time()

    association_rule_fp=fpgrowth(df, min_support=min_sup[i]*0.1,use_colnames=True)

    end_time2=time.time()

    run_time2=end_time2-st_time2

    run_time_fp.append(run_time2)

    

    

   

    
import matplotlib.pyplot as plt

plt.plot(min_sup,run_time_ap,'*',color='blue')

plt.plot(min_sup,run_time_fp,'*',color='red')

plt.xlabel('Minimum support')

plt.ylabel("Run time")

print(run_time_fp)

print(run_time_ap)