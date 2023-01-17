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
dataset=pd.read_csv('/kaggle/input/market-basket-optimization/Market_Basket_Optimisation.csv',header=None)

dataset.head()

dataset.shape

!pip install apyori

#Downloading Apyori
#creating a list of list format

transactions=[]

for i in range(0,7501):

    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

print(transactions)

    

    
#Analysing the model

from apyori import apriori

results=apriori(transactions,min_confidence=0.2,min_support=0.003,min_lift=3,length=2)

#Simple visualization

final_res=list(results)

final_res
df_results=pd.DataFrame(final_res)

df_results.head()
df_results.sort_values(by='support')
df_results.shape[0]
#preparing in a ordered_format

first_values=[]

second_values=[]

third_values=[]

fourth_value=[]

for i in range(0,df_results.shape[0]):

    single_list=df_results['ordered_statistics'][i][0]

    first_values.append(single_list[0])

    second_values.append(single_list[1])

    third_values.append(single_list[2])

    fourth_value.append(single_list[3])



df_lhs=pd.DataFrame(first_values)

df_rhs=pd.DataFrame(second_values)

confidence=pd.DataFrame(third_values,columns=['confidence'])

support=pd.DataFrame(fourth_value,columns=['support'])

    
df_final=pd.concat([df_lhs,df_rhs,support,confidence],axis=1)

df_final

df_final.sort_values(by='confidence',ascending=False)
df_final.sort_values(by='confidence',ascending=False).head(2).plot.bar()
