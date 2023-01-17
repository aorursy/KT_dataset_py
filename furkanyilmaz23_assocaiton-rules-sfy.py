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
df = pd.read_csv("../input/groceries/groceries.csv",names=["urunler"],header = None , sep =";") #Veri setini import ettik , tablomuzun başlığını düzenleyip verileri ; ayırdık.
df.head() # veri setini inceliyoruz
df.describe() # genel bir ortalama hesapı yaptık 
df
data = list(df["urunler"].apply(lambda x : x.split(","))) # Veri setini başka bir değişkene atayıp valuesleri tek tek virgülle ayırdık

data  
from mlxtend.preprocessing import TransactionEncoder #TransactionEncoder Encoder class for transaction data in Python lists
from mlxtend.frequent_patterns import apriori, association_rules 
te = TransactionEncoder()

te_data= te.fit(data).transform(data)

sfy = pd.DataFrame(te_data,columns=te.columns_)

sfy
from mlxtend.frequent_patterns import apriori
df1 = apriori(sfy,min_support=0.01,use_colnames=True)

df1
df1['length'] = df1['itemsets'].apply(lambda x:len(x))

df1
df1[(df1['length']==3) & (df1['support']>=0.01)]
rules = association_rules(df1, metric ="confidence", min_threshold = 0.20) 

rules.head(10) 