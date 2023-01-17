import pandas as pd

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori 

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori 

from mlxtend.preprocessing import TransactionEncoder

import pandas as pd

import numpy as np

import pandas as pd

%matplotlib inline

pd.options.display.max_rows = 100

import warnings

warnings.filterwarnings('ignore')

sns.set(style='darkgrid')

plt.rcParams["patch.force_edgecolor"] = True

import os

print(os.listdir("../input"))

testdata=pd.read_csv("../input/testcsv.csv")

testdata.shape

from mlxtend.frequent_patterns import association_rules
file = testdata

file.shape

file.dropna(how='all',inplace=True)

file.head()

strings = []

for i in range(0,2000):

    strings.append([str(file.values[i,j]) for j in range(0,25) ])

#strings=np.array(strings)

new=[]

for j in strings:

    j=[j for j in j if str(j)!='nan']

    new.append(j)

te = TransactionEncoder()

te_ary = te.fit_transform(new)

df = pd.DataFrame(te_ary, columns=te.columns_)

df.shape

frequent_itemsets_test = apriori(df, min_support=0.001,use_colnames=True)
rule_test = association_rules(frequent_itemsets_test, metric="confidence",min_threshold=0.4)
rule_test[rule_test['antecedents'] == {'whole milk'}]