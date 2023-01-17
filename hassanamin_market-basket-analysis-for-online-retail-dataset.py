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
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
retail = pd.read_csv('../input/onlineretail/OnlineRetail.csv', encoding = 'unicode_escape')

retail.head()
retail['Description'] = retail['Description'].str.strip()

retail.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

retail['InvoiceNo'] = retail['InvoiceNo'].astype('str')

retail = retail[~retail['InvoiceNo'].str.contains('C')]
basket = (retail[retail['Country'] =="France"]

          .groupby(['InvoiceNo', 'Description'])['Quantity']

          .sum().unstack().reset_index().fillna(0)

          .set_index('InvoiceNo'))
basket.head()
def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_sets = basket.applymap(encode_units)

basket_sets.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.head()
# Import seaborn under its standard alias

import seaborn as sns

import matplotlib.pyplot as plt





# Generate scatterplot using support and confidence

sns.scatterplot(x = "support", y = "confidence", 

                size = "lift", data = rules)

plt.show()

rules[ (rules['lift'] >= 6) &

       (rules['confidence'] >= 0.8) ]
print("ALARM CLOCK BAKELIKE GREEN : ",basket['ALARM CLOCK BAKELIKE GREEN'].sum())

print("ALARM CLOCK BAKELIKE RED : ",basket['ALARM CLOCK BAKELIKE RED'].sum())
basket2 = (retail[retail['Country'] =="Germany"]

          .groupby(['InvoiceNo', 'Description'])['Quantity']

          .sum().unstack().reset_index().fillna(0)

          .set_index('InvoiceNo'))





basket2.head()
basket_sets2 = basket2.applymap(encode_units)

basket_sets2.drop('POSTAGE', inplace=True, axis=1)

frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)

rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

# Import seaborn under its standard alias

import seaborn as sns

import matplotlib.pyplot as plt





# Generate scatterplot using support and confidence

sns.scatterplot(x = "support", y = "confidence", 

                size = "lift", data = rules2)

plt.show()

rules2[ (rules2['lift'] >= 4) &

        (rules2['confidence'] >= 0.5)]
# Import seaborn under its standard alias

import seaborn as sns

import matplotlib.pyplot as plt



# Transform the DataFrame of rules into a matrix using the lift metric



pivot = rules2.pivot(index = 'consequents', columns = 'antecedents', values= 'lift')



# Generate a heatmap with annotations on and the colorbar off



sns.heatmap(pivot, annot = True, cbar=False)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
