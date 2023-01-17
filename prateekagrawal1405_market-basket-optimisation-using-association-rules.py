import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This is my first notebook and I am really excited to share this.



# Read the data into a dataframe . The data does not have a header row in the csv file so the first transaction gets used as column names .

# To prevent it from happening we use header=None so that we retain all our transaction details.



df = pd.read_csv('/kaggle/input/market-basket-optimization/Market_Basket_Optimisation.csv',header=None)

df.head() # have a look at what the data looks like. We can see we have a lot of transactions containing NaN values.
# replace all the nan values with '' and inplace=True to commit the changes into the dataframe

df.fillna('',axis=1,inplace=True)

df.head()
# TransactionEncoder is what we are gonna use to convert the transaction dataframe into a table with True and False values for all the items

# in the transactions.



# We are gonna use the apriori algorithm for Association rule mining so we import it from the frequent patterns module of the mlxtend library.





from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori
# convert the dataframe into a list of list where each inner list represents a transaction.



df_list = df.to_numpy().tolist()

df_list

dataset = list()

for i in range(len(df_list)) :

    item = list()

    for j in df_list[i] :

        if pd.notna(j):

            item.append(j)

    dataset.append(item)
# Create an instace of our TransactionEncoder cabslass 

te = TransactionEncoder()

# Fit and transform our dataset which is a list of lists into an array of True and False.

te_array = te.fit(dataset).transform(dataset)

te_array
# Convert this into a dataframe for better visualisation and for applying association rules onto the dataframe.



final_df = pd.DataFrame(te_array,columns=te.columns_)

# remove the first column as it does not contain any infomation

final_df.drop(columns=[''],axis=1,inplace=True)

final_df
# Use the apriori algorithm and the min_support for finding out items or group of items which have a support greater than the minimum support.



frequent_itemsets_ap = apriori(final_df, min_support=0.01, use_colnames=True)
frequent_itemsets_ap
# import association rules class to find association rules amonng the items/group of items which have a support greater than the min support.

from mlxtend.frequent_patterns import association_rules



# we have used the metric as confidence and min_threshold to filter out the rules based on these parameters.

rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.2)

# Convert the rules obtained into a dataframe for better visualisation

result = pd.DataFrame(rules_ap)

result.sort_values(by='lift',inplace=True,ascending=False)

result