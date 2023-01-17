import pandas as pd
df = pd.read_csv('../input/GroceryStoreDataSet.csv',names=['products'],header=None)
df
df.columns 
df.values
data = list(df["products"].apply(lambda x:x.split(',')))

data 
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

df.head()
from mlxtend.frequent_patterns import apriori
df1 = apriori(df,min_support=0.01, use_colnames=True)

df1
df1.sort_values(by="support",ascending=False)
def check_in_list(ruleitem, basketitem):  #helper checking function

#     print("basketitem=", basketitem)

#     print("ruleitem=",list(ruleitem))

    ret = all(t in list(ruleitem) for t in basketitem)

#     print(ret)

    return ret
# Uncomment below lines to see what above function does

# df2 = apriori(df,min_support=0.01, max_len=2, use_colnames=True)  #list of only 1 items

# df2 = df2[df2["itemsets"].apply(len) > 1]

# df2["check_in_list"] = df2["itemsets"].apply(check_in_list, args=(['BISCUIT', 'BREAD'],))

# df2 = df2.sort_values(by="support",ascending=False)  #sor

# df2
df1 = apriori(df,min_support=0.01, max_len=1, use_colnames=True)  #list of only 1 items

df1 = df1.sort_values(by="support",ascending=False)  #sort desc by support value

def next_item(basketitems):

    if basketitems is None:

        return df1["itemsets"][0]

    max_len_apriori=len(basketitems) + 1

    df2 = apriori(df,min_support=0.01, max_len=max_len_apriori, use_colnames=True)  #list of only 1 items

    df2 = df2[df2["itemsets"].apply(len) > max_len_apriori-1]

    df2["check_in_list"] = df2["itemsets"].apply(check_in_list, args=(basketitems,))

    df2 = df2[df2["check_in_list"] == True]

    df2 = df2.sort_values(by="support",ascending=False)  #sort desc by support value

#     print(df2)

    if(len(df2) > 0):

        return list(df2["itemsets"])[0]
item=next_item(None)

print(list(item))
item=next_item(['BISCUIT'])

print(list(item))
item=next_item(['BISCUIT', 'BREAD'])

print(list(item))
item=next_item(['BISCUIT', 'BREAD', 'MILK'])

print(list(item))