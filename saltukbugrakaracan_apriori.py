import pandas as pd # importing required library pandas
grocery = pd.read_csv("../input/supermarket/GroceryStoreDataSet.csv",header=None,names=['products']) #creating dataframe
df = grocery.copy()
df.head() #first five row of the dataframe
df.values #values of each row
df.shape # shape of the dataframe. (20 rows, 1 column)
#!pip install mlxtend #for using mlextend library you have to run this cell. After running once you can comment out
df.columns #showing column names
data = list(df["products"].apply(lambda x:x.split(','))) #splitting all rows by comma. Through this cell products are seperated
data
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()

te_data = te.fit(data).transform(data)

df = pd.DataFrame(te_data,columns=te.columns_)

#this cell converts all products to a column and if the product exists in the row, writes True inside of this index
df
te_data#all true false values for each basket.
from mlxtend.frequent_patterns import apriori
df1 = apriori(df,min_support=0.2,use_colnames=True,verbose=1)

#If minimum support value is more than 0.2 put it in the new dataframe.
df1
df1.sort_values(by="support",ascending=False)#sorting operation
df1['length'] = df1['itemsets'].apply(lambda x:len(x))#add length column to a dataframe

df1
from mlxtend.frequent_patterns import association_rules
association_rules(df1, metric = "confidence", min_threshold = 0.2)

#if confidence values are more than 0.2 bring these values.
association_rules(df1, metric = "support", min_threshold = 0.2)

#if support values are more than 0.2 bring these values.
association_rules(df1, metric = "lift", min_threshold = 1.0)

#if lift values are more than 1 bring these values.