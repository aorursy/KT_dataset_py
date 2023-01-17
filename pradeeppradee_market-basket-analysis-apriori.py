import pandas as pd
import numpy as np
df = pd.read_csv("../input/groceries/groceries - groceries.csv") #imported data
df.fillna(0,inplace=True) #preprocessing - - replacing Nan values with 'zero'
df
trans = []
for i in range(0,len(df)):
    trans.append([str(df.values[i,j]) for j in range(0,32) if str(df.values[i,j])!='0'])
from mlxtend.preprocessing import TransactionEncoder #imported trasactionencoder from mlxtend
encode = TransactionEncoder()
df = encode.fit_transform(trans)
df = pd.DataFrame(df, columns = encode.columns_)
df.columns
df.head()
from mlxtend.frequent_patterns import apriori
itemsets=apriori(df, min_support = 0.02, use_colnames = True)
itemsets           #support values for frequent itemsets
frequent_itemsets=itemsets.sort_values(by=['support'], ascending=False) #sorting to get top 10 frequent itemsets
Top_ten=frequent_itemsets[:10]
Top_ten
from mlxtend.frequent_patterns import association_rules
rules_mlxtend = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_mlxtend.head()
#rules=rules_mlxtend[(rules_mlxtend['lift'] >= 1)]
Top_rules=rules_mlxtend.sort_values(by=['lift'], ascending=False) #sorting to get top 10 rules based on lift
Top_rules[:10]
#Top_rules[Top_rules['antecedents'] == frozenset(['whole milk'])]
#Top_rules[Top_rules['antecedents'] == frozenset(['other vegetables'])]
#Top_rules[Top_rules['antecedents'] == frozenset(['root vegetables'])]
#Top_rules[Top_rules['antecedents'] == frozenset(['(pip fruit'])]
#Top_rules[Top_rules['antecedents'] == frozenset(['tropical fruit'])]
#Top_rules[Top_rules['antecedents'] == frozenset(['yogurt'])]


