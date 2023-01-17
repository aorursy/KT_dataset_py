

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules



import mlxtend as ml



data = pd.read_csv('../input/dokdoktemiz/dokdoktemiz.csv')

data
data.drop('muayeneler.ortopedi',axis=1, inplace=True)



data.drop('aksam',axis=1, inplace=True)



data.drop('ameliyat1',axis=1, inplace=True)

data.drop('ameliyat2',axis=1, inplace=True)

data.drop('ameliyat3',axis=1, inplace=True)

data.drop('ameliyat4',axis=1, inplace=True)



data.drop('adetduzen',axis=1, inplace=True)

data.drop('adetsuan',axis=1, inplace=True)

data.drop('adetgun',axis=1, inplace=True)

data.drop('menopoz',axis=1, inplace=True)



data.drop('alerjikhastalikilac',axis=1, inplace=True)



data.drop('alkol',axis=1, inplace=True)

data.drop('alkolmiktar',axis=1, inplace=True)



#boyu,cay,yas,kilo,su,uykusuresi,yuruyuskm



data.drop('ilacalerjisi',axis=1, inplace=True)

data.drop('ilacalerjisiilaci',axis=1, inplace=True)



data.drop('kahvalti',axis=1, inplace=True)



data.drop('kalpilac',axis=1, inplace=True)



data.drop('kayittarihi',axis=1, inplace=True)



data.drop('meslek',axis=1, inplace=True)



data.drop('romatizmailac',axis=1, inplace=True)



data.drop('searchKey',axis=1, inplace=True)



data.drop('sekerilac',axis=1, inplace=True)



data.drop('spor',axis=1, inplace=True)

data.drop('sporsure',axis=1, inplace=True)

data.drop('sporsıklık',axis=1, inplace=True)

data.drop('sporturu',axis=1, inplace=True)



data.drop('tansiyonilac',axis=1, inplace=True)



data.drop('yuruyuskm',axis=1, inplace=True)

birliktelikdata=data[:].applymap(str)

birliktelikdata.info()
birliktelikdata=pd.get_dummies(birliktelikdata)

birliktelikdata.info()

birliktelikdata
apriori(birliktelikdata, min_support=0.15)[1:]
print("Kural Sayısı:", len(apriori(birliktelikdata, min_support=0.15)))

apriori(birliktelikdata, min_support=0.15, use_colnames=True)[1:]

frequent_itemsets = apriori(birliktelikdata, min_support=0.15, use_colnames=True)





rules1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.30)
print("Oluşan Kural Sayısı:", len(rules1))

rules1 = rules1.sort_values(['confidence'], ascending=False)



rules1[1:11]

rules1["antecedent_len"] = rules1["antecedents"].apply(lambda x: len(x))

rules1["consequents_len"] = rules1["consequents"].apply(lambda x: len(x))

rules1[1:6]
rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)



rules2 = rules2.sort_values(['lift'], ascending=False)



rules2[1:6]
rules2["antecedent_len"] = rules2["antecedents"].apply(lambda x: len(x))



rules2["consequents_len"] = rules2["consequents"].apply(lambda x: len(x))



rules2[1:6]
rules1[(rules1['antecedent_len'] >= 1) &

       (rules1['confidence'] >= 0.20) &

       (rules1['lift'] > 1) ].sort_values(['confidence'], ascending=False)[1:10]