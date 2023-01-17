import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df_train = pd.read_csv('data/train.csv', index_col='client_id')

df_clients = pd.read_csv('data/clients2.csv', index_col='client_id')

df_test = pd.read_csv('data/test.csv', index_col='client_id')

df_products = pd.read_csv('data/products.csv')

df_pursh = pd.read_csv('train_purch/train_purch.csv')
df_train.head()
df_train.shape
df_clients.head()
df_clients.shape
df_test.shape
indices_train = df_train.index

df_clients_train = df_clients.loc[indices_train]

df_train_all = df_clients_train.merge(df_train, right_index=True, left_index=True)

df_train_all.shape
indices_test = df_test.index

df_clients_test = df_clients.loc[indices_test]

df_test_all = df_clients_test.merge(df_test, right_index=True, left_index=True)

df_test_all.shape
df_test_all = df_test_all.drop(['client_id.1'], axis = 1)

df_train_all = df_train_all.drop(['client_id.1'], axis = 1)
df_train_all.head()
df_train_all.info()   #nan только в 'first_redeem_date'
df_train_all.gender = df_train_all.gender.map({'F':-1, 'U':0, 'M': 1})  #female -> -1 , unknown -> 0, male -> 1

df_test_all.gender = df_test_all.gender.map({'F':-1, 'U':0, 'M': 1})
df_train_all.head()
#Метод для вычисления аплифта

def uplift_score(data):

    return data[data.treatment_flg == 1].target.mean() - data[data.treatment_flg == 0].target.mean()
uplift_score(df_train_all[df_train_all.gender==-1])
uplift_score(df_train_all[df_train_all.gender==1])
max_uplf = 0

#найдем возраст, до которого и после аплифт наиболее отличается: 

for i in range(18,60, 1):

    if max_uplf < (uplift_score(df_train_all[df_train_all.age>i]) - uplift_score(df_train_all[df_train_all.age<=i])):

        max_uplf = (uplift_score(df_train_all[df_train_all.age>i]) - uplift_score(df_train_all[df_train_all.age<=i]))

        print(i)
print(uplift_score(df_train_all[df_train_all.age<=53]))

print(uplift_score(df_train_all[df_train_all.age>53]))
#Новый признак - возраст больше 53 или меньше 53 (вкллючительно)

df_train_all['age_53'] = df_train_all.age.apply(lambda x: 1 if (x > 53) else 0)
(uplift_score(df_train_all[df_train_all.age>34]) - uplift_score(df_train_all[df_train_all.age<=34]))
df_train_all.head()
df_train_all.gender.hist()

plt.show()
df_train_all.age_53.hist()

plt.show()  #довольно много пожилых клиентов
df_pursh.head()
most_pop_product = df_pursh['product_id'].value_counts(normalize=True).index[0]
most_pop_product
train_with_most = set(df_pursh[df_pursh.product_id==most_pop_product].client_id)

train_without_most = set(indices_train) - train_with_most
uplift_score(df_train.loc[list(with_most)]) - uplift_score(df_train.loc[list(train_without_most)])
df_train.loc[list(with_most)]

df_most_popular = pd.concat([df_train.loc[list(with_most)],df_train.loc[list(train_without_most)]]).drop(['target', 'treatment_flg'],

                                                                                                        axis = 1)

df_most_popular['most_popular'] = 0

df_most_popular['most_popular'][:df_train.loc[list(with_most)].shape[0]] = 1 
df_train_all = df_train_all.merge(df_most_popular, right_index=True, left_index=True)
df_train_all.shape
df_train_all.head()
gr = df_pursh.groupby(['client_id', 'transaction_id'])['purchase_sum'].mean()  #убираем повторение в транзакциях, одна транзакция = одна покупка

pr = gr.reset_index().groupby(['client_id'])['purchase_sum'].sum()

median_sum = pr.loc[indices_train].median()
df_train_all = df_train_all.merge(pr, right_index=True, left_index=True)
df_train_all.head()
median_sum
df_train_all['better_median_purchase'] = df_train_all.purchase_sum.apply(lambda x: 1 if (x > median_sum) else 0)
df_train_all.head()
df_train_all.loc[:, ['purchase_sum', 'better_median_purchase']].corr()
uplift_score(df_train_all[df_train_all.better_median_purchase == 1])
uplift_score(df_train_all[df_train_all.better_median_purchase == 0]) #снова сильно отличается
df_products.head()
owner_products = df_products[df_products['is_own_trademark']==1].product_id
with_owner = set(list(df_pursh[df_pursh['product_id'].isin(owner_products)]['client_id']))
df_train_all['with_owner'] = 0
df_train_all.loc[with_owner]['with_owner']  = 1
uplift_score(df_train_all[df_train_all.with_owner == 1])
uplift_score(df_train_all[df_train_all.with_owner == 0])  #те, кто не покупали, у них аплифт гораздо выше