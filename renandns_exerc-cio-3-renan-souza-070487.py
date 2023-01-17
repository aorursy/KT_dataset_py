%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

blackfriday = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

blackfriday.head()
pd.DataFrame(blackfriday["Age"].value_counts())
# Just switch x and y

#sns.violinplot( y=blackfriday["Purchase"], x=blackfriday["Age"] )

sns.violinplot( x=blackfriday["Purchase"], y=blackfriday["Age"] )

#sns.plt.show()
# Gráfico gerado com a área dos violinos do mesmo tamanho

sns.set(style="whitegrid")

plt.figure(figsize=(16, 6))

g1 = sns.violinplot( y=blackfriday["Purchase"], x=blackfriday["Age"].sort_values(ascending=True))
# Gráfico gerado com o parametro scale = "count" para que a área do violino demontre a quantidade de registros presentes naquele grupo de dados

plt.figure(figsize=(16, 6))

g1 = sns.violinplot( y=blackfriday["Purchase"], x=blackfriday["Age"].sort_values(ascending=True) , scale="count")

pd.DataFrame(blackfriday["Age"].value_counts())
pd.DataFrame(blackfriday["Product_ID"].value_counts())
products = blackfriday["Product_ID"].value_counts().head(9)

products

plt.figure(figsize=(16, 6))

for i, v in products.iteritems():

    plt.bar(i, v, label = i)

    plt.text(i, v, v, va='bottom', ha='center')    

    

plt.title('Produtos mais comprados')

plt.show()
top_5_occupation = blackfriday['Occupation'].value_counts().head(5)



bf_by_top_5_occupation = pd.DataFrame

for i, v in top_5_occupation.iteritems():    

    if bf_by_top_5_occupation.empty :        

        bf_by_top_5_occupation = blackfriday[blackfriday['Occupation'] == i]

    else:

        bf_by_top_5_occupation = bf_by_top_5_occupation.append(blackfriday[blackfriday['Occupation'] == i])
bf_by_top_5_occupation

 

plt.figure(figsize=(20, 10))

sns.boxenplot(x=bf_by_top_5_occupation['Occupation'], y=bf_by_top_5_occupation['Purchase'], hue=bf_by_top_5_occupation['Age'])



# use the function regplot to make a scatterplot

#sns.regplot(x=bf_by_top_5_occupation["Age"], y=bf_by_top_5_occupation["Purchase"])

#sns.plt.show()



#plt.figure(figsize=(16, 6))

#g1 = sns.violinplot( y=bf_by_top_5_occupation["Purchase"], x=bf_by_top_5_occupation["Age"].sort_values(ascending=True),scale="count")

#sns.jointplot(x="Purchase", y="Age", data=bf_by_top_5_occupation);

 

# Without regression fit:

#sns.regplot(x=bf_by_top_5_occupation["Age"], y=bf_by_top_5_occupation["Purchase"], fit_reg=False)

#sns.plt.show()
bf_purchase_9000 = blackfriday[blackfriday['Purchase'] > 9000]

#sns.regplot(x=bf_purchase_9000["Occupation"], y=bf_purchase_9000["Marital_Status"])

#sns.regplot(x=bf_purchase_9000["Occupation"], y=bf_purchase_9000["Marital_Status"], fit_reg=False)

# multiple line plot

#plt.plot( 'Purchase', 'Occupation', data=bf_purchase_9000, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

#plt.plot( 'Purchase', 'Occupation', data=bf_purchase_9000, marker='', color='olive', linewidth=2)

#plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")

#plt.legend()

#plt.show()

#sns.jointplot(x="Marital_Status", y="Purchase", data=bf_purchase_9000);

plt.figure(figsize=(20, 10))

sns.boxenplot(x=bf_purchase_9000['Occupation'], y=bf_purchase_9000['Purchase'], hue=bf_purchase_9000['Marital_Status'])

#sns.boxenplot(x=bf_purchase_9000['Marital_Status'], y=bf_purchase_9000['Purchase'], hue=bf_purchase_9000['Occupation'])