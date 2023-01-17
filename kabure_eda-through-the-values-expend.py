import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

# figure size in inches

rcParams['figure.figsize'] = 9,6
df_dep = pd.read_csv("../input/deputies_dataset.csv")

df_dep2 = pd.read_csv("../input/dirty_deputies_v2.csv")
df_dep.head()
print("Unique values: ")

print(df_dep.nunique())



print("Description")

print(df_dep.describe())
print("Mean values receipt: ")

print(df_dep["receipt_value"].mean())

print("Max values receipt: ")

print(df_dep["receipt_value"].max())

print("Min values receipt: ")

print(df_dep["receipt_value"].min())

print("Std values receipt: ")

print(df_dep["receipt_value"].std())

print("Median values receipt: ")

print(df_dep["receipt_value"].median())

print("Count values receipt: ")

print(df_dep["receipt_value"].count())

print("Sum values receipt: ")

print(df_dep["receipt_value"].sum())
contagem_reemb = df_dep.groupby(["political_party"])["receipt_value"].sum()

print("Número de reembolsos registrados: ")

print(contagem_reemb.sort_values(ascending=False)[:30])



sns.factorplot(x="political_party", y="receipt_value",

               data = df_dep[df_dep["receipt_value"] < 5000],

               kind="violin",

               size=7,aspect= 2)

plt.show()
#I will start looking the frequency of registers by deputy and how the top 30 deputy's 

# by their political party

contagem_reemb = df_dep.groupby(["political_party","deputy_name"])["receipt_value"].count()

print("Número de reembolsos registrados: ")

print(contagem_reemb.sort_values(ascending=False)[:30])



contagem_reemb.sort_values(ascending=False)[:30].plot(kind="bar")
ax=sns.factorplot(x="political_party", y="receipt_value",

               data = df_dep[df_dep["receipt_value"] < 5000],

               kind="box",

               size=7,aspect= 2)

ax.set_xticklabels(rotation=30)

plt.show()
#Now I will show the sum of 30 highest total values

values_reemb = df_dep.groupby(["political_party","deputy_name"])["receipt_value"].sum()

print("Número de reembolsos registrados: ")

print(values_reemb.sort_values(ascending=False)[:30])



sns.factorplot(x="deputy_name", y="receipt_value",data=df_dep)
#values by descriptions 

received = df_dep.groupby(['receipt_description'])["receipt_value"].sum()



print(received.sort_values(ascending=False)[:30])





g = sns.factorplot(x='receipt_description', y="receipt_value", 

                   data=df_dep[df_dep["receipt_value"] <=100000],

                   size=8,aspect=2, kind="box")

g.set_xticklabels(rotation=45)

plt.show()

#values by descriptions combined with political party and deputy names

description_received = df_dep.groupby(['receipt_description','political_party','deputy_name'])["receipt_value"].sum()



print(description_received.sort_values(ascending=False)[:30])



description_received
#values by partys



partys_receipt = df_dep.groupby(['political_party'])["receipt_value"].sum()



print(partys_receipt.sort_values(ascending=False))



partys_receipt.sort_values(ascending=False).plot(kind='bar')

plt.show()
#Values by states

state_receipt = df_dep.groupby(['state_code'])["receipt_value"].sum()



print(state_receipt.sort_values(ascending=False))



state_receipt.sort_values(ascending=False).plot(kind='bar')

plt.show()
#Looking the distribuition by establisment name



stab_receipt = df_dep.groupby(['establishment_name'])["receipt_value"].sum()

print(stab_receipt.sort_values(ascending=False)[:30])



stab_receipt.sort_values(ascending=False)[:30].plot(kind='bar')

plt.show()
contagem_reemb = df_dep.groupby(["political_party","deputy_name"])["receipt_value"].sum()

print("Número de reembolsos registrados: ")



print(contagem_reemb.sort_values(ascending=False)[:30])



contagem_reemb.sort_values(ascending=False)[:30].plot(kind='bar')

plt.show()
refund_partys = df_dep2.groupby(['party_pg'])["refund_value"].mean()



print(refund_partys.sort_values(ascending=False)[:30])



refund_partys.plot(kind="bar")

plt.show()



sns.factorplot(x='party_pg',y='refund_value',data=df_dep2,size=10,kind="count")
#Refunds by ideology's

part_ide_refund = df_dep2.groupby(["party_pg","party_ideology1","party_ideology2"])["refund_value"].mean()



print(part_ide_refund.sort_values(ascending=False)[:30])



part_ide_refund.sort_values(ascending=False)[:30].plot(kind="bar")

plt.show()