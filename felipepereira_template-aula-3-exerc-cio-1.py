import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv')
df.head(5)
dfaux = df.where(df['uf']=='BA').dropna().nlargest(15, 'total_eleitores')
dfaux2 = df.where(df['uf']=='BA').dropna().nlargest(15, 'total_eleitores')[-12:]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].violinplot([dfaux['total_eleitores'], dfaux['gen_masculino'], dfaux['gen_feminino']])
axes[1].violinplot([dfaux2['total_eleitores'], dfaux2['gen_masculino'], dfaux2['gen_feminino']])
labels = ['', 'Total', '', 'Masc', '', 'Fem']
axes[0].set_title('15 maiores cidades')
axes[0].set_xticklabels(labels)
axes[1].set_title('15 maiores cidades menos as 3 primeiras')
axes[1].set_xticklabels(labels)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].violinplot([dfaux['f_18_20'], dfaux['f_21_24'],dfaux['f_70_79'],dfaux['f_sup_79']])
axes[1].violinplot([dfaux2['f_18_20'], dfaux2['f_21_24'],dfaux2['f_70_79'],dfaux2['f_sup_79']])
labels = ['', '18-20', '', '21-24', '', '70-79', '', '79+']
axes[0].set_title('15 maiores cidades')
axes[0].set_xticklabels(labels)
axes[1].set_title('15 maiores cidades menos as 3 primeiras')
axes[1].set_xticklabels(labels)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].boxplot([dfaux['total_eleitores'], dfaux['gen_masculino'], dfaux['gen_feminino']])
axes[1].boxplot([dfaux2['total_eleitores'], dfaux2['gen_masculino'], dfaux2['gen_feminino']])
labels = ['Total', 'Masc', 'Fem']
axes[0].set_title('15 maiores cidades')
axes[0].set_xticklabels(labels)
axes[1].set_title('15 maiores cidades menos as 3 primeiras')
axes[1].set_xticklabels(labels)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].boxplot([dfaux['f_18_20'], dfaux['f_21_24'],dfaux['f_70_79'],dfaux['f_sup_79']])
axes[1].boxplot([dfaux2['f_18_20'], dfaux2['f_21_24'],dfaux2['f_70_79'],dfaux2['f_sup_79']])
labels = ['18-20', '21-24', '70-79', '79+']
axes[0].set_title('15 maiores cidades')
axes[0].set_xticklabels(labels)
axes[1].set_title('15 maiores cidades menos as 3 primeiras')
axes[1].set_xticklabels(labels)
plt.show()
