import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.dataframeName = 'BR_eleitorado_2016_municipio.csv'
df
dfBahia = df[df.uf == 'BA'].sort_values(by='total_eleitores', ascending=False).head(15)[['nome_municipio','gen_masculino','gen_feminino']]
dfBahia
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

axes[0].violinplot(dfBahia['gen_masculino'],showmeans=False,showmedians=True)
axes[0].set_title('Masculino por Cidade')

axes[1].violinplot(dfBahia['gen_feminino'],showmeans=False,showmedians=True)
axes[1].set_title('Feminino por Cidade')

for ax in axes:
    ax.yaxis.grid(True)

plt.show()
dfMinas = df[df.uf == 'MG'].sort_values(by='total_eleitores', ascending=False).head(15)[['nome_municipio','f_18_20','f_21_24','f_70_79','f_sup_79']]
dfMinas
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 10))

axes[0].violinplot(dfMinas['f_18_20'],showmeans=False,showmedians=True)
axes[0].set_title('Eleitorxs entre 18 e 20 anos')

axes[1].violinplot(dfMinas['f_21_24'],showmeans=False,showmedians=True)
axes[1].set_title('Eleitorxs entre 21 e 24 anos')

axes[2].violinplot(dfMinas['f_70_79'],showmeans=False,showmedians=True)
axes[2].set_title('Eleitorxs entre 70 e 79 anos')

axes[3].violinplot(dfMinas['f_sup_79'],showmeans=False,showmedians=True)
axes[3].set_title('Eleitorxs com mais de 79 anos')

for ax in axes:
    ax.yaxis.grid(True)

plt.show()
fig3, axes3 = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

axes3[0].boxplot([dfBahia.gen_feminino, dfBahia.gen_masculino])
axes3[0].set_title('Eleitorxs na Bahia')

axes3[1].boxplot([dfMinas.f_18_20, dfMinas.f_21_24, dfMinas.f_70_79, dfMinas.f_sup_79])
axes3[1].set_title('Eleitorxs em Minas Gerais')

for ax in axes3:
    ax.yaxis.grid(True)

plt.show()