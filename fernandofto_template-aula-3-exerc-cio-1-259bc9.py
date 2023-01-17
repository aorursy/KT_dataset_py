import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.dataframeName = 'BR_eleitorado_2016_municipio.csv'
nRow, nCol = df.shape
print(f'There are {nRow} rows and {nCol} columns')
df
df_BA = df[['gen_masculino','gen_feminino','total_eleitores']][df.uf == 'BA'].nlargest(15,'total_eleitores')

# plot violin plot
plt.violinplot([df_BA['gen_masculino'],df_BA['gen_feminino']],showmeans=False,showmedians=True)

plt.show()
df_MG_ID = df[['f_18_20','f_21_24','f_70_79','f_sup_79','total_eleitores']][df.uf == 'MG'].nlargest(15,'total_eleitores')

plt.violinplot([df_MG_ID['f_18_20'],df_MG_ID['f_21_24'],df_MG_ID['f_70_79'],df_MG_ID['f_sup_79']],showmeans=False,showmedians=True)

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot box plot
axes[0].boxplot([df_BA['gen_masculino'],df_BA['gen_feminino']])
axes[0].set_title('Bahia')

axes[1].boxplot([df_MG_ID['f_18_20'],df_MG_ID['f_21_24'],df_MG_ID['f_70_79'],df_MG_ID['f_sup_79']])
axes[1].set_title('Minas Gerais')

plt.show()
