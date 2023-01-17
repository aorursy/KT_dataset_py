import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_exe1 = pd.read_csv("../input/BR_eleitorado_2016_municipio.csv")
df_exe1.head(3)
df1 = df_exe1[df_exe1['uf'] == 'BA'].sort_values(by='total_eleitores', ascending=False).head(15)
plt.violinplot([df1['gen_masculino'],df1['gen_feminino']],showmeans=True, showmedians=True,widths = 0.1, bw_method=0.1) #default
plt.xticks([0,1,2], ('', 'Masculino','Feminino'))

plt.suptitle('Eleitores por Sexo - 15 Maiores Cidades Bahia', fontsize=14)
plt.show()
df2 = df_exe1[df_exe1['uf'] == 'MG'].sort_values(by='total_eleitores', ascending=False).head(15)

plt.violinplot([df2['f_18_20'],df2['f_21_24'], df2['f_70_79'], df2['f_sup_79']],showmeans=True, showmedians=True,widths = 0.1, bw_method=0.1) #default
plt.xticks([0,1,2,3,4], ('', '18 a 20','21 a 24','70 a 79', '79+'))

plt.suptitle('Eleitores por Idade - 15 Maiores Cidades Minas Gerais', fontsize=14)
plt.show()
plt.boxplot([df2['f_18_20'],df2["f_21_24"], df2["f_70_79"], df2["f_sup_79"]],showmeans=True, widths = 0.1) #default
plt.xticks([0,1,2,3,4], ('', '18 a 20','21 a 24','70 a 79', '79+'))
plt.suptitle('Eleitores por Idade - 15 Maiores Cidades Minas Gerais', fontsize=14)
plt.show()

plt.boxplot([df1['gen_masculino'],df1["gen_feminino"]],showmeans=True, widths = 0.1) #default
plt.xticks([0,1,2], ('', 'Masculino','Feminino'))

plt.suptitle('Eleitores por Sexo - 15 Maiores Cidades Bahia', fontsize=14)
plt.show()
