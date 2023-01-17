import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv')
df_ba = df[df.uf == 'BA'].sort_values(by='total_eleitores', ascending = False).head(15)
plt.violinplot([df_ba.gen_feminino, df_ba.gen_masculino],showmeans=False, showmedians=True) #default
plt.xticks([0, 1, 2], ('', 'Feminino', 'Masculino'))
plt.show()
df_mg = df[df.uf == 'MG'].sort_values(by='total_eleitores', ascending = False).head(15)
plt.violinplot([df_mg.f_18_20, df_mg.f_21_24, df_mg.f_70_79, df_mg.f_sup_79],showmeans=False, showmedians=True) #default
plt.xticks([0, 1, 2, 3, 4], ('', '18 a 20', '21 a 24', '70 a 79', '79+'))
plt.show()
plt.boxplot([df_ba.gen_feminino, df_ba.gen_masculino])
plt.xticks([0, 1, 2], ('', 'Feminino', 'Masculino'))
plt.show()

plt.boxplot([df_mg.f_18_20, df_mg.f_21_24, df_mg.f_70_79, df_mg.f_sup_79])
plt.xticks([0, 1, 2, 3, 4], ('', '18 a 20', '21 a 24', '70 a 79', '79+'))
plt.show()