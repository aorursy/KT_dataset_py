import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(5)

estado_ba = df.loc[df['uf']=='BA'].nlargest(15,'total_eleitores')
estado_ba.head(15)

plt.violinplot([estado_ba['gen_masculino'],estado_ba["gen_feminino"]],showmeans=True, showmedians=True,widths = 2.0, bw_method=0.1) #default
plt.xticks([0,1,2], ('', 'gen_masculino','gen_feminino'))

plt.show()
estado_mg = df.loc[df['uf']=='MG'].nlargest(15,'total_eleitores')
estado_mg.head(15)

plt.violinplot([estado_mg['f_18_20'],estado_mg["f_21_24"],estado_mg["f_70_79"],estado_mg["f_sup_79"]],showmeans=True, showmedians=True,widths = 1.0, bw_method=0.1) #default
plt.xticks([0,1,2,3,4], ('', 'f_18_20','f_21_24','f_70_79','f_sup_79'))

plt.show()
df_ba = pd.DataFrame(estado_ba[['gen_masculino','gen_feminino']])
df_ba.plot.box()

df_mg = pd.DataFrame(estado_mg[['f_18_20','f_21_24','f_70_79','f_sup_79']])
df_mg.plot.box()