import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head()
ba = df[df['uf']=='BA'].nlargest(columns='total_eleitores',n=15)
ba


plt.violinplot([ba['gen_masculino'],ba["gen_feminino"]],showmeans=True, showmedians=True,widths = 0.5, bw_method=0.5) #default
plt.xticks([0,1,2], ('', 'Masculino','Feminino'))

plt.show()
mg = df[df['uf']=='MG'].nlargest(columns='total_eleitores',n=15)
mg
plt.violinplot([mg['f_18_20'],mg["f_21_24"],mg["f_70_79"],mg["f_sup_79"]],showmeans=True, showmedians=True,widths = 0.5, bw_method=0.5) #default
plt.xticks([0,1,2,3,4], ('', '18 a 20','21 a 24','70 a 79','superior a 79'))

plt.show()
plt.boxplot([mg['f_18_20'],mg["f_21_24"],mg["f_70_79"],mg["f_sup_79"]],labels=['18 a 20','21 a 24','70 a 79','superior a 79']);
plt.show()
plt.boxplot([ba['gen_masculino'],ba["gen_feminino"]],labels=['Masculino','Feminino']);
plt.show()
