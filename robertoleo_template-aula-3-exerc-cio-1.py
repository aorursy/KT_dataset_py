import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head()
df1 = df[df['uf']=='BA']
df1 = df1.nlargest(columns='total_eleitores',n=15)

plt.violinplot([df1['gen_feminino'],df1["gen_masculino"]],showmeans=True, showmedians=True,widths = 0.5, bw_method=0.5) #default
plt.xticks([0,1,2], ('', 'Feminino','Masculino'))
plt.show()
df2 = df[df['uf']=='MG']
df2 = df2.nlargest(columns='total_eleitores',n=15)
plt.violinplot([df2['f_18_20'],df2["f_21_24"],df2['f_70_79'],df2['f_sup_79']],showmeans=True, showmedians=True,widths = 0.5, bw_method=0.5) #default
plt.xticks([0,1,2,3,4], ('', '18-20','21-24','70-79','>79'))
plt.show()
plt.boxplot([df1['gen_feminino'],df1["gen_masculino"]],labels=['Feminino','Masculino']);
plt.grid(axis='y');
plt.show()
plt.boxplot([df2['f_18_20'],df2["f_21_24"],df2['f_70_79'],df2['f_sup_79']],labels=['18-20','21-24','70-79','>79']);
plt.grid(axis='y');
plt.show()
