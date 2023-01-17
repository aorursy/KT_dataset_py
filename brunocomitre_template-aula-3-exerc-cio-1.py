# Realizando Importações
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Lendo Dataset
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(1)
uf_BA = df[(df['uf'] == 'BA')]
BA_15 = uf_BA.nlargest(15, 'total_eleitores')
plt.violinplot([BA_15['total_eleitores'],BA_15['gen_feminino'], BA_15['gen_masculino']])
plt.xticks([1,2,3], ('Total', 'Masculino','Feminino'))
plt.show()
uf_MG = df[(df['uf'] == 'MG')]
MG_I = uf_MG.nlargest(15, 'total_eleitores')
plt.violinplot([MG_I['total_eleitores'], MG_I['f_18_20'],MG_I['f_21_24'], MG_I['f_70_79'], MG_I['f_sup_79']])
plt.xticks([1, 2, 3, 4, 5], ('Total', 'f_18_20','f_21_24', 'f_70_79', 'f_sup_79'))
plt.show()
plt.boxplot([BA_15['gen_feminino'],BA_15["gen_masculino"]],labels=['Feminino','Masculino']);
plt.grid(axis='y');
plt.show()
plt.boxplot([MG_I['f_18_20'],MG_I["f_21_24"],MG_I['f_70_79'],MG_I['f_sup_79']],labels=['18-20','21-24','70-79','>79']);
plt.grid(axis='y');
plt.show()