import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


read = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv')
read.head(10)
Muni = read.loc[read['uf']=='BA'].nlargest(15,'total_eleitores')


plt.violinplot([Muni['gen_feminino'],Muni['gen_masculino']],showmeans=True, showmedians=True,widths = 2.0, bw_method=0.1) #default
plt.xticks([0,1,2], ('', 'gen_feminino','gen_masculino'))
plt.show()

Muni = read.loc[read['uf']=='MG'].nlargest(15,'total_eleitores')


plt.violinplot([Muni['f_18_20'],Muni['f_21_24'],Muni['f_70_79'],Muni['f_sup_79']],showmeans=True, showmedians=True,widths = 1.0, bw_method=0.1) #default
plt.xticks([0,1,2,3,4], ('', 'f_18_20','f_21_24','f_70_79','f_sup_79'))
plt.show()
plt.boxplot([Muni['f_18_20'],Muni['f_21_24'],Muni['f_70_79'],Muni['f_sup_79']],showmeans=True,widths = 0.1) #default
plt.xticks([0,1,2,3,4], ('', 'f_18_20','f_21_24','f_70_79','f_sup_79'))
plt.show()

plt.boxplot([Muni['gen_feminino'],Muni['gen_masculino']],showmeans=True,widths = 0.1) #default
plt.xticks([0,1,2], ('', 'gen_feminino','gen_masculino'))
plt.show()
