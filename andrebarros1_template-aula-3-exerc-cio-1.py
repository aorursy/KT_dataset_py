import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# open csv data
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df_BA = df[df.uf == "BA"].nlargest(15, 'total_eleitores')[['gen_feminino','gen_masculino','gen_nao_informado']]
df_BA
plt.violinplot([df_BA['gen_feminino'],df_BA['gen_masculino']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default    
plt.xticks([1,2], ('gen_feminino','gen_masculino')) 
plt.show()
# filter_data
df_MG = df[df.uf == "MG"].nlargest(15, 'total_eleitores')[['f_18_20','f_21_24','f_70_79','f_sup_79']]
plt.violinplot([df_MG['f_18_20'],df_MG['f_21_24'],df_MG['f_70_79'],df_MG['f_sup_79']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default
plt.xticks([0,1,2,3,4], ('','f_18_20','f_21_24','f_70_79','f_sup_79'))

plt.show()
#maneira 1 sem nome no eixo
'''

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# plot violin plot
axes[0].violinplot([df_BA['gen_feminino'],df_BA['gen_masculino']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default
axes[0].set_title('Questão 1')
axes[1].violinplot([df_MG['f_18_20'],df_MG['f_21_24'],df_MG['f_70_79'],df_MG['f_sup_79']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2)
axes[1].set_title('Questão 2')

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    
plt.show()
'''
#Maneira correta

plt.figure(figsize=(15,6))
plt.subplot(121)
plt.violinplot([df_BA['gen_feminino'],df_BA['gen_masculino']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default
plt.xticks([1,2], ('gen_feminino','gen_masculino')) 

plt.subplot(122)
plt.violinplot([df_MG['f_18_20'],df_MG['f_21_24'],df_MG['f_70_79'],df_MG['f_sup_79']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2)
plt.xticks([0,1,2,3,4], ('','f_18_20','f_21_24','f_70_79','f_sup_79'))
plt.show()