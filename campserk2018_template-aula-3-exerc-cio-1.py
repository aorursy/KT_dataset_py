import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv',)
df_BA = df[df.uf == "BA"].nlargest(15, 'total_eleitores')
df_BA
plt.violinplot([df_BA['gen_feminino'],df_BA['gen_masculino']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default    
plt.xticks([1,2], ('gen_feminino','gen_masculino')) 
plt.show()
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv',)
df_MG = df[df.uf == "MG"].nlargest(15, 'total_eleitores')
df_MG

plt.figure(figsize=(15,6))
plt.subplot(121)
plt.violinplot([df_BA['gen_feminino'],df_BA['gen_masculino']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2) #default
plt.xticks([1,2], ('gen_feminino','gen_masculino')) 


plt.subplot(122)
plt.violinplot([df_MG['f_18_20'],df_MG['f_21_24'],df_MG['f_70_79'],df_MG['f_sup_79']],showmeans=True, showmedians=True,widths = 0.9, bw_method=0.2)
plt.xticks([0,1,2,3,4], ('','f_18_20','f_21_24','f_70_79','f_sup_79'))
plt.show()
df_SP = df[['gen_masculino','gen_feminino','total_eleitores']][df.uf == 'SP'].nlargest(15,'total_eleitores')
df_RJ = df[['f_18_20','f_21_24','f_70_79','f_sup_79','total_eleitores']][df.uf == 'RJ'].nlargest(15,'total_eleitores')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot box plot
axes[0].boxplot([df_SP['gen_masculino'],df_SP['gen_feminino']])
axes[0].set_title('São PAulo')

axes[1].boxplot([df_RJ['f_18_20'],df_RJ['f_21_24'],df_RJ['f_70_79'],df_RJ['f_sup_79']])
axes[1].set_title('Rio de Janeiro')

plt.show()
