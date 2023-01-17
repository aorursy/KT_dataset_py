#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Importing File
file_path = '../input/'
file_name = 'BR_eleitorado_2016_municipio.csv'
df = pd.read_csv(file_path + file_name,delimiter = ',')

#Reading Dataset
df.head(10)
#Sorting by BA state
mask = df['uf'] == 'BA'
by_BA = df[mask]
#Picking up the 15 largest voters
by_BA = by_BA.nlargest(15,'total_eleitores').sort_values(by='total_eleitores')
#by_BA
# plot violin plot
all_data = [by_BA['gen_feminino'],by_BA['gen_masculino']]

plt.violinplot(all_data)
plt.title('Nº ELEITORES MASCULINO x FEMININO')
plt.xticks([1,2],('Feminino','Masculino'))
plt.show()
#Sorting by MG state
mask1 = df['uf'] == 'MG'
by_MG = df[mask]
#Picking up the 15 largest voters
by_MG = by_MG.nlargest(15,'total_eleitores').sort_values(by='total_eleitores')
#by_MG
# plot violin plot
all_data = [by_MG['f_18_20'],by_MG['f_21_24'],by_MG['f_70_79'],by_MG['f_sup_79']]

plt.violinplot(all_data)
plt.title('Nº ELEITORES ESTADO MG - POR IDADE')
plt.xticks([1,2,3,4],('18-20','21-24','70-79','> 79'))
plt.show()
all_data = [by_BA['gen_feminino'],by_BA['gen_masculino']]
green_diamond = dict(markerfacecolor='g', marker='D')

plt.boxplot(all_data,flierprops=green_diamond)
plt.title('Nº ELEITORES MASCULINO x FEMININO')
plt.xticks([1,2],('Feminino','Masculino'))
plt.show()
all_data = [by_MG['f_18_20'],by_MG['f_21_24'],by_MG['f_70_79'],by_MG['f_sup_79']]
green_diamond = dict(markerfacecolor='g', marker='D')

plt.boxplot(all_data,flierprops=green_diamond)
plt.title('Nº ELEITORES ESTADO MG - POR IDADE')
plt.xticks([1,2,3,4],('18-20','21-24','70-79','> 79'))
plt.show()