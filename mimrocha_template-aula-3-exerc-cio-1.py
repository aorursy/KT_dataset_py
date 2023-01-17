import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head()
state_BA =  df[df['uf']=='BA'].sort_values(by='total_eleitores', ascending=False).head(15)
state_BA

plt.violinplot([state_BA['gen_masculino'],state_BA["gen_feminino"]],showmeans=True, showmedians=True,widths = 0.1, bw_method=0.1) #default
plt.xticks([0,1,2], ('', 'gen_masculino','gen_feminino'))
plt.title('Eleitores Masculinos e Femininos das 15 maiores cidade do estado da Bahia')
plt.show()
state_MG =  df[df['uf']=='MG'].sort_values(by='total_eleitores', ascending=False).head(15)
state_MG
plt.violinplot([state_MG['f_18_20'],state_MG["f_21_24"],state_MG["f_70_79"],state_MG["f_sup_79"]],showmeans=True, showmedians=True,widths = 0.1, bw_method=0.1) #default
plt.xticks([0,1,2,3,4], ('', '18-20','21-24','70-79','> 79'))
plt.title('Faixas de idade dos eleitores das 15 maiores cidade do estado de Minas Gerais')
plt.show()
plt.boxplot([state_BA['gen_masculino'],state_BA["gen_feminino"]],showmeans=True,widths = 0.1) #default
plt.xticks([0,1,2], ('', 'gen_masculino','gen_feminino'))
plt.title('Eleitores Masculinos e Femininos das 15 maiores cidade do estado da Bahia')
plt.show()
plt.boxplot([state_MG['f_18_20'],state_MG["f_21_24"],state_MG["f_70_79"],state_MG["f_sup_79"]],showmeans=True,widths = 0.1) #default
plt.xticks([0,1,2,3,4], ('', '18-20','21-24','70-79','> 79'))
plt.title('Faixas de idade dos eleitores das 15 maiores cidade do estado de Minas Gerais')
plt.show()