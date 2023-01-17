import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head()
Top15Bahia = df[(df['uf'] == 'BA')].nlargest(15, 'total_eleitores')
#Top15Bahia

plt.violinplot([Top15Bahia['gen_feminino'], Top15Bahia['gen_masculino']],showmeans=True, showmedians=True)
plt.xticks([1,2], ('Feminino','Masculino'))
plt.suptitle('Gênero dos eleitores das Top 15 Cidades da Bahia em eleitorado')
plt.show()
Top15Minas = df[(df['uf'] == 'MG')].nlargest(15, 'total_eleitores')
#Top15Minas

plt.violinplot([Top15Minas['f_18_20'],Top15Minas['f_21_24'], Top15Minas['f_70_79'], Top15Minas['f_sup_79']],showmeans=True, showmedians=True)
plt.xticks([1,2,3,4], ('18 até 20 anos', '21 até 24 anos','70 até 79 anos', 'Maior de 79 anos'))
plt.suptitle('Comparativo idade dos eleitores das Top 15 Cidades de Minas Gerais em eleitorado')
plt.show()
plt.boxplot([Top15Bahia['gen_feminino'], Top15Bahia['gen_masculino']],showmeans=True)
plt.xticks([1,2], ('Feminino','Masculino'))
plt.suptitle('Gênero dos eleitores das Top 15 Cidades da Bahia em eleitorado')
plt.show()

plt.boxplot([Top15Minas['f_18_20'],Top15Minas["f_21_24"], Top15Minas["f_70_79"], Top15Minas["f_sup_79"]],showmeans=True)
plt.xticks([1,2,3,4], ('18 até 20 anos', '21 até 24 anos','70 até 79 anos', 'Maior de 79 anos'))
plt.suptitle('Comparativo idade dos eleitores das Top 15 Cidades de Minas Gerais em eleitorado')
plt.show()