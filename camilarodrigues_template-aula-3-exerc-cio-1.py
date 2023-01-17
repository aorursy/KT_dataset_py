import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(10)
estado = df[(df['uf'] == 'BA')]
x = estado.nlargest(15, 'total_eleitores')

plt.violinplot([x['total_eleitores'],x['gen_feminino'], x['gen_masculino']])
plt.xticks([1,2,3], ('Total', 'Masculino','Feminino'))
plt.show()
estado = df[(df['uf'] == 'MG')]
x = estado.nlargest(15, 'total_eleitores')
plt.violinplot([x['total_eleitores'], x['f_18_20'],x['f_21_24'], x['f_70_79'], x['f_sup_79']])
plt.xticks([1, 2, 3, 4, 5], ('Total', 'f_18_20','f_21_24', 'f_70_79', 'f_sup_79'))
plt.show()
