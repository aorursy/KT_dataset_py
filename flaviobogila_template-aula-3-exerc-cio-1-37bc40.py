from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(5)
df_sexo = df[['total_eleitores','gen_masculino','gen_feminino']][df.uf == 'BA'].nlargest(15, 'total_eleitores')
# plot violin plot
plt.violinplot([df_sexo['gen_masculino'],df_sexo['gen_feminino']],showmeans=False,showmedians=True)
plt.xticks([0,1,2], ('','Masculino', 'Feminino'))

plt.show()
df_idade = df[['total_eleitores','f_18_20','f_21_24','f_70_79','f_sup_79']][df.uf == 'MG'].nlargest(15,'total_eleitores')

# plot violin plot
plt.violinplot([df_idade['f_18_20'],df_idade['f_21_24'],df_idade['f_70_79'],df_idade['f_sup_79']],showmeans=False,showmedians=True)
plt.xticks([0,1,2,3,4,5], ('','18 a 20', '21 a 24','70 a 79','sup a 79'))

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot violin plot
axes[0].boxplot([df_sexo['gen_masculino'],df_sexo['gen_feminino']])
axes[0].set_title('violin plot')

axes[1].boxplot([df_idade['f_18_20'],df_idade['f_21_24'],df_idade['f_70_79'],df_idade['f_sup_79']])
axes[1].set_title('violin plot')

plt.show()