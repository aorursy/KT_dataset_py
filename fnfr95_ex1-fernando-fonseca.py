import pandas as pd

df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(5)
tipo_variaveis = [['uf', 'Qualitativa Nominal'],
            ['f_16', 'Quantitativa Discreta'],
            ['f_17', 'Quantitativa Discreta'],
            ['f_21_24', 'Quantitativa Discreta'],
            ['f_25_34', 'Quantitativa Discreta'],
            ['f_70_79', 'Quantitativa Discreta'],
            ['f_sup_79', 'Quantitativa Discreta'],
            ['gen_feminino', 'Quantitativa Discreta'],
            ['gen_masculino', 'Quantitativa Discreta']]

tipo_variaveis = pd.DataFrame(tipo_variaveis, columns = ['Variavel', 'Classificação'])

display(tipo_variaveis)
display(df['uf'].value_counts())
import numpy as np
import matplotlib.pyplot as plt
ages = np.array(['f_16', 'f_17', 'f_21_24', 'f_25_34', 'f_70_79', 'f_sup_79'])

total_by_age = np.array([df[i].sum() for i in ages])

new_index = np.argsort(total_by_age)[::-1]
total_by_age = total_by_age[new_index]
label_ages = np.array(['16', '17', '21 - 24', '25 - 34', '70 - 79', '> 79'])
label_ages = label_ages[new_index]
# Create horizontal bars
plt.barh(label_ages, total_by_age, color=['green', 'yellow'])
plt.title('Eleitores por faixa etária')

# Create names on the y-axis
plt.yticks(label_ages)
 
# Show graphic
plt.show()
total_by_uf = df.groupby(['uf']).sum()['total_eleitores']
ufs = df.groupby(['uf']).sum().index

new_index_uf = np.argsort(total_by_uf)[::-1]
total_by_uf = total_by_uf[new_index_uf]
ufs = ufs[new_index_uf]
plt.figure(figsize=(20,5))

plt.bar(ufs, total_by_uf,   label = 'Quantidade', color = ['green', 'yellow'])
plt.title('Quantidade de eleitores por unidade da federação Brasileira')
plt.xticks(rotation='vertical')

plt.legend()

plt.show()
total_man = df['gen_masculino'].sum()
total_women = df['gen_feminino'].sum()
plt.pie(np.array([total_man, total_women]), labels = ['Homens', 'Mulheres'], colors = ['green', 'yellow'], shadow = True)
plt.title('Total de votos por gênero')
plt.show()
grouped_uf = df.groupby(['uf']).sum()
display(grouped_uf)
labels_gen = []

for i in ufs:
    labels_gen.append(i + '_MASCULINO')
    labels_gen.append(i + '_FEMININO')
    
labels_gen = np.array(labels_gen)
masculino_uf = grouped_uf['gen_masculino'][new_index_uf]
feminino_uf = grouped_uf['gen_feminino'][new_index_uf]
gridsize = (2, 1) # 4 rows, 2 columns
fig = plt.figure(figsize=(20, 10)) # this creates a figure without axes
ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=1, rowspan=1)
ax2 = plt.subplot2grid(gridsize, (1, 0), colspan=1, rowspan=2)

ax1.set_title('Eleitores por unidade da federação')
ax1.bar(ufs, masculino_uf, width=0.5, label = 'Masculino', color = 'green')
ax2.set_title('Eleitoras por unidade da federação')
ax2.bar(ufs, feminino_uf,  width=0.5, label = 'Feminino', color = 'yellow')