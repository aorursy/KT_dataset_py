import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

df.head()
colunas = [['uf','Qualitativa Nominal'],

           ['total_eleitores','Quantitativa Discreta'],['f_16','Quantitativa Discreta'],

           ['f_sup_79','Quantitativa Discreta'],['f_18_20','Quantitativa Discreta'],

           ['f_70_79','Quantitativa Discreta'],['gen_feminino','Qualitativa Nominal'],

           ['gen_masculino','Qualitativa Nominal']]

df_final = pd.DataFrame(colunas, columns=["Variavel", "Classificação"])
df_final
df[df_final['Variavel']]
df["uf"].value_counts().plot(kind='bar')
fatias = [len(df["f_18_20"].value_counts()), len(df["f_70_79"].value_counts())]

labels = ['18 a 20 anos', '70 a 79 anos']

plt.title('Obrigados a votar')

cores  = ['r', 'm']

 

plt.pie(fatias, labels = labels, colors = cores, shadow = False)

 

plt.show()
fatias = [len(df["f_16"].value_counts()), len(df["f_sup_79"].value_counts())]

labels = ['16 anos', 'Acima de 79 anos']

plt.title('Não obrigados a votar')

cores  = ['r', 'm']

 

plt.pie(fatias, labels = labels, colors = cores, shadow = False)

 

plt.show()
x, y =["Feminino","Masculino"], [len(df["gen_feminino"].value_counts()),len(df["gen_masculino"].value_counts())]

plt.bar(x, y, color = ['b','g'])



plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)