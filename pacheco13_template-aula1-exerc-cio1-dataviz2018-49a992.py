import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(5)

resposta = [["gen_feminino", "Quantitativa Discreta"],["gen_masculino","Quantitativa Discreta"],["gen_nao_informado","Quantitativa Discreta"], ["f_sup_79","Quantitativa Discreta"],["f_16","Quantitativa Discreta"],["f_17","Quantitativa Discreta"],["uf","Qualitativa Nominal"],["total_eleitores","Quantitativa Discreta"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
#Tabela de Frequência para a coluna uf
uf = df['uf'].value_counts()
freq = pd.concat([uf], axis=1, keys=['Frequência'])
freq
#Total de Eleitores Femininos e Masculinos
x = df['gen_feminino'].sum() 
y = df['gen_masculino'].sum()

# Data to plot
labels = 'Feminino', 'Masculino'
sizes = [x, y]
colors = ['pink','lightblue']
# Plot
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()

#Número de Eleitores por UF
df['uf'].value_counts().plot.bar()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)