import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/BR_eleitorado_2016_municipio.csv', delimiter=',')

df.head()
df = df.drop(['cod_municipio_tse', 'f_16', 'f_17', 'gen_nao_informado', 'gen_feminino', 'gen_masculino'], axis=1)
resposta_1a = [['nome_municipio','Qualitativa Nominal'], ['uf', 'Qualitativa Nominal'],['total_eleitores','Quantitativa Discreta'], ['f_18_20', 'Quantitativa Discreta'], ['f_21_24', 'Quantitativa Discreta'], ['f_25_34', 'Quantitativa Discreta'], ['f_35_44', 'Quantitativa Discreta'], ['f_45_59', 'Quantitativa Discreta'], ['f_60_69', 'Quantitativa Discreta'], ['f_70_79', 'Quantitativa Discreta'], ['f_sup_79', 'Quantitativa Discreta']]

resposta_1a = pd.DataFrame(resposta_1a, columns=["Variavel", "Classificação"])

resposta_1a
count = df['uf'].value_counts()

freq = count/sum(count)

resposta_1b = pd.DataFrame([count, freq], index=['Cidades', 'Frequência']).T

resposta_1b
import matplotlib.pyplot as plt



bins = df[['f_18_20', 'f_21_24', 'f_25_34', 'f_35_44', 'f_45_59', 'f_60_69', 'f_70_79', 'f_sup_79']]

x = bins.columns



fig, ax = plt.subplots()

plt.title("Distribuição dos eleitores por faixa etária")

plt.bar(x = x, height = bins.sum(), width=0.50, color='green')

plt.show()
somatorio = df.groupby(by='uf')

labels = somatorio.groups.keys()

values = somatorio.sum()['total_eleitores']



plt.figure(figsize=(20,6))

plt.title("Distribuição dos eleitores por estado")

plt.bar(x=labels, height=values)

plt.plot()
import numpy as np

sao_paulo = df.groupby(by='uf').get_group('SP')

data = sao_paulo.sum()[3:11]

y_pos = np.arange(len(data))

plt.figure(figsize=(20,6))

plt.title("Distribuição por faixa etária no estado com maior número de eleitores")

plt.barh(y_pos, data, align='center', color='red', tick_label=('18-20', '21-24', '25-34', '35-44', '45-59', '60-69', '60-79', '79-'))

plt.plot()