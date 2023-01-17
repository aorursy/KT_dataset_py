#Primeiro, importamos as bibliotecas relevantes para a análise.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Abrimos o arquivo csv dos gastos públicos com educação, em formato de DataFrame do Pandas. Depois 
#abriremos os com os resultados do PISA.

csv_spend = pd.read_csv('../input/public_spend.csv')

#Antes de manipular os dados, precisamos organizar a DataFrame, pois ela vem bastante bagunçada.
#Faço um .groupby para juntar todos os países, que estão sob a coluna LOCATION, e faço a média
#com os dados disponíveis (de 2014 a 2016; alguns países têm dados dos 3 anos, enquanto outros, só de 1).
#Além disso, excluo as linhas sem valor (.dropna()) só por precaução, renomeio as colunas para 
#algo mais compreensível e defino os países como índice da tabela.

mean_spend = csv_spend.groupby('LOCATION')['Value'].mean().reset_index()
mean_spend = mean_spend.dropna()
mean_spend.columns = ['País','Gasto %']
mean_spend = mean_spend.set_index('País')
mean_spend = mean_spend.sort_values('Gasto %')
mean_spend.head()
#Pronto, com isso, a DataFrame do gasto público em educação está organizada.
mean_spend.describe()
#Com a função .describe() percebemos que a média de investimento em educação é de ~3.28% do PIB,
#com um mínimo de ~1.9% e um máximo de ~4.8%, dentre outras estatísticas.
#Agora, com os dados organizados, podemos usar a biblioteca seaborn para plotar um gráfico 
#dos países e seus gastos públicos em educação 

plt.figure(figsize=(16,8))
plot_mean_spend = sns.barplot(x = mean_spend.index,y = mean_spend['Gasto %'],palette='Blues')
plt.setp(plot_mean_spend.get_xticklabels(),rotation=70)
plt.show()

#Observamos que o Brasíl é o 7o país que mais investe em Educação.
#Para analisar, agora, os dados do PISA, fazemos parecido com o que foi feito para o .csv dos gasto públicos:
#Importar o csv em formato de DataFrame, selecionar os dados dos anos para os quais temos as informações
#dos gastos com educação, ordenar por resultado no PISA, etc (basicamente, organizar a tabela).

#Matemática
csv_pisa_math = pd.read_csv('../input/pisa1.csv')
pisa_math_2015 = csv_pisa_math[csv_pisa_math['TIME']>2013]

pisa_math_final = pisa_math_2015.groupby('LOCATION')['Value'].mean().reset_index()
pisa_math_final = pisa_math_final.sort_values('Value')
pisa_math_final.columns = ['País','Pisa Math']
pisa_math_final.head()

#Ciência
csv_pisa_sci = pd.read_csv('../input/pisa2.csv')
pisa_sci_2015 = csv_pisa_sci[csv_pisa_sci['TIME']>2013]

pisa_sci_final = pisa_sci_2015.groupby('LOCATION')['Value'].mean().reset_index()
pisa_sci_final = pisa_sci_final.sort_values('Value')
pisa_sci_final.columns = ['País','Pisa Science']
pisa_sci_final.head()

#Leitura
csv_pisa_read = pd.read_csv('../input/pisa3.csv')
pisa_read_2015 = csv_pisa_read[csv_pisa_read['TIME']>2013]

pisa_read_final = pisa_read_2015.groupby('LOCATION')['Value'].mean().reset_index()
pisa_read_final = pisa_read_final.sort_values('Value')
pisa_read_final.columns = ['País','Pisa Read']
pisa_read_final.head()
#Plotando os gráficos dos melhores desempenhos em cada categoria:

plt.figure(figsize=(16,8))
pisa_math_final_plot = sns.barplot(x = pisa_math_final['País'],y=pisa_math_final['Pisa Math'],palette='Blues')
plt.setp(pisa_math_final_plot.get_xticklabels(),rotation=70)
plt.show()
#Brasil em último

plt.figure(figsize=(16,8))
pisa_sci_final_plot = sns.barplot(x = pisa_sci_final['País'],y=pisa_sci_final['Pisa Science'],palette='Blues')
plt.setp(pisa_sci_final_plot.get_xticklabels(),rotation=70)
plt.show()
#Brasil em penúltimo

plt.figure(figsize=(16,8))
pisa_read_final_plot = sns.barplot(x = pisa_read_final['País'],y=pisa_read_final['Pisa Read'],palette='Blues')
plt.setp(pisa_read_final_plot.get_xticklabels(),rotation=70)
plt.show()
#Brasil em antipenúltimo

#Juntando os resultados em uma única DataFrame:
pisa = pd.concat([pisa_read_final,pisa_math_final,pisa_sci_final],sort=False)
pisa = pisa.groupby('País').mean()
pisa
#Agora, é interessante visualizarmos um gráfico de correlação entre os dados, como um scatterplot, nesse
#caso com regressão.

sns.jointplot(x=pisa['Pisa Math'],y=pisa['Pisa Science'],kind='reg')
sns.jointplot(x=pisa['Pisa Math'],y=pisa['Pisa Read'],kind='reg')
sns.jointplot(x=pisa['Pisa Read'],y=pisa['Pisa Science'],kind='reg')
#O que se percebe é que, de modo geral, quando um país tem um bom desempenho em uma das matérias, 
#ele tem nas outras também.
#Com base nessa informação, decidi fazer uma média das três matérias, o que acredito que não implique
#perda de informação, já que, de modo geral, um país bom em ciência é bom em matemática e leitura, e vice
#versa.

pisa['Pisa Mean']=(pisa['Pisa Read']+pisa['Pisa Math']+pisa['Pisa Science'])/3
pisa_e_gasto = pisa.join(mean_spend)
pisa_e_gasto
#Com isso, por último, podemos finalmente responder se um país que investe em educação tem bons desempenhos
#no PISA, plotando um gráfico de correlação do tipo scatter.

sns.jointplot(kind='scatter',x=pisa_e_gasto['Gasto %'],y=pisa_e_gasto['Pisa Mean'])
#A conclusão é que não há relação. Na realidade, existem países com pouquíssimo investimento e muito
#retorno, e vice-versa. 