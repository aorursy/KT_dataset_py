import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)
import pandas as pd

import matplotlib.pyplot as plt

import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

%matplotlib inline
anv = pd.read_csv('../input/anv.csv', delimiter=',')

var = anv[['aeronave_operador_categoria','aeronave_tipo_veiculo','aeronave_fabricante','aeronave_modelo','aeronave_ano_fabricacao','aeronave_pais_fabricante','aeronave_fase_operacao','aeronave_nivel_dano','total_fatalidades']]

var.head()
cls_anv = [['aeronave_operador_categoria','Qualitativa Nominal'],['aeronave_tipo_veiculo','Qualitativa Nominal'],['aeronave_fabricante','Qualitativa Nominal'],['aeronave_modelo','Qualitativa Nominal'],['aeronave_ano_fabricacao','Quantitativa Discreta'],['aeronave_pais_fabricante','Qualitativa Nominal'],['aeronave_fase_operacao','Qualitativa Nominal'],['aeronave_nivel_dano','Qualitativa Nominal'],['total_fatalidades','Quantitativa Discreta']]

classificar = pd.DataFrame(cls_anv,columns=['Variável','Classificação'])

classificar
categoria = var['aeronave_operador_categoria'].value_counts()

tipo = var['aeronave_tipo_veiculo'].value_counts()

pais = var['aeronave_pais_fabricante'].value_counts()

modelo = var['aeronave_modelo'].value_counts()

fabricante = var['aeronave_fabricante'].value_counts()

fase = var['aeronave_fase_operacao'].value_counts()

dano = var['aeronave_nivel_dano'].value_counts()

tipo
ax = categoria.plot(kind='bar',rot=90, figsize=(10,7),color='coral',fontsize=13,label='Categoria')

ax.set_title('Porcentagem de acidentes por categoria\n', fontsize=18, color='black')

ax.set_ylabel('Número de Acidentes', fontsize=18)

ax.set_yticks([0,400,800,1200,1600,2000,2400])

ax.set_ylim((0,2500))

totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+25, \

          str(round((i.get_height()/total)*100, 2))+'%', fontsize=15,

             color='dimgrey')

plt.style.use('ggplot')

plt.legend()

plt.show()
princ = tipo[[1,2,3]]

outros = tipo[[4,5,6,7,8,9]]

ax = tipo.plot(kind='barh',rot=0, figsize=(10,7),color='dodgerblue',fontsize=13,label='Tipos')

ax.set_title('Porcentagem de acidentes por tipo de aeronave\n', fontsize=18, color='black')

ax.set_xlabel('Número de Acidentes', fontsize=18)

ax.set_xticks([0,500,1000,1500,2000,2500,3000,3500,4000,4500])

ax.set_xlim((0,5000))

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+.15,

          str(round((i.get_width()/total)*100, 2))+'%', fontsize=15,

             color='dimgrey')

plt.style.use('ggplot')

plt.legend()

plt.show()
ax = tipo.plot(kind='barh',rot=0, figsize=(10,7),color='dodgerblue',fontsize=13,label='Tipos')

#delete labelbottom

plt.tick_params(axis='x',which='both',bottom=False,top='False',labelbottom='False')

ax.set_title('Quantidade de acidentes por tipo de aeronave\n', fontsize=20, color='black')

ax.set_xlabel('Número de Acidentes', fontsize=18)

ax.set_xlim((0,4700))

totals = []

for i in ax.patches:

    totals.append(i.get_width())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_width()+15, i.get_y()+.15,i.get_width(),fontsize=15,color='black')

plt.style.use('ggplot')

plt.legend()

plt.show()

#b
danos = pd.DataFrame(dano).reset_index()

fat = danos['aeronave_nivel_dano']

lab = danos['index']

color = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#8c564b"]

explode = (0.1, 0, 0, 0, 0) 

plt.pie(fat,labels=lab,explode=explode,colors=color,autopct='%1.1f%%',shadow=True,startangle=140)

plt.title('Porcentagem de danos dos acidentes\n',fontsize=25,color='black')

#move legend to right

leg = plt.legend(loc='upper right')

bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

xoffset = 0.4

bb.x0 += xoffset

bb.x1 += xoffset

leg.set_bbox_to_anchor(bb, transform = ax.transAxes)

plt.show()
ax = pais.plot(kind='bar',rot=90, figsize=(10,7),color='g',fontsize=13,label='País')

ax.set_title('Quantidade de acidentes por país\n', fontsize=22, color='black')

ax.set_ylabel('Número de Acidentes', fontsize=20)

ax.set_yticks([0,400,800,1200,1600,2000,2400])

ax.set_ylim((0,2500))

totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+25,i.get_height(), color='black')

plt.style.use('ggplot')

plt.legend()

plt.show()
modelos = modelo.head(20)

ax = modelos.plot(kind='bar',rot=90, figsize=(10,7),color='brown',fontsize=13,label='Modelo')

#a.set_alpha(0.8)

ax.set_title('Os 20 modelos de aeronave que mais sofreram acidente\n', fontsize=20, color='black')

ax.set_ylabel('Número de Acidentes', fontsize=18, color='brown')

ax.set_xlabel('Modelos', fontsize=18, color='brown')

ax.set_yticks([0,50,100,150,200,250,300])

ax.set_ylim((0,250))

ax.set_facecolor('xkcd:grey')

totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+5,i.get_height(),color='black',fontsize=13)

plt.style.use('classic')

plt.legend()

plt.show()
ano = pd.DataFrame({'qtd':var.groupby(['aeronave_ano_fabricacao']).size()}).reset_index().drop([0])
def scatterplot(x_data, y_data, x_label, y_label, title):



    # Create the plot object

    _, ax = plt.subplots()



    # Plot the data, set the size (s), color and transparency (alpha)

    # of the points

    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    

    # Label the axes and provide a title

    ax.set_title(title,fontsize=20)

    ax.set_xlabel(x_label,fontsize=18)

    ax.set_ylabel(y_label,fontsize=18)

    ax.set_xticks([0,25,50,75,100,125,150,175,200,225])

    ax.set_xlim((-5,250))

plt.style.use('ggplot')



# Call the function to create plot

scatterplot(x_data = ano['qtd']

            , y_data = ano['aeronave_ano_fabricacao']

            , x_label = 'Número de Acidentes'

            , y_label = 'Ano de Fabricação'

            , title = 'Gráfico de Dispersão\n' + 'Ano de Fabricação VS Número de Acidentes\n')
fases = fase.head(10)

ax = fases.plot(kind='bar',rot=90, figsize=(10,7),color='b',fontsize=13,label='Fase de operação')

#a.set_alpha(0.8)

ax.set_title('As 10 fases de operação com maior número de acidentes\n', fontsize=20, color='black')

ax.set_ylabel('Número de Acidentes', fontsize=18, color='black')

ax.set_yticks([0,200,400,600,800,1000])

ax.set_ylim((0,1000))

ax.set_facecolor('xkcd:grey')

totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x(), i.get_height()+5,i.get_height(),color='black',fontsize=13)

plt.style.use('ggplot')

plt.legend()

plt.show()
fabr_fatal = var[(var.total_fatalidades>0)].aeronave_fabricante.value_counts().nlargest(21).drop(['***'])
ax = fabr_fatal.plot(kind='bar',rot=90, figsize=(15,10),color='g',fontsize=13,label='Fabricante')

ax.set_title('20 maiores fabricantes com acidentes fatais\n', fontsize=30, color='black')

ax.set_ylabel('Quantidade de acidentes com fatalidades', fontsize=20)

ax.set_yticks([0,10,20,30,40,50,60,70,80])

ax.set_ylim((0,80))

totals = []

for i in ax.patches:

    totals.append(i.get_height())

total = sum(totals)

for i in ax.patches:

    ax.text(i.get_x()+0.1, i.get_height()+1,i.get_height(), color='black',fontsize=13)

plt.style.use('ggplot')

plt.legend()

plt.show()