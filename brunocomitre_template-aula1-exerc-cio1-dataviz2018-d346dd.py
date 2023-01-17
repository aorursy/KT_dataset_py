# Realizando as importações e definindo o corte das casas decimais, e outros valores:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.2%}'.format
pd.set_option('display.max_row', 800)# set jupyter's max row display
pd.set_option('display.max_columns', 50)# set jupyter's max column width to 50

plt.style.use('ggplot')
%matplotlib inline
# Checkando as 5 primeiras linhas do arquivo:
dataset = pd.read_csv('../input/anv.csv', delimiter=',')
dataset.head()
# Analisando as dimensões do dataset:
print('Este DataSet possui linhas:', dataset.shape[0])
print('Este DataSet possui colunas:', dataset.shape[1])
# Verificando se existem dados nulos:
dataset.isnull().sum()
dataset.drop(['codigo_ocorrencia','aeronave_tipo_icao','aeronave_pmd','aeronave_pmd_categoria','aeronave_assentos',
              'aeronave_ano_fabricacao','aeronave_pais_fabricante','aeronave_registro_categoria','aeronave_registro_segmento',
              'aeronave_voo_origem','aeronave_voo_destino','aeronave_fase_operacao','aeronave_fase_operacao_icao',
              'aeronave_tipo_operacao','aeronave_nivel_dano','aeronave_dia_extracao'],axis=1, inplace = True)
dataset.head()
# Verificando quantidade de dados nulos:
dataset.isnull().sum()
classification = [['aeronave_fabricante' , 'Qualitativa Nominal'],['aeronave_matricula' , 'Qualitativa Nominal'],
            ['aeronave_operador_categoria' , 'Qualitativa Ordinal'],['aeronave_tipo_veiculo' , 'Qualitativa Nominal'],
            ['aeronave_modelo' , 'Qualitativa Ordinal'],['aeronave_motor_tipo' , 'Qualitativa Nominal'],
            ['aeronave_motor_quantidade' , 'Qualitativa Ordinal'],['aeronave_pais_registro' , 'Qualitativa Nominal'],
            ['total_fatalidades' , 'Quantitativa Discreta']]
classification = pd.DataFrame(classification, columns=['Variavel' , 'Classificação'])
classification
manufacturer = dataset['aeronave_fabricante'].value_counts()
p_manufacturer = dataset['aeronave_fabricante'].value_counts(normalize=True)
gf1 = pd.concat([manufacturer, p_manufacturer], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf1.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
registration = dataset['aeronave_matricula'].value_counts()
p_registration = dataset['aeronave_matricula'].value_counts(normalize=True)
gf2 = pd.concat([registration, p_registration], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf2.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
operator_category = dataset['aeronave_operador_categoria'].value_counts()
p_operator_category = dataset['aeronave_operador_categoria'].value_counts(normalize=True)
gf3 = pd.concat([operator_category, p_operator_category], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf3.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
vehicle = dataset['aeronave_tipo_veiculo'].value_counts()
p_vehicle = dataset['aeronave_tipo_veiculo'].value_counts(normalize=True)
gf4 = pd.concat([vehicle, p_vehicle], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf4.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
model = dataset['aeronave_modelo'].value_counts()
p_model = dataset['aeronave_modelo'].value_counts(normalize=True)
gf5 = pd.concat([model, p_model], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf5.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
engine_type = dataset['aeronave_motor_tipo'].value_counts()
p_engine_type = dataset['aeronave_motor_tipo'].value_counts(normalize=True)
gf6 = pd.concat([engine_type, p_engine_type], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf6
engine_quantity = dataset['aeronave_motor_quantidade'].value_counts()
p_engine_quantity = dataset['aeronave_motor_quantidade'].value_counts(normalize=True)
gf7 = pd.concat([engine_quantity, p_engine_quantity], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf7
registration_country = dataset['aeronave_pais_registro'].value_counts()
p_registration_country = dataset['aeronave_pais_registro'].value_counts(normalize=True)
gf8 = pd.concat([registration_country, p_registration_country], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf8.head()
#.head() para mostrar os 5 primeiros valores, caso deseje ver a lista completa delete-o (utilizado para facilitar vizualização)
fatalities = dataset['total_fatalidades'].value_counts()
p_fatalities = dataset['total_fatalidades'].value_counts(normalize=True)
gf9 = pd.concat([fatalities, p_fatalities], axis=1, keys=['Frequência Absoluta', 'Frequência Relativa %'])
gf9
ax = manufacturer.plot(kind='barh', figsize=(10,200),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem de cada empresa em relação a quantidade de aeronaves no arquivo de estudo", fontsize=16)
ax.set_xlabel("Quantidade de Aeronaves por Empresas", fontsize=12);
ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = registration .plot(kind='barh', figsize=(10,900),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem de combinação de Marcas atribuidas por matrícula das Aeronaves", fontsize=16)
ax.set_xlabel("Número de Aeronaves", fontsize=12);
ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = operator_category.plot(kind='barh', figsize=(10,7),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Categoria de uma aeronave separada por uso", fontsize=16)
ax.set_xlabel("Quantidade de Aeronaves", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = vehicle.plot(kind='barh', figsize=(10,7),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem referente a Categoria das Aeronaves", fontsize=16)
ax.set_xlabel("Numero de Aeronaves", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = model.plot(kind='barh', figsize=(10,700),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem de modelos diferenes para a mesma categoria", fontsize=16)
ax.set_xlabel("Quantidade", fontsize=12);
ax.set_xticks([0, 50, 100, 150, 200, 250])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = engine_type.plot(kind='barh', figsize=(10,7),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem de Aeronaves por tipo de motor", fontsize=16)
ax.set_xlabel("Quantidade de Aeronaves", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = engine_quantity.plot(kind='barh', figsize=(10,7),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Modelo de motor por Aeronave", fontsize=16)
ax.set_xlabel("Quantidade", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = registration_country.plot(kind='barh', figsize=(10,20),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Porcentagem de Naves registradas por País", fontsize=16)
ax.set_xlabel("Quantidade", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
ax = fatalities.plot(kind='barh', figsize=(10,20),color="coral", fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Fatalidades", fontsize=16)
ax.set_xlabel("Quantidade", fontsize=12);
ax.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])

totals = []# cria uma lista para coletar os dados de plt.patches

for i in ax.patches:# encontrar os valores e acrescentar à lista
    totals.append(i.get_width())

total = sum(totals)# definir barras individuais usando a lista acima

for i in ax.patches:# get_width puxa para a esquerda ou para a direita; get_y empurra para cima ou para baixo
    ax.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=13,
color='dimgrey')

ax.invert_yaxis()# invertido para maior no topo
af = dataset[dataset.aeronave_tipo_veiculo != 'INDETERMINADA'].groupby(['aeronave_tipo_veiculo'])['total_fatalidades'].sum()

ax = af.plot(kind='bar',figsize=(15,7), color=['dodgerblue', 'slategray'], fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Analise da quantidade de acidentes fatais por tipo de aeronave",
fontsize=18)
ax.set_ylabel("Numero de Vitimas", fontsize=18);
ax.set_yticks([0, 200, 400, 600, 800, 1000])
ax.set_xticklabels(["ANFÍBIO","AVIÃO","BALÃO","DIRIGÍVEL","HELICOPTERO",
                    "HIDROAVIÃO","PLANADOR","TRIKE","ULTRALEVE"], rotation=0, fontsize=11)

for i in ax.patches:# definir barras individuais usando a lista acima 
    ax.text(i.get_x()+.04, i.get_height()+130, \
            str(round((i.get_height()), 2)), fontsize=11, color='dimgrey',
                rotation=45)


ag = dataset[dataset.aeronave_motor_quantidade != '***'].groupby(['aeronave_motor_quantidade'])['aeronave_motor_quantidade'].value_counts()
ax = ag.plot(kind='bar',figsize=(15,7), color=['dodgerblue', 'slategray'], fontsize=13);

ax.set_alpha(0.8)
ax.set_title("Analise para identificar os principais tipos de motores utilizados nas aeronaves",
fontsize=18)
ax.set_ylabel("Quantidade", fontsize=18);
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax.set_xticklabels(["BIMOTOR","MONOMOTOR","QUADRIMOTOR","SEM TRAÇÃO","TRIMOTOR"], rotation=0, fontsize=11)

for i in ax.patches:# definir barras individuais usando a lista acima 
    ax.text(i.get_x()+.04, i.get_height()+130, \
            str(round((i.get_height()), 2)), fontsize=11, color='dimgrey',
                rotation=45)
ar = dataset[dataset.aeronave_motor_tipo != '***'].groupby(['aeronave_motor_tipo'])['aeronave_motor_tipo'].value_counts()
ax = ar.plot(kind='bar',figsize=(15,7), color=['dodgerblue', 'slategray'], fontsize=13);

ax.set_alpha(0.8)
ax.set_title("ACIDENTES POR TIPO DE AERONAVE",
fontsize=18)
ax.set_ylabel("Quantidade", fontsize=18);
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
ax.set_xticklabels(["JATO","PISTÃO","SEM TRAÇÃO","TURBOELICE","TURBOEIXO"], rotation=0, fontsize=11)

for i in ax.patches:# definir barras individuais usando a lista acima 
    ax.text(i.get_x()+.04, i.get_height()+130, \
            str(round((i.get_height()), 2)), fontsize=11, color='dimgrey',
                rotation=45)

pd.show_versions ()
gs1 = engine_type.plot.bar(title='ACIDENTES POR TIPO DE AERONAVE')
gs1.set_xlabel('Tipo de Aeronave')
gs1.set_ylabel('Quantidade Acidentes')
gs1.plot()
dataset.groupby(['aeronave_motor_quantidade','aeronave_tipo_veiculo'])['aeronave_tipo_veiculo'].size().unstack().plot(kind='bar',
                                                                                                                    stacked=True)
plt.show()
y_axis = p_vehicle
x_axis = range(len(y_axis))
width_n = 0.5
bar_color = 'yellow'

plt.bar(x_axis, y_axis, width=width_n, color=bar_color)
plt.show()
test = dataset['aeronave_pais_registro'].value_counts()
test.plot.hist(x='Tipo de Aeronave', y='Quantidade Acidentes')
plt.plot(p_engine_type)# Criando um gráfico
 
plt.title('Exemplo utilizando Plot')# Atribuindo um título ao gráfico
plt.xlabel('Variavel 1')
plt.ylabel('Variavel 2')
 
plt.plot(p_engine_type, label = 'Uma legenda')# Atribuindo uma legenda
plt.legend()

plt.show()# Exibindo o gráfico gerado
plt.plot(p_engine_type)
plt.title("Muito Fácil")
plt.show()
dataset.groupby(['aeronave_motor_tipo','aeronave_motor_quantidade'])['aeronave_modelo'].size().unstack().plot(kind='bar',
                                                                                                              stacked=True)
plt.show()
p_model[:20].plot(kind='barh')
test = dataset['aeronave_pais_registro'].value_counts(normalize=True)
test.plot.line(x='Tipo de Aeronave', y='Quantidade Acidentes')
test = dataset['aeronave_pais_registro'].value_counts()
test.plot.pie(x='Tipo de Aeronave', y='Quantidade Acidentes')
model.plot.bar()