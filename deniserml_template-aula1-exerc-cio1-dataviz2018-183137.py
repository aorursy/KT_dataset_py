#import pandas as pd
#resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
#resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
#resposta
# Importing Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Importing file
df = pd.read_csv('../input/anv.csv', delimiter=',')
#df.head(10)
#Droping columns that I'm not going to use
df.drop(columns= ['codigo_ocorrencia','aeronave_matricula','aeronave_modelo','aeronave_tipo_icao','aeronave_motor_tipo','aeronave_pmd','aeronave_assentos','aeronave_pais_fabricante','aeronave_pais_registro','aeronave_registro_categoria','aeronave_registro_segmento','aeronave_voo_origem','aeronave_voo_destino','aeronave_fase_operacao','aeronave_fase_operacao_icao','aeronave_tipo_operacao','aeronave_dia_extracao'],axis = 1, inplace = True)
#Replacing  manufacturing year missing values with 0
df['aeronave_ano_fabricacao'] = df['aeronave_ano_fabricacao'].replace(np.nan,0)
#Changing manufacturing year datatype
df['aeronave_ano_fabricacao'] = df['aeronave_ano_fabricacao'].astype('int64')
#Changing *** values por "NAO INFORMADO"
df['aeronave_fabricante'] = df['aeronave_fabricante'].replace('***','NAO INFORMADO')
df['aeronave_motor_quantidade'] = df['aeronave_motor_quantidade'].replace('***','NAO INFORMADO')
df['aeronave_pmd_categoria'] = df['aeronave_pmd_categoria'].replace('***','NAO INFORMADO')

df.tail(10)
#Creating dataframe
resposta = [["aeronave_operador_categoria", "Qualitativa Nominal"],["aeronave_tipo_veiculo", "Qualitativa Nominal"],["aeronave_motor_quantidade", "Qualitativa Ordinal"],["aeronave_pmd_categoria", "Qualitativa Ordinal"],["aeronave_ano_fabricacao", "Quantitativa Continua"],["aeronave_nivel_dano", "Quantitativa Ordinal"],["aeronave_fabricante", "Qualitativa Nominal"]]
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df['aeronave_operador_categoria'].value_counts()
df['aeronave_tipo_veiculo'].value_counts()
df['aeronave_pmd_categoria'].value_counts()
df['aeronave_motor_quantidade'].value_counts()
#Bar Chart Variable
y_axis = df['aeronave_operador_categoria'].value_counts()
x_axis = df['aeronave_operador_categoria'].unique()
bar_color = 'grey'
width = 0.80 

for i, v in enumerate(y_axis):
 plt.text(v, i, " "+str(v), color='black', va='center')
    
plt.barh(x_axis, y_axis, width, color=bar_color , label = "Categoria")
plt.xticks(rotation='vertical')
plt.title('OCORRÊNCIAS POR CATEGORIA DO OPERADOR')
plt.show()
#aeronave_tipo_veiculo
#Pie Chart Variable
fatias = df.groupby(['aeronave_tipo_veiculo'])['total_fatalidades'].agg('sum').sort_values(ascending=False).head(3)
label = ['AVIÃO','HELICÓPTERO','ULTRALEVE']
explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(fatias, explode=explode, labels = label, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title('FATALIDADES POR TIPO DE VEÍCULO')
ax1.axis('equal') 
plt.show()
#aeronave_motor_quantidade
#Bar Chart Variable

y_axis2 = df.groupby(['aeronave_motor_quantidade'])['total_fatalidades'].agg('sum').sort_values(ascending=False)
x_axis2 = df['aeronave_motor_quantidade'].unique()

y_axis = df['aeronave_motor_quantidade'].value_counts()
x_axis = df['aeronave_motor_quantidade'].unique()
   
plt.bar(x_axis, y_axis,  color='grey', label = "Acidentes")
plt.bar(x_axis2, y_axis2, color='red',label = "Fatalidades")

plt.xticks(rotation='45')
plt.title('OCORRENCIAS X FATALIDADES POR TIPO DE MOTOR')
plt.legend()
plt.show()
#aeronave_pmd_categoria
y_axis = df['aeronave_pmd_categoria'].value_counts()
x_axis = df['aeronave_pmd_categoria'].unique()

for a,b in zip(x_axis, y_axis):
    plt.text(a, b, str(b), color='black', ha='center', va = 'bottom')
  
plt.bar(x_axis, y_axis,  width=0.5, color='green')

#plt.xticks(rotation='vertical')
plt.title('OCORRÊNCIAS POR CATEGORIA DE MOTOR')
plt.show()
#aeronave_ano_fabricacao
newdf = df[df['aeronave_ano_fabricacao'] != 0 ]
newdf['CONTAGEM'] = 1

newdf.groupby('aeronave_ano_fabricacao')['CONTAGEM'].sum().plot(color = 'purple',figsize=(12,5),grid = True)

plt.xlabel('ANO FABRICAÇÃO')
plt.ylabel('OCORRÊNCIAS')
plt.show()
#aeronave_nivel_dano
y_axis = df['aeronave_nivel_dano'].value_counts()
x_axis = df['aeronave_nivel_dano'].unique()

plt.bar(x_axis, y_axis,  color='orange', label = "Categoria")

plt.xticks(rotation='45')
plt.title('OCORRÊNCIA POR NÍVEL DE DANO')
plt.legend()
plt.show()
#aeronave_fabricante
data_fab = newdf.groupby('aeronave_fabricante')['CONTAGEM'].sum().sort_values(ascending = False)
data_fab.head(10).plot(kind = 'barh', color='purple',figsize=(10,5),grid = True)

plt.title('OCORRÊNCIA POR FABRICANTE (OS 10 +)')
plt.xlabel('FABRICANTE')
plt.ylabel('OCORRÊNCIAS')
plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)