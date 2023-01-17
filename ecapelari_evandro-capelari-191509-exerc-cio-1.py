import pandas as pd

df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')
df.head(5)
import pandas as pd


variaveis = [["aeronave_motor_quantidade", "Qualitativa Nominal", 'Quantidade Motores'],
            ["aeronave_pmd_categoria","Qualitativa Ordinal", 'Peso da Aeronave'],
            ["aeronave_pais_fabricante","Qualitativa Nominal", 'País Fábricante'],
            ["aeronave_fase_operacao","Qualitativa Nominal", 'Fase de Operação'],
            ["aeronave_tipo_operacao","Qualitativa Nominal", 'Tipo da Operação'],
            ["aeronave_nivel_dano","Qualitativa Ordinal", 'Nível Dano'],
            ["total_fatalidades","Quantitativa Discreta", 'Total Fatalidades']]
variaveis = pd.DataFrame(variaveis, columns=["Variavel", "Classificação", 'Nome_Apresentacao_Variavel'])
variaveis
for index, row in variaveis.iterrows():
    if('Qualitativa' in row['Classificação']):
        print(row['Nome_Apresentacao_Variavel'] + '\n')
        print(df[row['Variavel']].value_counts())
        print('\n')
#     print(df['aeronave_motor_quantidade'].value_counts())

import matplotlib.pyplot as plt
import numpy as np


def show_me_graph(index):
    row = variaveis.iloc[index, : ]
    dfna = df[row['Variavel']].dropna()
    
    plt.figure(figsize=(20,10))
    
    plt.title(row['Nome_Apresentacao_Variavel'])
    if 'Qualitativa' in row['Classificação']:
        idf = dfna.value_counts()
        
        if 'Ordinal' in row['Classificação'] and len(idf.tolist()) < 8:
            plt.pie(idf.tolist(), labels=idf.keys(), autopct='%1.1f%%',
                    shadow=True, startangle=90)
            plt.axis('equal')
        else:
            plt.ylabel('nº de Acidentes')
            plt.bar(idf.keys(),idf.tolist())
        if len(idf.tolist()) > 8:
            plt.xticks(rotation=45)
    else:
        sorted_data = np.sort(dfna)
        if 'fatalidades' in row['Variavel']: 
            plt.step(np.arange(sorted_data.size),sorted_data)  # From 0 to the number of data points-1
        else:
            if 'assentos' in row['Variavel']:
                newDf = dfna[dfna > 0]
                plt.scatter(np.arange(len(newDf)), newDf)
                plt.xlabel('nº de Acidentes')
                plt.ylabel('nº de Assentos')
            else:
                plt.plot(dfna)
    print(plt.show())


# Código Original removido pois a visualização de todos os gráficos em uma só imagem não estava bom
# fig, axs = plt.subplots(nrows=len(variaveis['Variavel']), ncols=1, figsize=(20, 5 * variaveis.size))
# for index, row in variaveis.iterrows():
#     dfna = df[row['Variavel']].dropna()
    
#     axs[index].title.set_text(row['Nome_Apresentacao_Variavel'])
#     if 'Qualitativa' in row['Classificação']:
#         idf = dfna.value_counts()
        
#         if 'Ordinal' in row['Classificação'] and len(idf.tolist()) < 8:
#             axs[index].pie(idf.tolist(), labels=idf.keys(), autopct='%1.1f%%',
#                     shadow=True, startangle=90)
#             axs[index].axis('equal')
#         else:
#             axs[index].set_ylabel('nº de Acidentes')
#             axs[index].bar(idf.keys(),idf.tolist())
#         if len(idf.tolist()) > 8:
#             axs[index].tick_params(labelrotation=45)
#     else:
#         if 'assentos'  in row['Variavel']: 
#             sorted_data = np.sort(dfna)  # Or data.sort(), if data can be modified
        
#             axs[index].step(sorted_data, np.arange(sorted_data.size))  # From 0 to the number of data points-1
#         else:
#             axs[index].plot(dfna)

# fig.suptitle('Fatores medidos para análise relativos a acidentes aéreos')
# print(plt.show())

show_me_graph(0)

show_me_graph(1)
show_me_graph(2)
show_me_graph(3)
show_me_graph(4)
show_me_graph(5)
show_me_graph(6)