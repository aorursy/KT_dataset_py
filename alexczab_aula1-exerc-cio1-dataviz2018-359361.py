import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df.head(1)
resposta = [["uf","Qualitativa Nominal"],
            ["nome_municipio","Qualitativa Nominal"],
            ["gen_feminino","Quantitativa Discreta"],
            ["gen_masculino","Quantitativa Discreta"],
            ["f_16","Quantitativa Discreta"],
            ["f_17","Quantitativa Discreta"],
            ["f_18_20","Quantitativa Discreta"],
            ["f_21_24","Quantitativa Discreta"],
            ["f_25_34","Quantitativa Discreta"],
            ["f_35_44","Quantitativa Discreta"],
            ["f_45_59","Quantitativa Discreta"],
            ["f_60_69","Quantitativa Discreta"],
            ["f_70_79","Quantitativa Discreta"],
            ["f_sup_79","Quantitativa Discreta"]]
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df.dtypes
df_uf = df["uf"].value_counts()
df_uf
df_nome_municipio = df["nome_municipio"].value_counts()
df_nome_municipio
#Dataframe por região
df_norte = df[(df['uf'] == 'AM') | 
                (df['uf'] == 'RR') | 
                (df['uf'] == 'AP') | 
                (df['uf'] == 'PA') | 
                (df['uf'] == 'TO') |
                (df['uf'] == 'RO') | 
                (df['uf'] == 'AC')]

df_nordeste = df[(df['uf'] == 'MA') | 
                (df['uf'] == 'PI') | 
                (df['uf'] == 'CE') | 
                (df['uf'] == 'RN') | 
                (df['uf'] == 'PE') |
                (df['uf'] == 'PB') | 
                (df['uf'] == 'SE') |
                (df['uf'] == 'AL') |
                (df['uf'] == 'BA') ]

df_centro_oeste = df[(df['uf'] == 'MT') | 
                (df['uf'] == 'MS') | 
                (df['uf'] == 'GO') |
                (df['uf'] == 'DF')]

df_sudeste = df[(df['uf'] == 'SP') | 
                (df['uf'] == 'RJ') | 
                (df['uf'] == 'ES') | 
                (df['uf'] == 'MG')]

df_sul = df[(df['uf'] == 'PR') | 
                (df['uf'] == 'RS') | 
                (df['uf'] == 'SC')]

#Total de eleitores
total_eleitores_geral = df.groupby(['uf'])['total_eleitores'].apply(lambda x : x.astype(int).sum()).sum()

#por genero total eleitores
tot_feminino = df.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino = df.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_eleitores = pd.DataFrame({'porcentagem': [round(tot_feminino / total_eleitores_geral * 100,2),
                                           round(tot_masculino / total_eleitores_geral * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])

#por genero por região Norte
tot_norte = df_norte['total_eleitores'].sum()
tot_feminino_norte = df_norte.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino_norte = df_norte.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_norte = pd.DataFrame({'porcentagem': [round(tot_feminino_norte / tot_norte * 100,2),
                                           round(tot_masculino_norte / tot_norte * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])

#por genero por região Nordeste
tot_nordeste = df_nordeste['total_eleitores'].sum()
tot_feminino_nordeste = df_nordeste.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino_nordeste = df_nordeste.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_nordeste = pd.DataFrame({'porcentagem': [round(tot_feminino_nordeste / tot_nordeste * 100,2),
                                           round(tot_masculino_nordeste / tot_nordeste * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])

#por genero por região centro-oeste
tot_centro_oeste = df_nordeste['total_eleitores'].sum()
tot_feminino_centro_oeste = df_centro_oeste.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino_centro_oeste = df_centro_oeste.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_centro_oeste = pd.DataFrame({'porcentagem': [round(tot_feminino_centro_oeste / tot_centro_oeste * 100,2),
                                           round(tot_masculino_centro_oeste / tot_centro_oeste * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])

#por genero por região sudeste
tot_sudeste = df_nordeste['total_eleitores'].sum()
tot_feminino_sudeste = df_sudeste.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino_sudeste = df_sudeste.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_sudeste = pd.DataFrame({'porcentagem': [round(tot_feminino_sudeste / tot_sudeste * 100,2),
                                           round(tot_masculino_sudeste / tot_sudeste * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])

#por genero por região sul
tot_sul = df_nordeste['total_eleitores'].sum()
tot_feminino_sul = df_sul.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).sum()
tot_masculino_sul = df_sul.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).sum()
df_genero_sul = pd.DataFrame({'porcentagem': [round(tot_feminino_sul / tot_sul * 100,2),
                                           round(tot_masculino_sul / tot_sul * 100,2)]},
                          index=['gen_feminino', 'gen_masculino'])
# Criação da figure com uma linha e duas colunas. Figsize define o tamanho da
# figure
fig, eixos = plt.subplots(nrows=2, ncols=3, figsize=(18,9))

# Cria o gráfico de pizza na primeira posição com as configurações definidas
pie_1 = eixos[0][0].pie(df_genero_eleitores, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

pie_2 = eixos[0][1].pie(df_genero_norte, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

pie_3 = eixos[0][2].pie(df_genero_nordeste, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

pie_4 = eixos[1][0].pie(df_genero_centro_oeste, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

pie_5 = eixos[1][1].pie(df_genero_sudeste, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

pie_6 = eixos[1][2].pie(df_genero_sul, labels=['Feminino','Masculino'],
                    autopct='%1.1f%%', colors=['pink', 'lightskyblue'])

# Define o título deste gráfico
eixos[0][0].set_title('Percentual por Genero no Brasil')
eixos[0][1].set_title('Percentual por Genero na Região Norte')
eixos[0][2].set_title('Percentual por Genero na Região Nordeste')
eixos[1][0].set_title('Percentual por Genero na Região Centro-Oeste')
eixos[1][1].set_title('Percentual por Genero na Região Sudeste')
eixos[1][2].set_title('Percentual por Genero na Região Sul')

# Deixa os dois eixos iguais, fazendo com que o gráfico mantenha-se redondo
eixos[0][0].axis('equal')
eixos[0][1].axis('equal')
eixos[0][2].axis('equal')
eixos[1][0].axis('equal')
eixos[1][1].axis('equal')
eixos[1][2].axis('equal')


plt.axis('equal')

# Ajusta o espaço entre os dois gráficos
plt.subplots_adjust(wspace=1)
plt.show()
#Tabela Frequencia de eleitores por estado
df_total_eleitores_estado = df.groupby(['uf'])['total_eleitores'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_eleitores')

#Váriavel do Total de Eleitores
total_eleitores_geral = df_total_eleitores_estado['total_eleitores'].sum()

#Tabela de frequencia por genero feminino por estado
df_gen_feminino = df.groupby(['uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_feminino')
df_gen_masculino = df.groupby(['uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_masculino')
df_gen_feminino['%_feminino'] = round(df_gen_feminino['total_feminino'] / df_total_eleitores_estado['total_eleitores'] * 100,2)
df_gen_masculino['%_masculino'] = round(df_gen_masculino['total_masculino'] / df_total_eleitores_estado['total_eleitores'] * 100,2)

#Largura da barra no gráfico
width = 0.50

#Count dos grupos
N = df_gen_masculino['uf'].count()

#Indices
ind = np.arange(N)

#Valores
colx = df_gen_feminino.sort_values('%_feminino', ascending=False)['uf']
val1 = df_gen_feminino.sort_values('%_feminino', ascending=False)['%_feminino']
val2 = df_gen_masculino.sort_values('%_masculino', ascending=True)['%_masculino']


#estilo
plt.style.use('ggplot')

#padronizando o tamanho dos gráficos
plt.rcParams['figure.figsize'] = (16,7)

p1 = plt.bar(ind, val1, width, color='purple')
p2 = plt.bar(ind, val2, width, color='lightskyblue')

plt.xlabel('Estados (UF)')
plt.ylabel('% porcentagem')
plt.title('Percentual de eleitores por generos nos Estado (UF)')
plt.xticks(ind, colx)
plt.yticks(np.arange(0, 61, 10))
plt.legend((p1[0], p2[0]), ('Feminino','Masculino'))

plt.show()

df_gen_feminino.sort_values('%_feminino', ascending=False)
#Tabela Frequencia de eleitores por estado
df_total_eleitores_municipio = df.groupby(['cod_municipio_tse','nome_municipio','uf'])['total_eleitores'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_eleitores')

#Total de eleitores feminino nos Municipios
df_gen_feminino = df.groupby(['cod_municipio_tse','nome_municipio','uf'])['gen_feminino'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_feminino')

#Adiciona na dataframe a porcentagem de eleitores feminino
df_gen_feminino['%_feminino'] = round(df_gen_feminino['total_feminino'] / df_total_eleitores_municipio['total_eleitores'] * 100,2)

#Adiciona na dataframe a media de eleitores feminino por Municipios
df_gen_feminino['%_media'] = round(df_gen_feminino['%_feminino'].mean(),2)

#Ordenação por valores
sort_gen_feminino = df_gen_feminino.sort_values(by='%_feminino', ascending=False)

#top 20
top20 = sort_gen_feminino.head(20).sort_values(by='%_feminino', ascending=True)

#Largura da barra no gráfico
width = 0.50

#Count dos grupos
N = top20['uf'].count()

#Indices
ind = np.arange(N)

#Valores
colx = top20['nome_municipio'] + ' - ' + top20['uf']
val1 = top20['%_feminino']
val2 = top20['%_media']

p1 = plt.barh(ind, val1, width, color='purple')
p2 = plt.barh(ind, val2, width, color='silver')

plt.xlabel('% porcentagem')
plt.ylabel('Municípios')
plt.title('Top 20 dos municípios que tiveram o maior porcentual de eleitores femininos')
plt.yticks(ind, colx)
plt.xticks(np.arange(0, 71, 10))
plt.legend((p1[0], p2[0]), ('Resultado','Média'))

#estilo
plt.style.use('ggplot')

#padronizando o tamanho dos gráficos
plt.rcParams['figure.figsize'] = (16,7)

plt.show()

top20
#Tabela Frequencia de eleitores por estado
df_total_eleitores_municipio = df.groupby(['cod_municipio_tse','nome_municipio','uf'])['total_eleitores'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_eleitores')

#Total de eleitores feminino nos Municipios
df_gen_masculino = df.groupby(['cod_municipio_tse','nome_municipio','uf'])['gen_masculino'].apply(lambda x : x.astype(int).sum()).reset_index(name='total_masculino')

#Adiciona na dataframe a porcentagem de eleitores feminino
df_gen_masculino['%_masculino'] = round(df_gen_masculino['total_masculino'] / df_total_eleitores_municipio['total_eleitores'] * 100,2)

#Adiciona na dataframe a media de eleitores feminino por Municipios
df_gen_masculino['%_media'] = round(df_gen_masculino['%_masculino'].mean(),2)

#Ordenação por valores
sort_gen_masculino = df_gen_masculino.sort_values(by='%_masculino', ascending=False)

#top 20
top20 = sort_gen_masculino.head(20).sort_values(by='%_masculino', ascending=True)

#Largura da barra no gráfico
width = 0.50

#Count dos grupos
N = top20['uf'].count()

#Indices
ind = np.arange(N)

#Valores
colx = top20['nome_municipio'] + ' - ' + top20['uf']
val1 = top20['%_masculino']
val2 = top20['%_media']

p1 = plt.barh(ind, val1, width, color='blue')
p2 = plt.barh(ind, val2, width, color='silver')

plt.xlabel('% porcentagem')
plt.ylabel('Municípios')
plt.title('Top 20 dos municípios que tiveram o maior porcentual de eleitores masculino')
plt.yticks(ind, colx)
plt.xticks(np.arange(0, 71, 10))
plt.legend((p1[0], p2[0]), ('Resultado','Média'))

#estilo
plt.style.use('ggplot')

#padronizando o tamanho dos gráficos
plt.rcParams['figure.figsize'] = (16,7)

plt.show()

top20
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
fig, ax = plt.subplots()

df_sum = df.groupby(['uf']).sum()

idade = ["16", "17", "18 - 20","21 - 24", "25 - 34", "35 - 44", "45 - 59", "60 - 69", "70 - 79", "+79"]

estados = df_sum.index

eleitores = np.array([df_sum['f_16'],
                     df_sum['f_17'],
                     df_sum['f_18_20'],
                     df_sum['f_21_24'],
                     df_sum['f_25_34'],
                     df_sum['f_35_44'],
                     df_sum['f_45_59'],
                     df_sum['f_60_69'],
                     df_sum['f_70_79'],
                     df_sum['f_sup_79']])

im, cbar = heatmap(eleitores, idade, estados, ax=ax,
                   cmap="Reds", cbarlabel="Eleitores [Idade]")

fig.tight_layout()
plt.show()
