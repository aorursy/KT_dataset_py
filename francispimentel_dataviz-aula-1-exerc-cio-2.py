import pandas as pd
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.shape
df.dtypes
df.head()
df['aeronave_nivel_dano'].value_counts()
variaveis = [

    ["aeronave_tipo_veiculo", "Qualitativa Nominal", ""],

    ["aeronave_fabricante", "Qualitativa Nominal", ""],

    ["aeronave_motor_quantidade", "Qualitativa Ordinal", "Pode ser discretizada"],

    ["aeronave_pmd", "Quantitativa Discreta", ""],

    ["aeronave_registro_segmento", "Qualitativa Nominal", ""],

    ["aeronave_fase_operacao", "Qualitativa Nominal", ""],

    ["aeronave_nivel_dano", "Qualitativa Ordinal", ""],

    ["total_fatalidades", "Quantitativa Discreta", ""],

]

variaveis = pd.DataFrame(variaveis, columns=["Variavel", "Classificação", "Observações"])

variaveis
from IPython.display import display



for variavel, classificacao in zip(variaveis['Variavel'], variaveis['Classificação']):

    if "Qualitativa" in classificacao:

        display(df[variavel].value_counts().reset_index())
import matplotlib.pyplot as plt
data = df['aeronave_tipo_veiculo'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(10, 5))

    plt.pie(x=data['aeronave_tipo_veiculo'])

    plt.title('Tipo de aeronave')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
data = df['aeronave_fabricante'].value_counts().reset_index().head(30)



with plt.style.context('default'):

    plt.figure(figsize=(6, 6))

    plt.barh(y=data['index'][::-1], width=data['aeronave_fabricante'][::-1], color='forestgreen')

    plt.title('Fabricantes mais frequentes')

    plt.xlabel('Número de aeronaves')

    plt.ylabel('Fabricante')

    plt.show()
data = df['aeronave_motor_quantidade'].value_counts().reset_index()

data['qtd'] = pd.Series([1, 2, 3, 0, -1, 4])

data['perc'] = data['aeronave_motor_quantidade'] * 100 / data['aeronave_motor_quantidade'].sum()

data = data[data['qtd'] >= 0].sort_values(by='qtd')



with plt.style.context('default'):

    fig, axs = plt.subplots(ncols=2,figsize=(10, 3))

    plt.suptitle('Quantidade de Motores')

    plt.subplots_adjust(wspace=0, hspace=0)

    rect0 = axs[0].barh(y=data['index'], width=data['perc'], color='red')

    axs[0].invert_xaxis()

    axs[0].set_xlabel('Percentual')

    rect1 = axs[1].barh(y=data['index'], width=data['aeronave_motor_quantidade'])

    axs[1].set_yticks([])    

    axs[1].set_xlabel('Total')

    plt.show()
with plt.style.context('default'):

    fig, axs = plt.subplots(2, 1, figsize=(15, 8))

    axs[0].boxplot(df['aeronave_pmd'], sym='go', vert=False, patch_artist=True)

    axs[0].set_xlim([-1000, 15000])

    axs[0].set_title('Distribuição do campo \'aeronave_pmd\'')

    axs[0].set_yticks([])

    axs[1].boxplot(df['aeronave_pmd'], sym='go', vert=False, patch_artist=True)

    axs[1].set_xlim([ -10000, 410000])

    axs[1].set_yticks([])

    plt.show()
data = df['aeronave_registro_segmento'].value_counts().reset_index()



with plt.style.context('default'):

    fig = plt.figure(figsize=(15, 8))

    plt.barh(y=data['index'][::-1], width=data['aeronave_registro_segmento'][::-1], color='coral')

    plt.title('Segmento das aeronaves')

    plt.xlabel('Quantidade')

    plt.ylabel('Segmento')

    plt.show()
data = df['aeronave_fase_operacao'].value_counts().reset_index()



with plt.style.context('default'):

    plt.figure(figsize=(6, 6))

    plt.barh(y=data['index'][::-1], width=data['aeronave_fase_operacao'][::-1])

    plt.title('Distribuição das Fases de Operação')

    plt.xlabel('Número de registros')

    plt.ylabel('Fase de Operação')

    plt.show()
data = df['aeronave_nivel_dano'].value_counts().reset_index()

data['qtd'] = pd.Series([2, 4, 3, 5, 1])

data['perc'] = data['aeronave_nivel_dano'] * 100 / data['aeronave_nivel_dano'].sum()

data = data[data['qtd'] >= 0].sort_values(by='qtd')



with plt.style.context('default'):

    fig, axs = plt.subplots(ncols=2,figsize=(10, 3))

    plt.suptitle('Nível do Dano')

    plt.subplots_adjust(wspace=0, hspace=0)

    rect0 = axs[0].barh(y=data['index'], width=data['perc'], color='lightblue')

    axs[0].invert_xaxis()

    axs[0].set_xlabel('Percentual')

    axs[1].barh(y=data['index'], width=data['aeronave_nivel_dano'], color='lightgreen')

    axs[1].set_yticks([])    

    axs[1].set_xlabel('Total')

    plt.show()
data = df['total_fatalidades'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(10, 5))

    plt.pie(x=data['total_fatalidades'])

    plt.title('Total de fatalidades')

    plt.legend(labels=['Nenhuma', 'Uma', 'Duas', 'Três', 'Quatro'], loc='lower right', bbox_to_anchor=(1.3, 0.1))

    plt.show()