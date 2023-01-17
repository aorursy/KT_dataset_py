import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
data_polluted = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv')

data_polluted.head()
resposta = [["codigo_ocorrencia", "Quantitativa Discreta"], ["aeronave_matricula","Qualitativa Nominal"],["aeronave_operador_categoria","Qualitativa Nominal"], 

            ["aeronave_tipo_veiculo","Qualitativa Nominal"], ["aeronave_fabricante","Qualitativa Nominal"], ["aeronave_modelo","Qualitativa Nominal"],  

            ["aeronave_nivel_dano","Qualitativa Nominal"], ["total_fatalidades","Quantitativa Discreta"], 

            ["aeronave_dia_extracao","Data"], ] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
def df_group(df, column):

    name_list = []

    count_list = []

    ele_col_list = pd.unique(df[column])

    ele_col_list.sort()

    for i in ele_col_list:

        discriminated = df.loc[df[column] == i]

        count = discriminated.size

        name_list.append(i)

        count_list.append(count)

        

    return name_list, count_list

#=====================================================================================

name, count = df_group(data_polluted, "aeronave_matricula")

aeronave_matricula = pd.DataFrame([name, count])

name1, count1 = df_group(data_polluted, "aeronave_operador_categoria")

aeronave_operador_categoria = pd.DataFrame([name1, count1])

name2, count2 = df_group(data_polluted, "aeronave_tipo_veiculo")

aeronave_tipo_veiculo = pd.DataFrame([name2, count2])

name3, count3 = df_group(data_polluted, "aeronave_fabricante")

aeronave_fabricante = pd.DataFrame([name3, count3])

name4, count4 = df_group(data_polluted, "aeronave_modelo")

aeronave_modelo = pd.DataFrame([name4, count4])

name5, count5 = df_group(data_polluted, "aeronave_nivel_dano")

aeronave_nivel_dano = pd.DataFrame([name5, count5])
aeronave_operador_categoria
from collections import Counter

def df_hist(df, column):

    listed = pd.DataFrame.from_dict(Counter(sorted(df[column])), orient='index')

    listed.plot(kind='bar')   
df_hist(data_polluted, "aeronave_operador_categoria")#### A função df_hist pode ser utilizada para qualquer uma das colunas, só utilizar o dataframe "data_polluted" e escolher uma das colunas.
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(1)