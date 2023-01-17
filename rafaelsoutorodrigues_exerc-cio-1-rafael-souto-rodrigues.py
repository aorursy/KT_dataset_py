import pandas as pd
anv_df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

anv_df.head(5)
class_vars = [["codigo_ocorrencia","Quantitativa Discreta"],

            ["aeronave_operador_categoria","Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_fabricante","Qualitativa Nominal"],

            ["aeronave_modelo","Qualitativa Nominal"],

            ["aeronave_motor_quantidade","Qualitativa Nominal"],

            ["aeronave_assentos","Quantitativa Discreta"],

            ["aeronave_pais_fabricante","Qualitativa Nominal"],

            ["aeronave_fase_operacao","Qualitativa Nominal"],

            ["total_fatalidades","Quantitativa Discreta"]]

class_vars_df = pd.DataFrame(class_vars, columns=["Variavel", "Classificação"])

class_vars_df
anv_df[class_vars_df["Variavel"]].head()
var_aoc = pd.DataFrame(anv_df["aeronave_operador_categoria"].value_counts())

var_aoc
var_atv = pd.DataFrame(anv_df["aeronave_tipo_veiculo"].value_counts())

var_atv
var_af = pd.DataFrame(anv_df["aeronave_fabricante"].value_counts())

var_af
var_am = pd.DataFrame(anv_df["aeronave_modelo"].value_counts())

var_am
var_amq = pd.DataFrame(anv_df["aeronave_motor_quantidade"].value_counts())

var_amq
var_apf = pd.DataFrame(anv_df["aeronave_pais_fabricante"].value_counts())

var_apf
var_afo = pd.DataFrame(anv_df["aeronave_fase_operacao"].value_counts())

var_afo
tit_graf = 'Total de Ocorrências  x Fase de Operação'

anv_df["aeronave_fase_operacao"].value_counts().plot(kind = 'bar', title = tit_graf)
tit_graf = "Top 10 - Total de Ocorrências  x Fase de Operação"

anv_df["aeronave_fase_operacao"].value_counts().head(10).plot(kind = 'bar', title = tit_graf)
tit_graf = "Total de Ocorrências  x Categoria Operador"

anv_df["aeronave_operador_categoria"].value_counts().plot(kind = 'bar', title = tit_graf)
tit_graf = "Total de Ocorrências  x Tipo de Aeronave"

anv_df["aeronave_tipo_veiculo"].value_counts().plot(kind = 'bar', title = tit_graf)
tit_graf = "Top 3 - Total de Ocorrências  x Tipo de Aeronave"

anv_df["aeronave_tipo_veiculo"].value_counts().head(3).plot(kind = 'pie', title = tit_graf)
tit_graf = "Total de Aeronaves  x Tipo de Motor"

anv_df["aeronave_motor_quantidade"].value_counts().plot(kind = 'bar', title = tit_graf)
tit_graf = "Total de Aeronaves  x Pais Fabricante"

anv_df["aeronave_pais_fabricante"].value_counts().plot(kind = 'bar', title = tit_graf)