import pandas as pd
df = pd.read_csv('../input/anv.csv', delimiter=',')
nRow, nCol = df.shape
print(f'{nRow} linhas e {nCol} colunas no DataFrame')
df.info()
df.head(10)

resposta = [
    ["aeronave_fabricante", "Qualitativa Nominal"],
    ["aeronave_modelo","Qualitativa Nominal"],
    ["aeronave_motor_tipo","Qualitativa Nominal"],
    ["aeronave_motor_quantidade","Qualitativa Ordenal"],
    ["aeronave_ano_fabricacao","Quantitativa Discreta"],
    ["aeronave_pais_fabricante","Qualitativa Nominal"],
    ["aeronave_voo_origem","Qualitativa Nominal"],
    ["aeronave_voo_destino","Qualitativa Nominal"],
    ["aeronave_fase_operacao","Qualitativa Nominal"],
    ["aeronave_nivel_dano","Qualitativa Ordinal"],
    ["total_fatalidades","Quantitativa Discreta"]
]
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
for index, row in resposta.iterrows():
    if ('Qualitativa' in row['Classificação']):
        print(df[row['Variavel']].value_counts().sort_index())
import matplotlib.pyplot as plt
allColumns = list(df)
for index, row in resposta.iterrows():
    allColumns.remove(row['Variavel'])
dfClean = df.copy()
for row in allColumns:
    dfClean.drop(row,1, inplace=True)
dfClean
fatalitiesByEngineType = dfClean.groupby('aeronave_motor_tipo')['total_fatalidades'].count()
fatalitiesByEngineType.plot(kind='pie')