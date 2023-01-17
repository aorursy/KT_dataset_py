import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df = df[["aeronave_tipo_veiculo", 

         "aeronave_fabricante", 

         "aeronave_motor_quantidade", 

         "aeronave_ano_fabricacao", 

         "aeronave_pais_fabricante", 

         "aeronave_nivel_dano", 

         "total_fatalidades"]].dropna()

df = df[(df[['aeronave_ano_fabricacao']] != 0).all(axis=1)]

df.head(1)
resposta = pd.DataFrame({"Variáveis": df.columns, "Classificação": ["Qualitativa nominal", 

                                                                     "Qualitativa nominal", 

                                                                     "Qualitativa nominal", 

                                                                     "Quantitativa discreta", 

                                                                     "Qualitativa nominal", 

                                                                     "Qualitativa nominal", 

                                                                     "Quantitativa discreta"]})

resposta
aeronave_tipo_veiculo = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

aeronave_tipo_veiculo
aeronave_fabricante = pd.DataFrame(df["aeronave_fabricante"].value_counts())

aeronave_fabricante
aeronave_motor_quantidade = pd.DataFrame(df["aeronave_motor_quantidade"].value_counts())

aeronave_motor_quantidade
aeronave_pais_fabricante = pd.DataFrame(df["aeronave_pais_fabricante"].value_counts())

aeronave_pais_fabricante
aeronave_nivel_dano = pd.DataFrame(df["aeronave_nivel_dano"].value_counts())

aeronave_nivel_dano
aeronave_tipo_veiculo.plot.pie(y="aeronave_tipo_veiculo", figsize=(10,10))
aeronave_tipo_veiculo[~aeronave_tipo_veiculo.index.isin(["AVIÃO", "HELICÓPTERO", "ULTRALEVE"])].plot.pie(y="aeronave_tipo_veiculo", figsize=(10,10))
aeronave_fabricante.sort_values("aeronave_fabricante", ascending=False).head(15).plot.bar(y="aeronave_fabricante", figsize=(10,10))
aeronave_motor_quantidade.plot.pie(y="aeronave_motor_quantidade", figsize=(10,10))
aeronave_motor_quantidade[~aeronave_motor_quantidade.index.isin(["MONOMOTOR", "BIMOTOR"])].plot.pie(y="aeronave_motor_quantidade", figsize=(10,10))
aeronave_pais_fabricante.sort_values("aeronave_pais_fabricante", ascending=False).head(5).plot.bar()
aeronave_nivel_dano.plot.bar(y="aeronave_nivel_dano", figsize=(10,10))
df.aeronave_ano_fabricacao.value_counts().describe()
df.aeronave_ano_fabricacao.hist(bins=50, figsize=(10,10))
df.total_fatalidades.value_counts().plot.bar(figsize=(10,10))