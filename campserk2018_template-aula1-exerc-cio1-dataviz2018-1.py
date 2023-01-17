import pandas as pd
resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
import pandas as pd
import numpy as np
df_anv = pd.read_csv('../input/anv.csv', delimiter=',')
df_anv.head()
df_anv.shape
df_anv.info
df_anv.isnull().sum()

Classificaçào = [["aeronave_tipo_veiculo", "Qualitativa Nominal"],
            ["aeronave_pmd_categoria","Qualitativa Ordinal"],
            ["aeronave_assentos","Quantitativa Discreta"],
            ["aeronave_ano_fabricacao","Qualitativa Ordinal"],
            ["aeronave_pais_fabricante","Qualitativa Nominal"],
            ["aeronave_registro_categoria","Qualitativa Ordinal"],
            ["aeronave_registro_segmento","Qualitativa Nominal"],
            ["aeronave_fase_operacao","Qualitativa Nominal"],
            ["aeronave_tipo_operacao","Qualitativa Nominal"],
            ["aeronave_nivel_dano","Qualitativa Ordinal"],
            ["total_fatalidades","Quantitativa Discreta"]] 
Classificaçào = pd.DataFrame(Classificaçào, columns=["Tipos", "Classificação"])
Classificaçào
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() 
def bar_chart(feature):
    Sobreviveu = df_anv[df_anv['total_fatalidades']==1][feature].value_counts()
    Moreu = df_anv[df_anv['total_fatalidades']==0][feature].value_counts()
    df = pd.DataFrame([Sobreviveu,Moreu])
    df.index = ['Sobreviveu','Moreu']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
bar_chart('aeronave_tipo_veiculo')

bar_chart('aeronave_tipo_operacao')
bar_chart('aeronave_nivel_dano')