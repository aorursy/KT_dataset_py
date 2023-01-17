import pandas as pd
from datetime import datetime as dt
# dataframe exemplo
df = pd.DataFrame({'Data': ['2020-08-02','2020-08-03','2020-08-04','2020-08-05','2020-08-06','2020-08-07','2020-08-08'],
                   'Vendas': [750, 500, 350, 420, 120, 325, 420],
                   'Faturamento': [1700, 950, 720, 820, 870, 1050, 1200]
                  })
df['Data'] = pd.to_datetime(df['Data'])
df
# Adicionar coluna com o dia da semana
df['Dia_da_semana'] = df['Data'].apply(lambda x: x.strftime("%A"))
df
df.plot.bar(x='Dia_da_semana',
            y='Faturamento',
            rot=45,
            title='Faturamento diário',
            legend=False)