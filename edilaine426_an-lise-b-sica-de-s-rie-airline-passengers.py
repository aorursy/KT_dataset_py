import pandas as pd
ts_pass = pd.read_csv('../input/passageiros/airline-passengers.csv')

ts_pass.head()
ts_pass.info()
ts_pass.describe()
#valores nulos 

ts_pass.isnull().sum()
#convertendo month para série de tempo

ts_pass['Month'] = pd.to_datetime(ts_pass['Month'])

ts_pass.info()
#trasnformando month em indice 

ts_pass.set_index('Month', inplace= True)

ts_pass.info()
ts_pass.plot(figsize = (15,6))
tendencia = ts_pass.rolling(12).mean()  #obs: leg(atraso) = 12

tendencia.plot(figsize = (15,6))
sazonalidade = ts_pass.diff()

sazonalidade.plot(figsize = (15,6))
#aplicado corte temporal 

filtro =  (ts_pass.index.year >= 1955) & (ts_pass.index.year <= 1957)

ts_pass[filtro].diff().plot(figsize = (15,6))
