import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("../input/03-04_operaciones.csv")

data.head()
data = data[['time','[CNC_Fase1_Corriente2]','[CNC_Fase2_Corriente2]','[CNC_Fase3_Corriente2]','time_stamp']]

data= data[~((data['[CNC_Fase1_Corriente2]'] == 0) & (data['[CNC_Fase2_Corriente2]'] == 0) & (data['[CNC_Fase2_Corriente2]'] == 0))]

data.head(20)
media20s = data[['[CNC_Fase1_Corriente2]','[CNC_Fase3_Corriente2]']].rolling(20).mean()

std10s = data[['[CNC_Fase1_Corriente2]','[CNC_Fase3_Corriente2]']].rolling(10).std()

data[['[CNC_Fase1_Corriente2]_mean','[CNC_Fase3_Corriente2]_mean']] = media20s

data[['[CNC_Fase1_Corriente2]_std','[CNC_Fase3_Corriente2]_std']] = std10s

no_op = data[(data['[CNC_Fase1_Corriente2]'] <= 2.2) & (data['[CNC_Fase1_Corriente2]_mean'] >= 1.6) & (data['[CNC_Fase1_Corriente2]_mean'] <= 2.2) & (data['[CNC_Fase1_Corriente2]_std'] <= 0.25)]

#no_op[(no_op["time_stamp"] > '2019-04-03 10:10:00') & (no_op["time_stamp"] < '2019-04-03 10:14:00')]

print(no_op.shape)

no_op.head(10)
data_op = data[~data.isin(no_op)]

print(data_op.shape)

data_op = data_op.dropna(axis=0)

print(data_op.shape)
import datetime

data_op.drop("time_stamp", axis=1)

date_time_obj = []

for date_time_str in data_op["time"]:

    date_time_obj.append(datetime.datetime.strptime(date_time_str, '%Y-%m-%dT%H:%M:%SZ'))

data_op["time_stamp"] = date_time_obj

diferencias = data_op["time_stamp"].diff()

df_ts_diferencias = pd.DataFrame()

df_ts_diferencias["diferencias"] = diferencias

df_ts_diferencias["time_stamp"] = data_op["time_stamp"]

df_ts_diferencias.head()
cortes = df_ts_diferencias[df_ts_diferencias["diferencias"] >= datetime.timedelta(seconds=10)]

cortes = cortes[4:] #Quito todo lo correspondiente al arranque

cortes.head()
operaciones = []

data_op_index_ts = data_op[["time_stamp","[CNC_Fase1_Corriente2]", "[CNC_Fase2_Corriente2]", "[CNC_Fase3_Corriente2]"]]

data_op_index_ts.set_index("time_stamp", inplace=True)
operaciones = []

list_cortes = cortes["time_stamp"].values 

for i in range(len(list_cortes)-1):

        #print("Desde: "+ str(list_cortes[i]) + " hasta: "+ str(list_cortes[i+1]))

        operaciones.append(data_op_index_ts[list_cortes[i]:list_cortes[i+1]])
for operacion in operaciones:

    print("Desde: "+ str(operacion.index.values[0]) + " hasta: "+ str(operacion.index.values[-1]))

    #op = input("Operacion: ")

    #operacion["Operacion"] = op
operaciones[24].iloc[:-1].plot()
"""

operaciones[24]["Operacion"] = 'Canteado'

operaciones[34]["Operacion"] = 'Taladro'

operaciones[35]["Operacion"] = 'Taladro'

operaciones[38]["Operacion"] = 'NoOp'

operaciones[3]["Operacion"] = 'NoOp'

"""

clasif_inicial = operaciones[0]

for i in operaciones[1:]:

    clasif_inicial = clasif_inicial.append(i)
clasif_inicial.to_csv('class_ini.csv',index=False)

clasif_inicial.to_csv('class_ini_idx.csv')
# https://www.kaggle.com/rtatman/download-a-csv-file-from-a-kernel

# import the modules we'll need

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv", idx=False):  

    csv = df.to_csv(index=idx)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
create_download_link(clasif_inicial, filename="class_ini.csv")
create_download_link(clasif_inicial, filename="class_ini_idx.csv", idx=True)