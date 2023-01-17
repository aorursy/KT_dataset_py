import plotly.graph_objects as go

import pandas as pd

from datetime import datetime

import numpy as np

import h2o

from h2o.automl import H2OAutoML
file = open('../input/input.csv', 'r', encoding = "utf-16") #so the data won't be imported in utf-16



data = []

for line in file:

    

    data1 = line.split(',')

    if len(data1) == 7:

        data.append([datetime.strptime(data1[0], '%Y.%m.%d %H:%M'),         

                    float(data1[1]),

                    float(data1[2]), float(data1[3]),

                    float(data1[4]), int(data1[5]),

                    int(data1[6])])



file.close()
df = pd.DataFrame(data)

df.columns = ['DATE', 'OPEN', 'HIGH',

              'LOW', 'CLOSE', 'TICKVOL', 'VOL'

             ]
fig = go.Figure(data=[go.Candlestick(x=df['DATE'],

                open=df['OPEN'], high=df['HIGH'],

                low=df['LOW'], close=df['CLOSE'])

                     ])



fig.update_layout(xaxis_rangeslider_visible=False)

fig.show()
cola = df.tail()
fig = go.Figure(data=[go.Candlestick(x=cola['DATE'],

                open=cola['OPEN'], high=cola['HIGH'],

                low=cola['LOW'], close=cola['CLOSE'])

                     ])



fig.update_layout(xaxis_rangeslider_visible=False)

fig.show()
# Let's add a new column "shifted" to the dataframe and populate it with the 

# closing value for the posterior row



shifted = df["CLOSE"][1:].tolist()

shifted.append(np.nan) # adds a NaN value so the list will be of same length as df

df["SHIFTED"] = shifted

df = df.dropna()
df = df.drop(['DATE', 'TICKVOL'], axis = 1)
df.to_csv(r'salida.csv', index = False)
h2o.init(nthreads = -1, max_mem_size = "16g")
info_df = h2o.import_file("salida.csv")
info_df.describe(chunk_summary=True)
parts = info_df.split_frame(ratios=[.8])

train = parts[0]

test = parts[1]
y = "SHIFTED"

x = info_df.columns

x.remove(y)
automodel = H2OAutoML(max_models=20, seed=1)

automodel.train(x=x, y=y, training_frame=train)
automodel.leader
predictions = automodel.leader.predict(test)
predictions