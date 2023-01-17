# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# carico il dataset della protezione civile
df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
df.describe()

%matplotlib inline
import matplotlib.pyplot as plt

# calcolo i morti del giorno, sottraendo il totale morti al totale del giorno precedente
mdg = df.groupby("data").agg({"deceduti":"sum"})-df.groupby("data").agg({"deceduti":"sum"}).shift(1)
# unisco al mio dataset
df.merge(mdg, on="data")
# nuova figura
fig = plt.Figure()

# aggrego il dataset per data, mantenendo solo totale di positivi e totale di ospedalizzati
positivi = df.groupby("data").agg({'totale_positivi': 'sum', "totale_ospedalizzati":"sum"})

# linee mock per la legenda
from matplotlib.pyplot import Line2D
mocks = [Line2D([0],[0], color = "green"),
         Line2D([0],[0], color = "blue")]

(mdg["deceduti"]/positivi["totale_ospedalizzati"]).plot(figsize=(22,12), color="green",x_compat=True)
(mdg["deceduti"]/positivi["totale_positivi"]).plot(color="blue")
plt.xticks(rotation="vertical")
plt.legend(mocks, ["deceduti/ospedalizzati","deceduti/positivi" ])
plt.ylabel("proporzioni")
plt.show()

%matplotlib inline
import matplotlib.pyplot as plt

# calcolo i morti del giorno, sottraendo il totale morti al totale del giorno precedente
morti = df.groupby("data").agg(sum)["deceduti"]

# nuova figura
fig = plt.Figure()

# aggrego il dataset per data, mantenendo solo totale di positivi e totale di ospedalizzati
positivi = df.groupby("data").agg(sum)[['totale_positivi',"totale_ospedalizzati"]]

dati= positivi.merge(morti, on="data")

# linee mock per la legenda
from matplotlib.pyplot import Line2D
mocks = [Line2D([0],[0], color = "green"),
         Line2D([0],[0], color = "red"),
        Line2D([0],[0], color = "orange"),
        Line2D([0],[0], color = "black")]



dati["cumul_ospedalizzati"] = dati["totale_ospedalizzati"].cumsum()

dati["totale_ospedalizzati"].plot(figsize=(22,12), color="green",x_compat=True)
dati["totale_positivi"].plot(color="red",x_compat=True)
dati["deceduti"].plot(color="orange",x_compat=True)




plt.xticks(rotation="vertical")
plt.legend(mocks, ["ospedalizzati","positivi", "deceduti" ])
plt.ylabel("proporzioni")
ratio = dati["deceduti"]/dati["cumul_ospedalizzati"]
ratio.plot(figsize=(22,12), color="red",x_compat=True)
plt.xticks(rotation="vertical")
plt.title("cumulativo Ospedalizzati / cumulativo deceduti")
n=10

ratio = dati["deceduti"]/dati["cumul_ospedalizzati"]
ospedalizzati = dati["totale_ospedalizzati"].iloc[n:]
ratio = ratio.iloc[n:]
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('ricoverati', color=color)
ospedalizzati.plot(figsize=(22,12), color="green",x_compat=True, ax=ax1)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('ratio D/I', color=color)  # we already handled the x-label with ax1
ratio.plot(figsize=(22,12), color="red",x_compat=True, ax=ax2)
ax2.tick_params(axis='y', labelcolor=color)

# linee mock per la legenda
from matplotlib.pyplot import Line2D
mocks = [Line2D([0],[0], color = "green"),
         Line2D([0],[0], color = "red")]

plt.legend(mocks, ["ospedalizzati","ratio morti/ospedalizzati"])

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


