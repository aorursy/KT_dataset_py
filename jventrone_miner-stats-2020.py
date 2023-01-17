from IPython.display import Image

Image("/kaggle/input/Mesa de trabajo 1bitcoin.png")
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# This Dataset was obtained from different Datasets if you want to know, how I made it? Link it. https://github.com/JuanVentrone/Miner_Stat



data=pd.read_csv("/kaggle/input/history_data_rewards.csv")

data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data["Timestamp"]=pd.to_datetime(data["Timestamp"])

data
fig, ax = plt.subplots()

fig.set_size_inches(12, 6.5)

ax.plot(data["Timestamp"],data['hash-rate'])



ax.set(xlabel='Date Time', ylabel='USD Millions',

       title='History Rewards USD')

ax.grid()

plt.show()
def graph(data):

    fig, ax = plt.subplots()

    ax.plot(data["Timestamp"],data["TH/USD Value"])

    fig.set_size_inches(10, 5.5)

    ax.set(xlabel='Date Time', ylabel='USD',

        title='History TH/s Value')

    ax.grid()

    plt.show()



graph(data)
graph(data[data["Timestamp"]>pd.to_datetime("2015-01-01")])
import seaborn as sns

plt.figure(figsize=(10,8))

sns.heatmap(data.corr(), annot=True,linewidths=.5)
# If you want to know how I got those data-set, please link: https://github.com/JuanVentrone/



data_country=pd.read_csv("/kaggle/input/electricity_bill_cost.csv")

data_country
# If you want to know how I got those data-set, please link: https://github.com/JuanVentrone/



data_model_country=pd.read_csv("/kaggle/input/country_model_dataset.csv")

data_model_country
data_model_country.set_index(['Country','Model'])
# concating Profit % for each country.



data_model_country["Real Profit %"]=((data_model_country["USD Profit"]-data_model_country["M.B.C USD"])/data_model_country["USD Profit"])*100

datos= data_model_country.groupby(["Country"])[["Real Profit %"]].sum()/9

datos=datos.sort_values(by="Real Profit %",ascending=False)

datos
plt.rcParams["figure.figsize"] = [13,30]

fig, ax = plt.subplots()

plt.yticks(size =10)

bar= ax.barh(datos.index, datos["Real Profit %"],align='center',color="r")

ax.invert_yaxis()  # labels read top-to-bottom

plt.text(x=15,y=27,s="Profitable Mining",fontdict={'weight': 'regular', 'size': 9})

ax.set_xlabel('Profit in %',size=15)

ax.set_title('Profit Mining of each Country',size=25)

plt.axhline(y=27.5, color='orange', linestyle='solid', linewidth=2)

for i in range(28):    

    ax.get_children()[i].set_color("g")

plt.savefig("Profit_country_rank.png")
# This Dataset was Scrapped in: https://shop.bitmain.com

# Checkout: https://github.com/JuanVentrone/



data_model_new=pd.read_csv("/kaggle/input/models_prices_th.csv")

data_model_new.drop(data_model_new.columns[data_model_new.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data_model_new
data_model_used=pd.read_csv("/kaggle/input/model_used.csv")

data_model_used.drop(data_model_used.columns[data_model_used.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

data_model_used=data_model_used.rename(columns={"TH":'TH/s',"Models":"Models "})

data_model_used
def time_model(data,data_model):

    fig, ax = plt.subplots()

    fig.set_size_inches(18.5, 10.5)

    ax.grid()

    ax.set(xlabel='Date Time', ylabel='Profitability',

       title='Profitability of new Asics models')

    for i in range(len(data_model)):

        data_temp=data[data["Timestamp"]>pd.to_datetime(data_model["time"][i])]

        data_temp[['Timestamp','market-price','TH/USD Value']]

        data_temp["Profit"]=(data_temp["market-price"]*data_temp["TH/USD Value"]*data_model["TH/s"][i])/data_model["Price"][i]

        ax.plot(data_temp["Timestamp"],data_temp["Profit"],label=data_model['Models '][i])

        ax.legend(loc="upper right", frameon=False)



    # ax.plot()   

    plt.show()
time_model(data,data_model_new)
time_model(data,data_model_used)
data_sum=pd.concat([data_model_new,data_model_used],ignore_index=True)

time_model(data,data_sum)