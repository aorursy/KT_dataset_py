!pip install pandas --quiet

!pip install numpy --quiet

!pip install matplotlib --quiet

##!pip install adjustText --quiet

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import ticker

##from adjustText import adjust_text
def insert_labels(subplot, prefixo='', sufixo=''): 

    for i in subplot.patches:

        subplot.text(i.get_width(),

                     i.get_y(), 

                     prefixo+str(i.get_width())+sufixo,

                     fontsize=12)
df = pd.read_csv("../input/datasetttt/datasetV0.4.csv")
df = df.set_index("Platforms")
brazil = df.query("country == 'BRA'")

brazil
amounts_df = round((df.sum(axis=1)/len(df.columns))*100,2)
categoria = 'Geral'

title = 'Quantidade de funcionalidades por plataforma'

ax = '% Funcionalidades'

ay = 'Plataformas'

amounts_brazil = amounts_df.plot(kind='barh')

amounts_brazil.xaxis.set_major_locator(ticker.MaxNLocator(10)) #ticks

amounts_brazil.set_title(title, pad=25)

amounts_brazil.set_xlabel(ax, labelpad=10)

amounts_brazil.set_ylabel(ay,labelpad=1)

amounts_brazil.spines['top'].set_visible(False)

amounts_brazil.spines['right'].set_visible(False)

texts = [ plt.text(x = amounts_df.values[i]+0.5 , y = i-0.1, s = f"{amounts_df[i]}%", size = 10) for i in range(len(amounts_df))]

plt.savefig("review4", bbox_inches="tight")
amounts_brazil = round((brazil.sum(axis=1)/len(brazil.columns))*100,2)
categoria = 'Geral'

title = 'Funcionalidades por plataforma brasileira'

ax = '% Funcionalidades'

ay = 'Plataformas'

amounts_barh_brazil = amounts_brazil.plot(kind='barh')

amounts_barh_brazil.xaxis.set_major_locator(ticker.MaxNLocator(10)) #ticks

amounts_barh_brazil.set_title(title, pad=25)

amounts_barh_brazil.set_xlabel(ax, labelpad=20)

amounts_barh_brazil.set_ylabel(ay,labelpad=20)

amounts_barh_brazil.spines['top'].set_visible(False)

amounts_barh_brazil.spines['right'].set_visible(False)

amounts_barh_brazil.grid(axis='x', color='grey', linestyle='-', linewidth=0.5, alpha=1)

texts = [ plt.text(x = amounts_brazil.values[i]+0.5 , y = i-0.1, s = f"{amounts_brazil[i]}%", size = 10) for i in range(len(amounts_brazil))]
pesos_df = pd.read_csv("../input/datasetttt/datasetV0.4_pesos.csv")
##price_per_user_df = pd.DataFrame(pesos_df["price/user"])

price_per_user = pesos_df.drop([7], axis = 0)

price_per_user.boxplot(column = "price/user")
price_per_user = price_per_user.set_index("platforms")

price_per_user = round(price_per_user["price/user"],2)
title = "Preço por usuário"

ay = "Plataformas"

ax = "Preço"

p_user = price_per_user.plot(kind = "barh")

p_user.set_title(title, pad = 25)

p_user.set_ylabel(ay, labelpad = 10)

p_user.set_xlabel(ax, labelpad = 10)

p_user.spines['right'].set_visible(False)

p_user.spines['top'].set_visible(False)

texts = [plt.text(x = price_per_user.values[i]+0.5 , y = i-0.1, s = f"R${price_per_user[i]}", size = 10) for i in range(len(price_per_user))]

plt.savefig("review5", bbox_inches="tight")
grade_per_price = pesos_df.drop(7, axis = 0)
grade_per_price = grade_per_price.reset_index().drop("index",axis=1)
grade_per_price
ax1 = "User price"

ay1 = "Score"

title1 = "Score x User price"

plt.figure(figsize=(10,10))

texts = [plt.text(grade_per_price['price/user'][i], grade_per_price['percent_grade'][i], grade_per_price['platforms'][i], ha='left', va='top') for i in range(len(grade_per_price['platforms']))]

for i,type in enumerate(grade_per_price["platforms"]):

    y = grade_per_price["percent_grade"][i]

    x = grade_per_price["price/user"][i]

    plt.scatter(x, y)

plt.xlabel(ax1, fontsize = 15)

plt.ylabel(ay1, fontsize = 15)

plt.title(title1, fontsize = 20, pad = 10)

#adjust_text(texts, arrowprops = dict(arrowstyle = "-", color = "black"), autoalign="y")

plt.grid(b=True, which='major', color='#666666', linestyle='-')

plt.minorticks_on()

plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

##plt.savefig("Data1")

plt.show()

pesos_bra = pesos_df.query("country == 'BRA'")

pesos_bra
pesos_bra = pesos_bra.drop(["country","grade","percent_grade","price","users","price/user"], axis=1)
pesos_bra = pesos_bra.set_index("platforms")
pesos_bra.sum(axis=1).plot(kind="barh")