import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_palette("Set3",10)

sns.palplot(sns.color_palette())

sns.set_context("talk")
game_sale=pd.read_csv(r"../input/vgsales.csv")
f,ax=plt.subplots(1,1,figsize=(4,4))

game_sale.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)[:10].plot.pie()

ax.set_ylabel("")

plt.tight_layout()
f,ax=plt.subplots(1,1,figsize=(12,5))

game_sale.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)[:10].plot(kind="bar")

ax.set_ylabel("revenue")

ticks=plt.setp(ax.get_xticklabels(), rotation=20, fontsize=7)
Nintedo_All_Sales=dict({"NA_Sales":game_sale[game_sale["Publisher"]=="Nintendo"]["NA_Sales"].sum(),

                         "EU_Sales":game_sale[game_sale["Publisher"]=="Nintendo"]["EU_Sales"].sum(),

                        "JP_Sales":game_sale[game_sale["Publisher"]=="Nintendo"]["JP_Sales"].sum(),

                       "Other_Sales":game_sale[game_sale["Publisher"]=="Nintendo"]["Other_Sales"].sum()})
f,ax=plt.subplots(figsize=(4,4))

colors=['#ffdc80', '#fcaf45']

labels=list(Nintedo_All_Sales.keys())

colors=['#ffdc80', '#fcaf45', '#f56040', '#e1306c', '#c13584']

plt.pie(list(Nintedo_All_Sales.values()),colors=colors,labels=labels,autopct='%1.1f%%',startangle=90)

plt.tight_layout()
EA_All_Sales=dict({"NA_Sales":game_sale[game_sale["Publisher"]=="Electronic Arts"]["NA_Sales"].sum(),

                         "EU_Sales":game_sale[game_sale["Publisher"]=="Electronic Arts"]["EU_Sales"].sum(),

                        "JP_Sales":game_sale[game_sale["Publisher"]=="Electronic Arts"]["JP_Sales"].sum(),

                       "Other_Sales":game_sale[game_sale["Publisher"]=="Electronic Arts"]["Other_Sales"].sum()})
f,ax=plt.subplots(figsize=(4,4))

colors=['#ffdc80', '#fcaf45']

labels=list(EA_All_Sales.keys())

colors=['#ffdc80', '#fcaf45', '#f56040', '#e1306c', '#c13584']

plt.pie(list(EA_All_Sales.values()),colors=colors,labels=labels,autopct='%1.1f%%',startangle=90)

plt.tight_layout()

Activision_All_Sales=dict({"NA_Sales":game_sale[game_sale["Publisher"]=="Activision"]["NA_Sales"].sum(),

                         "EU_Sales":game_sale[game_sale["Publisher"]=="Activision"]["EU_Sales"].sum(),

                        "JP_Sales":game_sale[game_sale["Publisher"]=="Activision"]["JP_Sales"].sum(),

                       "Other_Sales":game_sale[game_sale["Publisher"]=="Activision"]["Other_Sales"].sum()})
f,ax=plt.subplots(figsize=(4,4))

colors=['#ffdc80', '#fcaf45']

labels=list(Activision_All_Sales.keys())

colors=['#ffdc80', '#fcaf45', '#f56040', '#e1306c', '#c13584']

plt.pie(list(Activision_All_Sales.values()),colors=colors,labels=labels,autopct='%1.1f%%',startangle=90)

plt.tight_layout()
