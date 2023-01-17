import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
game = pd.read_csv("../input/videogamesales/vgsales.csv")

game
top_10 = game[0:10]

top_10
plt.figure(figsize = (18,8))

plt.barh(top_10["Name"],top_10["Global_Sales"])

plt.title("Top 10 games Global Sales",fontdict = {"fontsize":20})

plt.savefig("Top 10 games Global Sales.jpg",dpi = 300)

plt.show()
Publisher = list(game.Publisher.unique())
global_sale_of_every_Publisher = pd.Series(dtype = float)

for pub in Publisher :

    data = game.loc[game.Publisher == pub]

    global_sale = sum(data.Global_Sales)

    global_sale_of_every_Publisher[pub] = global_sale
top_5 = global_sale_of_every_Publisher[:5]
plt.figure(figsize = (10.5,9))

plt.pie(top_5,labels = top_5.index,autopct = "%.2f%%",textprops = {"fontsize":13},labeldistance = 1.05)

plt.legend(loc = 4,fontsize  = 12)

plt.title("Top 5 Publisher of Games",fontdict = {"fontsize":25,"fontweight":100})

plt.savefig("Top 5 Publisher of Games",dpi = 300)

plt.show()
Nintendo = game.loc[game.Publisher == "Nintendo"]

Nintendo_1 = Nintendo.sort_values(by = "Year")

Nintendo_1 = Nintendo_1.dropna()

Nintendo_years = Nintendo.Year.unique()

Nintendo_profit_year = pd.Series(dtype = float)

for yea in Nintendo_years:

    data_of_year = Nintendo_1.loc[Nintendo_1.Year == yea]

    total_of_year = data_of_year.Global_Sales.sum(axis = 0)

    Nintendo_profit_year[yea] = total_of_year

Nintendo_profit_year = Nintendo_profit_year.sort_index()

Nintendo_profit_year = Nintendo_profit_year
plt.plot(Nintendo_profit_year)

plt.xlabel("Years",size = 14)

plt.ylabel("Unit Sales",size = 14)

plt.title("Nintendo Gloal Sales from 1983 to 2016",fontdict = {"fontsize":15})

plt.xticks([i for i in range(1983,2017,3)])

plt.savefig("Nintendo Gloal Sales from 1983 to 2016",dpi = 300)

plt.show()
Genre = game.Genre

Genre = Genre.value_counts()
plt.figure(figsize = (8,8))

labels = Genre.index

colors = ["#0033ff","#ff0800","#f700ff","#eeff00","#51ff00","#00ffdd","#ff9d00","#850012","#c7714a","#04615b","#ab8d5e","#00004a"]

plt.pie(Genre,labels = labels,colors = colors,autopct = "%.2f%%") 

plt.title("Games Top Genres",fontdict = {"fontsize":20})

plt.savefig("Games Top Genres",dpi = 300)

plt.show()