import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





game_data = pd.read_csv("../input/videogamesales/vgsales.csv")

#Need to data clean the NANS first

game_data["Year"] = game_data["Year"].fillna(0)

game_data["Year"] = game_data["Year"].astype(int)

game_data.dtypes

#print("Setup Complete")
game_data.head()
yearly_sales = game_data[["Year", "Global_Sales"]].copy()

yearly_sales = yearly_sales.groupby("Year").Global_Sales.count()

yearly_sales = yearly_sales.drop(index = [2016.0, 2017.0, 2020.0, 0])

yearly_sales = yearly_sales.to_frame()

yearly_sales = yearly_sales.astype(int)



#df = df.astype(int)



plt.figure(figsize = (24,8))

plt.title("Number of Games with Sales > 100,000 by year")

sns.barplot(data = yearly_sales, x = yearly_sales.index, y = yearly_sales["Global_Sales"])
area_sales = game_data[["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]].copy().set_index("Year")

area_sales = area_sales.drop(index = [2016, 2017, 2020, 0])



NA_sales = area_sales.groupby("Year")["NA_Sales"].sum().to_frame()

EU_sales = area_sales.groupby("Year")["EU_Sales"].sum().to_frame()

JP_sales = area_sales.groupby("Year")["JP_Sales"].sum().to_frame()

Other_sales = area_sales.groupby("Year")["Other_Sales"].sum().to_frame()



plt.figure(figsize = (14,6))

p1 = plt.bar(x = NA_sales.index, height = NA_sales["NA_Sales"])

p2 = plt.bar(x = EU_sales.index, height = EU_sales["EU_Sales"])

p3 = plt.bar(x = JP_sales.index, height = JP_sales["JP_Sales"])

p4 = plt.bar(x = Other_sales.index, height = Other_sales["Other_Sales"])



plt.title("Regional Sales")

plt.ylabel('Copies of Games sold (Millions)')

plt.legend((p1[0], p2[0], p3[0], p4[0]), ('North America', 'Europe', "Japan", "Other"))

genre_sales = game_data[["Year", "Genre", "Global_Sales"]].copy().set_index("Year")

genre_sales = genre_sales.drop(index = [2016, 2017, 2020, 0])



genre_sales["Year"] = genre_sales.index.copy()

genre_sales = genre_sales.set_index("Genre")

genre_sales = genre_sales.pivot_table(index = "Genre",

                                      columns = "Year",

                                      values = "Global_Sales",

                                      aggfunc = "sum")

genre_sales = genre_sales.fillna(0).round(1)



plt.figure(figsize = (20,6))

plt.title("Genre Popularity By Year")

sns.heatmap(data=genre_sales, annot = True, fmt='g')





genre_sales#["Genre"].unique