import pandas as pd
data = pd.read_csv("../input/videogamescsv/videogames_sales.csv")
data.head()
data.tail()
data.columns
data.dtypes
is_NaN = data.isnull()
is_NaN
is_NaN.any(axis = 1)
data.isna().any()
data[data.isna().any(axis = 1)]
data["Year"].mean()
data["Year"] = data["Year"].fillna(data["Year"].mean())
data["Year"].isnull().any()
data[data.isna().any(axis = 1)]
data["Year"].dtype
data["Year"] = data["Year"].astype(int)
data["Year"].dtype
data["Year"]
data
%matplotlib inline
import matplotlib.pyplot as plt
data.plot.hist(subplots = True, figsize = (6, 25));
data.head()
year_2000 = data[data["Year"] > 2000]
year_2015 = data[data["Year"] > 2015]
year_2015 = year_2015.reset_index()
year_2015.head()
year_2015 = year_2015.drop(columns = "index")
year_2015.head()
year_2015 = year_2015.drop(columns = "Rank")
year_2015.head()
year_2015
year_2015.plot(kind = "scatter", x = "EU_Sales", y = "JP_Sales")
fig, ax = plt.subplots()
ax.scatter(x = year_2015["Global_Sales"], y  = year_2015["Other_Sales"])
ax.set(title = "Analysis between Global And Other Sales",
           xlabel = "Gloabl Sales",
           ylabel = "Other Sales")
ax.axhline(y = year_2015["Other_Sales"].mean(),
                  linestyle = "--")
year_2015.head()
year_2015.plot(kind = "scatter", 
                         x = "Platform", 
                         y = "EU_Sales", 
                         color = "r",
                         figsize = (10, 6), 
                         title = "Sales Analysis between Platform & EU Sales");
fig.savefig("Platfrom vs Eu.png")
year_2015["Genre"]
year_2015.plot(kind = "scatter", 
                          x = "Genre", 
                          y = "EU_Sales", 
                          color = "g",
                          figsize = (10, 6), 
                          title = "Sales Analysis between Genre & EU Sales");
plt.xticks(rotation = 45);
fig.savefig("Genre vs Eu.png")
year_2015.plot(kind = "scatter",  
                          x = "Platform",  
                          y ="Global_Sales", 
                          figsize = (10, 6), 
                          title = "Sales analysis between Platform & Global Sales");
fig.savefig("Platform vs global.png")