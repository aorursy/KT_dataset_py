#Import the data processing and visualization packages.

%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt
#Loading the the file.

# file = "C:/Users/Miguel/Desktop/Video_Games_Sales_as_at_22_Dec_2016.csv"

file = "../input/Video_Games_Sales_as_at_22_Dec_2016.csv"



df_videogames = pd.read_csv(file)
#Visualizing the size of the set.

df_videogames.shape
#Obtaining information of the variables.

df_videogames.info()
#Observing a sample from the set.

df_videogames.sample(5)
#Calculating the sum of the NaN values.

miss_values = df_videogames.isnull().sum()



#Calculating the percentaje from the previous sums.

miss_values_percent = (miss_values*100/len(df_videogames)).round(2)



#Joining both outputs on the same table

miss_values_table = pd.concat([miss_values,miss_values_percent], axis=1)



#Renaming the columns and selecting the rows which value is different from 0.

miss_values_table = miss_values_table.rename(columns={0:"Total de NaN", 1:"% de NaN"})

miss_values_table[miss_values_table.loc[:,"Total de NaN"] != 0]
#Eliminating all rows with NaN values.

df_videogames = df_videogames.dropna(axis=0)



#Eliminating all the columns that do not interest us.

df_videogames = df_videogames.drop(labels=["Critic_Count","User_Count","Developer"], axis=1)



#Counting possible duplicated rows and erasing them.

duplicated_rows = df_videogames.duplicated().sum()   #Conteo: 0 duplicados.

df_videogames = df_videogames.drop_duplicates().reset_index(drop=True)



#visualizing the final set shape.

df_videogames.shape
#Changing columns types. 

df_videogames["User_Score"] = pd.to_numeric(df_videogames.User_Score, errors="coerce")

df_videogames["Year_of_Release"] = df_videogames["Year_of_Release"].astype("int")

df_videogames["Critic_Score"] = df_videogames["Critic_Score"].apply(lambda x: x/10)



#Generating two new columns to group the Ratings as a categorical variable.

df_videogames["Critic_Category"] = df_videogames["Critic_Score"].apply(lambda x: "*" if 0<=x<=1.9 else "**" if 2<=x<=3.9 else "***" if 4<=x<=5.9 else "****" if 6<=x<=7.9 else "*****")

df_videogames["User_Category"] = df_videogames["User_Score"].apply(lambda x: "*" if 0<=x<=1.9 else "**" if 2<=x<=3.9 else "***" if 4<=x<=5.9 else "****" if 6<=x<=7.9 else "*****")



#Checking the final result of the data types.

df_videogames.dtypes
#Obtaining a descriptive analysis from the cuantitative variables rounding the result.

df_videogames.describe().round(2)
#Number of videogames sold by genre and platform.

plataforma_X_genero = df_videogames.groupby(["Platform","Genre"])["Name"].agg("count").reset_index()

plataforma_X_genero.pivot(index="Platform",columns="Genre", values="Name")
#Sales of videogames by zone and publisher.

ventas_desarrollador = df_videogames.groupby(["Publisher"])["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"].agg("sum").reset_index()



#Top10 of publishers with more incoming money.

ventas_desarrollador = ventas_desarrollador.sort_values(by=["Global_Sales"],ascending=False).reset_index(drop=True)

ventas_desarrollador.head(10)
#Sales of videogames by zone and name.

ventas_ratings = df_videogames.groupby(["Name"])["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"].agg("sum").reset_index()



#Top10 of videogames with more sells.

ventas_ratings = ventas_ratings.sort_values(by=["Global_Sales"],ascending=False).reset_index(drop=True)

ventas_ratings.iloc[:10,]
#Global sales by critics and users ratings.

ratings = df_videogames.groupby(["User_Category","Critic_Category"])["Global_Sales"].agg("sum").reset_index()

ratings.pivot(index="Critic_Category",columns="User_Category", values="Global_Sales")
#Generating a correlation table to observ the iteraction between variables.

df_videogames.corr().round(2)
#Histogram of the distribution of videogames's publications.

historico = df_videogames.Year_of_Release.max()-df_videogames.Year_of_Release.min()+1



fig, ax = plt.subplots(figsize=(15,7))

ax.hist(df_videogames["Year_of_Release"], bins=historico, color="plum", edgecolor="magenta")



ax.set_title("Figura 1", fontsize=18)

ax.set_xlabel("Año", fontsize=12)

ax.set_ylabel("Número total de videojuegos", fontsize=12)
#Lines graph of sales for each area and genre.

genre_global = df_videogames.groupby(["Genre"])["Global_Sales"].agg("sum")

genre_na = df_videogames.groupby(["Genre"])["NA_Sales"].agg("sum")

genre_eu = df_videogames.groupby(["Genre"])["EU_Sales"].agg("sum")

genre_jp = df_videogames.groupby(["Genre"])["JP_Sales"].agg("sum")

genre_other = df_videogames.groupby(["Genre"])["Other_Sales"].agg("sum")



fig, ax = plt.subplots(figsize=(15,7))

ax.plot(genre_global,"bo--",genre_na, "g^--", genre_eu, "m*--", genre_jp, "rd--", genre_other, "cx--")



ax.grid()

plt.legend(("Ventas globales","Ventas en Norte-América", "Ventas en Europa", "Ventas en Japón", "Ventas en otras regiones"))

ax.set_title("Figura 2",fontsize=18)

ax.set_xlabel("Genero",fontsize=12)

ax.set_ylabel("Número total",fontsize=12)
#Lines graph of sales for each area and platform.

platform_global = df_videogames.groupby(["Platform"])["Global_Sales"].agg("sum")

platform_na = df_videogames.groupby(["Platform"])["NA_Sales"].agg("sum")

platform_eu = df_videogames.groupby(["Platform"])["EU_Sales"].agg("sum")

platform_jp = df_videogames.groupby(["Platform"])["JP_Sales"].agg("sum")

platform_other = df_videogames.groupby(["Platform"])["Other_Sales"].agg("sum")



fig, ax = plt.subplots(figsize=(15,7))

ax.plot(platform_global,"bo--",platform_na, "g^--", platform_eu, "m*--", platform_jp, "rd--", platform_other, "cx--")



ax.grid()

plt.legend(("Ventas globales","Ventas en Norte-América", "Ventas en Europa", "Ventas en Japón", "Ventas en otras regiones"))

ax.set_title("Figura 3",fontsize=18)

ax.set_xlabel("Plataforma",fontsize=12)

ax.set_ylabel("Valor total",fontsize=12)
#Scatter graph to see the correlation of ratings from critics and users.

users = df_videogames.User_Score

critics = df_videogames.Critic_Score

g_sales = df_videogames.Global_Sales

year = df_videogames.Year_of_Release



fig,ax=plt.subplots(figsize=(10,10))

ax.scatter(users, critics, c="goldenrod", edgecolors="chocolate", marker="h", linewidths=1.5, alpha=0.3)



ax.grid()

ax.set_title("Figura 4",fontsize=18)

ax.set_xlabel("Puntuación de los usuarios",fontsize=12)

ax.set_ylabel("Puntuación de los críticos",fontsize=12)
#Sector graph of the differents ESRB Rating.

ratings = df_videogames.groupby(["Rating"])["Name"].agg("count")

labels = ["AO","E", "E10+", "K-A", "M", "RP", "T"]

explode= [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]



fig,ax = plt.subplots(figsize=(12,12))

ax.pie(ratings, labels=labels, explode=explode, autopct="%.2f%%", startangle=90)



ax.set_title("Figura 5", fontsize=18)

plt.legend(["AO: Mayores de 18 años", "E: Todas las edades", "E10+: Mayores de 10 años", "K-A: Equivale a 'E'", "M: Mayores de 17 años", "RP: Pendiente de rating", "T: Mayores de 13 años"])