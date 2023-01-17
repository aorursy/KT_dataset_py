import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns
dataset = pd.read_csv('../input/videogamesales/vgsales.csv', index_col='Rank')

print(dataset.head())

print(dataset.info())
#Missing Values 



print(dataset.isna().any())



# Bar plot of missing values by variable

dataset.isna().sum().plot(kind="bar")



# Show plot

plt.show()



#Drop duplicates 



dataset.drop_duplicates()
#How many platforms are there in the dataset? 



dataset_countplatforms = dataset['Platform'].nunique()

print("There are", dataset_countplatforms, "platforms.")



#How many videogames are there in the dataset? 



dataset_countnames = dataset['Name'].nunique()

print("There are", dataset_countnames, "games.")



#Years to years?



Year_Min = dataset['Year'].min()

Year_Max = dataset['Year'].max()

print("The dataset starts in:", Year_Min)

print("The dataset ends in:", Year_Max)
#How many games by platform? 



Games_Count = dataset.groupby('Platform')['Name'].agg('count')

Games_Count = Games_Count.sort_values(ascending = False)

print(Games_Count)
#Most popular genre in the dataset by Sum of Worldwide Sales



Grouped_Genre = dataset.pivot_table(values='Global_Sales',index='Genre', aggfunc='sum')

print(Grouped_Genre)

Grouped_Genre = Grouped_Genre.sort_values(by = 'Global_Sales',ascending  = False)



# Plot the data  

plt.rcParams['figure.figsize'] = (10,10)

sns.barplot(Grouped_Genre['Global_Sales'],Grouped_Genre.index, orient='h')

plt.title("Most sold genre games in terms of Global sales")

plt.ylabel("Genres")

plt.xlabel("Sales")

plt.show()
#5 best Publisher  



plt.rcParams['figure.figsize'] = (10,10)

Publisher_Sales  = dataset.pivot_table(index = 'Publisher' ,values = 'Global_Sales',aggfunc = np.sum)

Publisher_Sales  = Publisher_Sales.sort_values(by = 'Global_Sales',ascending  = False).head(5)

sns.barplot(Publisher_Sales['Global_Sales'],Publisher_Sales.index, orient='h')

plt.title("Most important publishers in terms of Global sales")

plt.ylabel("Publishers")

plt.xlabel("Sales")

plt.show()
#5 Best Platform



plt.rcParams['figure.figsize'] = (10,10)

Platform_Sales  = dataset.pivot_table(index = 'Platform' ,values = 'Global_Sales',aggfunc = np.sum)

Platform_Sales  = Platform_Sales.sort_values(by = 'Global_Sales',ascending  = False).head(5)

sns.barplot(Platform_Sales['Global_Sales'],Platform_Sales.index, orient='h')

plt.title("Most sold consoles in terms of Global Sales")

plt.ylabel("Consoles")

plt.xlabel("Sales")

plt.show()
#Best games 



plt.rcParams['figure.figsize'] = (10,10)

Games_Sales  = dataset.pivot_table(index = 'Name' ,values = 'Global_Sales',aggfunc = np.sum)

Games_Sales  = Games_Sales.sort_values(by = 'Global_Sales', ascending  = False).head(10)

print(Games_Sales)

sns.barplot(Games_Sales['Global_Sales'],Games_Sales.index, orient='h')

plt.title("Most sold games in terms of Global sales")

plt.ylabel("Games")

plt.xlabel("Sales")

plt.show()
#Global Sales Evolution 



Global_Sales_Evolution  = dataset.pivot_table(index = 'Year' ,values = 'Global_Sales',aggfunc = np.sum)

sns.lineplot(Global_Sales_Evolution.index, Global_Sales_Evolution['Global_Sales'])



#EU Sales Evolution 



EU_Sales_Evolution  = dataset.pivot_table(index = 'Year' ,values = 'EU_Sales',aggfunc = np.sum)

sns.lineplot(EU_Sales_Evolution.index, EU_Sales_Evolution['EU_Sales'])



#NA Sales Evolution 



NA_Sales_Evolution  = dataset.pivot_table(index = 'Year' ,values = 'NA_Sales',aggfunc = np.sum)

sns.lineplot(NA_Sales_Evolution.index, NA_Sales_Evolution['NA_Sales'])



#JP Sales Evolution 



JP_Sales_Evolution  = dataset.pivot_table(index = 'Year' ,values = 'JP_Sales',aggfunc = np.sum)

sns.lineplot(JP_Sales_Evolution.index, JP_Sales_Evolution['JP_Sales'])



#Other Sales Evolution 



O_Sales_Evolution  = dataset.pivot_table(index = 'Year' ,values = 'Other_Sales',aggfunc = np.sum)

sns.lineplot(O_Sales_Evolution.index, O_Sales_Evolution['Other_Sales'])



plt.rcParams['figure.figsize'] = (10,10)

plt.legend(title='Areas', loc='upper right', labels=['Worldwide', 'Europe', 'North America', 'Japan', 'Others'])

plt.title("Evolution of Sales (all regions) from 1980 to 2020")

plt.ylabel("Sum of Sales")

plt.xlabel("Years")

plt.show()
#Market Share Publishers 



All_Sales = dataset["Global_Sales"].sum()

Sales_Nintendo = dataset[dataset["Publisher"] == "Nintendo"]["Global_Sales"].sum() 

Sales_Nintendo = (Sales_Nintendo / All_Sales) * 100

Sales_Electronic_Arts = dataset[dataset["Publisher"] == "Electronic Arts"]["Global_Sales"].sum() * 100 / All_Sales

Sales_Activision = dataset[dataset["Publisher"] == "Activision"]["Global_Sales"].sum() * 100 / All_Sales

Sales_Sony = dataset[dataset["Publisher"] == "Sony Computer Entertainment"]["Global_Sales"].sum() * 100 / All_Sales

Sales_Ubisoft = dataset[dataset["Publisher"] == "Ubisoft"]["Global_Sales"].sum() * 100 / All_Sales

Market_Plot = [Sales_Nintendo, Sales_Electronic_Arts, Sales_Activision, Sales_Sony, Sales_Ubisoft]

plt.pie(Market_Plot, labels=['Nintendo: 20%', 'Electronic Arts: 12,4%', 'Activision: 8%', 'Sony: 7%', 'Ubisoft: 5%'])

plt.title("Market share among the 5 best publishers")

plt.show()
#Nintendo 



Top_Nintendo_Game = dataset[dataset.Publisher == "Nintendo"][["Name","Global_Sales"]].drop_duplicates(["Name"]).head(1)

Top_Nintendo_Game = Top_Nintendo_Game.set_index('Name')

Top_Nintendo_Game.columns.name = 'Nintendo'



print(Top_Nintendo_Game)



#Electronic Arts



Top_ElectronicArts_Game = dataset[dataset.Publisher == "Electronic Arts"][["Name","Global_Sales"]].drop_duplicates(["Name"]).head(1)

Top_ElectronicArts_Game = Top_ElectronicArts_Game.set_index('Name')

Top_ElectronicArts_Game.columns.name = 'Electronic Arts'

print(Top_ElectronicArts_Game)



#Activision 



Top_Activision_Game = dataset[dataset.Publisher == "Activision"][["Name","Global_Sales"]].drop_duplicates(["Name"]).head(1)

Top_Activision_Game = Top_Activision_Game.set_index('Name')

Top_Activision_Game.columns.name = 'Activision'

print(Top_Activision_Game)



#Sony 



Top_Sony_Game = dataset[dataset.Publisher == "Sony Computer Entertainment"][["Name","Global_Sales"]].drop_duplicates(["Name"]).head(1)

Top_Sony_Game = Top_Sony_Game.set_index('Name')

Top_Sony_Game.columns.name = 'Sony'

print(Top_Sony_Game)



#Ubisoft 



Top_Ubisoft_Game = dataset[dataset.Publisher == "Ubisoft"][["Name","Global_Sales"]].drop_duplicates(["Name"]).head(1)

Top_Ubisoft_Game = Top_Ubisoft_Game.set_index('Name')

Top_Ubisoft_Game.columns.name = 'Ubisoft'

print(Top_Ubisoft_Game)
#By Regions 



##Others



Sales_Others = dataset.groupby(['Name', 'Publisher', 'Platform', 'Genre'])['Other_Sales'].sum()

Sales_Others = Sales_Others.sort_values(ascending=False).head(5)

Sales_Others = Sales_Others.reset_index()



##JAPAN



Sales_INJapan = dataset.groupby(['Name', 'Publisher', 'Platform', 'Genre'])['JP_Sales'].sum()

Sales_INJapan = Sales_INJapan.sort_values(ascending=False).head(5)





##EUROPE 

Sales_INEU = dataset.groupby(['Name', 'Publisher', 'Platform', 'Genre'])['EU_Sales'].sum()

Sales_INEU = Sales_INEU.sort_values(ascending=False).head(5)

Sales_INEU = Sales_INEU.reset_index()



##North America 



Sales_INNA = dataset.groupby(['Name', 'Publisher', 'Platform', 'Genre'])['NA_Sales'].sum()

Sales_INNA = Sales_INNA.sort_values(ascending=False).head(5)

Sales_INNA = Sales_INNA.reset_index()



print(Sales_Others)

print(Sales_INJapan)

print(Sales_INEU)

print(Sales_INNA)