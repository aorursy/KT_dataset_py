import pandas as pd

import matplotlib.pyplot as plt # forplotting the data

%matplotlib inline

df=pd.read_csv("../input/FIFA_Data.csv")

df.head(30)

df["Wage"]

new_df=df.head(30)

new_df
df.columns
#finding the columns with missing data

df.isnull().sum()

        
#indexing data frame

new_df.set_index("Club",inplace=True)

new_df



#Replacing the symbols of currency for the graph plot order to be consistent

new_df=new_df[["Name","Wage","Nationality","Age","Potential"]]

new_df["Wage"]=new_df["Wage"].str.replace("€","")

new_df["Wage"]=new_df["Wage"].str.replace("K","")

new_df["Wage"]=new_df["Wage"].astype(int)

new_df

#plotting top players in FIFA 2019 on  graph based on players and Wage

my_data= new_df.head(5)

plt.plot(my_data.Name,my_data.Wage)#linewidth=2,markersize=10)

plt.xlabel("Name")

plt.ylabel("Wage")

plt.title("Players and Wages")





#plotting top 5 players in FIFA 2019 on Bar chart based on players and Wage 



my_data= new_df.head(5)

#my_data.plot(kind="bar",x="Name",y="Wage") # not working as wage is not a numeric value

plt.bar(my_data.Name,my_data.Wage,color ="Green",width=0.2)

plt.xlabel("Name")

plt.ylabel("Wage")

plt.title("Players and Wages")
#plotting top players in FIFA 2019 on  graph based on players and Wage

my_data= new_df.head(5)

#plt.plot(my_data.Age,my_data.Potential,linewidth=1,markersize=10,marker = "+")

plt.plot(my_data.Name,my_data.Potential,linewidth=1,markersize=10,marker = "*")

plt.xlabel("players")

plt.ylabel("Potential")

plt.title("Players  vs Potential")
