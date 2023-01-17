# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
main_data = pd.read_csv("../input/crime-against-women-20012014-india/crimes_against_women_2001-2014.csv")
main_data.sample(5)
# we don't need Unnamed: 0 column so let's drop it
main_data.drop("Unnamed: 0", axis = 1, inplace = True)
# let's rename the columns
main_data.rename(columns = {"STATE/UT": "State", "DISTRICT": "District",
                            "Kidnapping and Abduction": "Kidnapping",
                            "Assault on women with intent to outrage her modesty": "Assault",
                           "Insult to modesty of Women": "Insult",
                            "Cruelty by Husband or his Relatives":"Cruelty by Husband",
                           "Importation of Girls":"Importation"}, inplace = True)
# replace space to underscore
main_data.columns = main_data.columns.str.replace(" ", "_")
main_data.columns
main_data.sample(5)
# let's convert all the state and district name into upper case
main_data.State = main_data.State.str.upper()
main_data.District = main_data.District.str.upper()
# let's check unique state names
main_data.State.unique()
# there is A & N ISLANDS and A&N ISLANDS and D & N HAVELI and D&N HAVELI both are same so let's fix it
main_data.State.replace("A&N ISLANDS", "A & N ISLANDS", inplace = True)
main_data.State.replace("D&N HAVELI", "D & N HAVELI", inplace = True)
# there is district name is TOTAL DISTRICT(S) which is totals of district cases but we don't need it at this time
# so let's drop that row
main_data.drop(index = main_data.loc[main_data.District == "TOTAL DISTRICT(S)"].index, inplace = True)
# there is also one district named ZZ TOTAL so it's not need so let's drop it
main_data.drop(index = main_data.loc[main_data.District == "ZZ TOTAL"].index, inplace = True)
# preparing year wise data
year_wise_data = main_data.groupby("Year").sum().reset_index()
classy = year_wise_data.melt(id_vars = "Year", var_name = "Cases")
px.treemap(data_frame=classy, path = ["Year", "Cases"], values="value", title = "Women Cromes Year Wise")
# women crime year wise line plot
plt.figure(figsize=(17,10))
for i in year_wise_data.columns:
    if i == "Year":
        continue
    plt.plot(year_wise_data.Year, year_wise_data[i], label = i)
plt.xlabel("Years")
plt.ylabel("Spread of Cases")
plt.xticks(year_wise_data.Year)
plt.legend()
plt.show()
time_wise = main_data.groupby(["State", "Year"]).sum().reset_index()
state = input("Enter the State Name: ").upper() or "UTTAR PRADESH"
state = state if state in main_data.State.unique() else "UTTAR PRADESH"
time_wise = time_wise.loc[time_wise.State == state]
time_wise = time_wise.melt(id_vars = ["Year", "State"], var_name = "Cases")
px.area(data_frame=time_wise, x = "Year", y = "value", color = "Cases", title = "Time wise analysis of " + state)
year_2001 = year_wise_data.loc[year_wise_data.Year == 2001]
year_2001 = year_2001.melt(id_vars = "Year", var_name = "Cases")
year_2014 = year_wise_data.loc[year_wise_data.Year == 2014]
year_2014 = year_2014.melt(id_vars = "Year", var_name = "Cases")
plt.figure(figsize = (12, 8))
plt.bar(year_2014.Cases, year_2014.value, color = "r", label = "2014")
plt.bar(year_2001.Cases, year_2001.value, color = "g", label = "2001")
plt.legend()
plt.xlabel("Cases")
plt.ylabel("Spread")
plt.show()
classy = main_data.groupby(["State", "Year"]).sum().reset_index()
classy = classy.loc[classy.Year == 2014]
Rape = classy.loc[:, ["State", "Rape"]]
plt.figure(figsize = (17, 8))
plt.bar(Rape.State, Rape.Rape, color = "r")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Rape.State, rotation = "vertical")
plt.title("Rape Cases in 2014 state wise analysis")
plt.show()
Kidnapping = classy.loc[:, ["State", "Kidnapping"]]
plt.figure(figsize = (17, 8))
plt.bar(Kidnapping.State, Kidnapping.Kidnapping, color = "y")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Kidnapping.State, rotation = "vertical")
plt.title("Kidnapping Cases in 2014 state wise analysis")
plt.show()
Dowry_Deaths = classy.loc[:, ["State", "Dowry_Deaths"]]
plt.figure(figsize = (17, 8))
plt.bar(Dowry_Deaths.State, Dowry_Deaths.Dowry_Deaths, color = "black")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Dowry_Deaths.State, rotation = "vertical")
plt.title("Dowry_Deaths Cases in 2014 state wise analysis")
plt.show()
Assault = classy.loc[:, ["State", "Assault"]]
plt.figure(figsize = (17, 8))
plt.bar(Assault.State, Assault.Assault, color = "orange")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Assault.State, rotation = "vertical")
plt.title("Assault Cases in 2014 state wise analysis")
plt.show()
Insult = classy.loc[:, ["State", "Insult"]]
plt.figure(figsize = (17, 8))
plt.bar(Insult.State, Insult.Insult, color = "purple")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Insult.State, rotation = "vertical")
plt.title("Insult Cases in 2014 state wise analysis")
plt.show()
Cruelty_by_Husband = classy.loc[:, ["State", "Cruelty_by_Husband"]]
plt.figure(figsize = (17, 8))
plt.bar(Cruelty_by_Husband.State, Cruelty_by_Husband.Cruelty_by_Husband, color = "b")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Cruelty_by_Husband.State, rotation = "vertical")
plt.title("Cruelty_by_Husband Cases in 2014 state wise analysis")
plt.show()
Importation = classy.loc[:, ["State", "Importation"]]
plt.figure(figsize = (17, 8))
plt.bar(Importation.State, Importation.Importation, color = "g")
plt.xlabel("States")
plt.ylabel("Total Cases")
plt.xticks(Importation.State, rotation = "vertical")
plt.title("Importation Cases in 2014 state wise analysis")
plt.show()
topRape = Rape.sort_values("Rape", ascending = False).reset_index(drop = True)[:10]
topRape.style.background_gradient(cmap = "Reds")
px.pie(data_frame=topRape, names = "State", values="Rape", title = "Rape cases in top 10 states")
topKidnapping = Kidnapping.sort_values("Kidnapping", ascending = False).reset_index(drop = True)[:10]
topKidnapping.style.background_gradient(cmap = "YlGn")
px.pie(data_frame=topKidnapping, names = "State", values="Kidnapping", title = "Kidnapping cases in top 10 states")
topDowry_Deaths = Dowry_Deaths.sort_values("Dowry_Deaths", ascending = False).reset_index(drop = True)[:10]
topDowry_Deaths.style.background_gradient(cmap = "Greys")
px.pie(data_frame=topDowry_Deaths, names = "State", values="Dowry_Deaths", title = "Dowry_Deaths cases in top 10 states")
topAssault = Assault.sort_values("Assault", ascending = False).reset_index(drop = True)[:10]
topAssault.style.background_gradient(cmap = "Oranges")
px.pie(data_frame=topAssault, names = "State", values="Assault", title = "Assault cases in top 10 states")
topInsult = Insult.sort_values("Insult", ascending = False).reset_index(drop = True)[:10]
topInsult.style.background_gradient(cmap = "Purples")
px.pie(data_frame=topInsult, names = "State", values="Insult", title = "Insult cases in top 10 states")
topCruelty_by_Husband = Cruelty_by_Husband.sort_values("Cruelty_by_Husband", ascending = False).reset_index(drop = True)[:10]
topCruelty_by_Husband.style.background_gradient(cmap = "Blues")
px.pie(data_frame=topCruelty_by_Husband, names = "State", values="Cruelty_by_Husband", title = "Cruelty_by_Husband cases in top 10 states")
topImportation = Importation.sort_values("Importation", ascending = False).reset_index(drop = True)[:10]
topImportation.style.background_gradient(cmap = "Greens")
px.pie(data_frame=topImportation, names = "State", values="Importation", title = "Importation cases in top 10 states")
district_2014 =  main_data.groupby(["State", "District", "Year"]).sum().reset_index()
district_2014 = district_2014.loc[district_2014.Year == 2014]
state = topRape.State[0]
districtRape = district_2014.loc[district_2014.State == state, ["State", "District", "Rape"]]
plt.figure(figsize = (17, 8))
plt.bar(districtRape.District, districtRape.Rape, color = "r")
plt.xlabel("District of " + state)
plt.ylabel("Total Rape Cases")
plt.xticks(districtRape.District, rotation = "vertical")
plt.title("Rape cases in " + state)
plt.show()
state = topKidnapping.State[0]
districtKidnapping = district_2014.loc[district_2014.State == state, ["State", "District", "Kidnapping"]]
plt.figure(figsize = (17, 8))
plt.bar(districtKidnapping.District, districtKidnapping.Kidnapping, color = "y")
plt.xlabel("District of " + state)
plt.ylabel("Total Kidnapping Cases")
plt.xticks(districtKidnapping.District, rotation = "vertical")
plt.title("Kidnapping cases in " + state)
plt.show()
state = topDowry_Deaths.State[0]
districtDowry_Deaths = district_2014.loc[district_2014.State == state, ["State", "District", "Dowry_Deaths"]]
plt.figure(figsize = (17, 8))
plt.bar(districtDowry_Deaths.District, districtDowry_Deaths.Dowry_Deaths, color = "black")
plt.xlabel("District of " + state)
plt.ylabel("Total Dowry_Deaths Cases")
plt.xticks(districtDowry_Deaths.District, rotation = "vertical")
plt.title("Dowry_Deaths cases in " + state)
plt.show()
state = topAssault.State[0]
districtAssault = district_2014.loc[district_2014.State == state, ["State", "District", "Assault"]]
plt.figure(figsize = (17, 8))
plt.bar(districtAssault.District, districtAssault.Assault, color = "orange")
plt.xlabel("District of " + state)
plt.ylabel("Total Assault Cases")
plt.xticks(districtAssault.District, rotation = "vertical")
plt.title("Assault cases in " + state)
plt.show()
state = topInsult.State[0]
districtInsult = district_2014.loc[district_2014.State == state, ["State", "District", "Insult"]]
plt.figure(figsize = (17, 8))
plt.bar(districtInsult.District, districtInsult.Insult, color = "purple")
plt.xlabel("District of " + state)
plt.ylabel("Total Insult Cases")
plt.xticks(districtInsult.District, rotation = "vertical")
plt.title("Insult cases in " + state)
plt.show()
state = topCruelty_by_Husband.State[0]
districtCruelty_by_Husband = district_2014.loc[district_2014.State == state, ["State", "District", "Cruelty_by_Husband"]]
plt.figure(figsize = (17, 8))
plt.bar(districtCruelty_by_Husband.District, districtCruelty_by_Husband.Cruelty_by_Husband, color = "b")
plt.xlabel("District of " + state)
plt.ylabel("Total Cruelty_by_Husband Cases")
plt.xticks(districtCruelty_by_Husband.District, rotation = "vertical")
plt.title("Cruelty_by_Husband cases in " + state)
plt.show()
state = topImportation.State[0]
districtImportation = district_2014.loc[district_2014.State == state, ["State", "District", "Importation"]]
plt.figure(figsize = (17, 8))
plt.bar(districtImportation.District, districtImportation.Importation, color = "g")
plt.xlabel("District of " + state)
plt.ylabel("Total Importation Cases")
plt.xticks(districtImportation.District, rotation = "vertical")
plt.title("Importation cases in " + state)
plt.show()
state = input("Enter State Name: ").upper() or "ANDHRA PRADESH"
classy = district_2014.loc[district_2014.State == state]
classy = classy.melt(id_vars = ["Year", "State", "District"], var_name = "Cases")
px.sunburst(data_frame = classy, path = ["District", "Cases"], values = "value", color = "Cases", height=600,
           title = "District wise status of " + state)