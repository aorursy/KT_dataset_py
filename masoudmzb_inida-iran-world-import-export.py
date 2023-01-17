# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #statistical data visualization

import matplotlib.pyplot as plt #visualization library

from statsmodels.graphics.tsaplots import plot_acf #Auto-Correlation Plots

from statsmodels.graphics.tsaplots import plot_pacf #Partial-Auto Correlation Plots

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import_df = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")

export_df = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
import_df.isnull().sum()
import_df = import_df.dropna()

import_df = import_df.reset_index(drop=True)

import_df.drop_duplicates(keep="first",inplace=True) # removving all the duplicate rows

export_df = export_df.dropna()

export_df = export_df.reset_index(drop=True)

export_df.drop_duplicates(keep="first",inplace=True) # removving all the duplicate rows
import_countries = import_df["country"].nunique()

export_countries = export_df["country"].nunique()



print("import : {}, export : {}".format(import_countries, export_countries))
# let's find that 248th country :D

country_list = list(import_df.country.unique())

country_list
total_unknown_import = import_df.loc[import_df.country=="UNSPECIFIED"].value.sum()

total_unknown_export = export_df.loc[export_df.country=="UNSPECIFIED"].value.sum()

print("import unknown : {}M$ , export unknown : {}M$".format(total_unknown_import, total_unknown_export))

import_df.loc[import_df.country=="UNSPECIFIED"].sort_values(by="value").tail(5)
most_import_from = import_df.groupby("country").value.sum().sort_values(ascending=False).head()

labels = list(most_import_from.index)

values = list(most_import_from)



most_import_df = pd.DataFrame( columns=["country", "total_import"])

most_import_df.country = labels

most_import_df.total_import = values



most_import_df
g = sns.catplot(data=most_import_df, x="country", y="total_import", kind="bar")

plt.ylabel("million dollar")

g.set_xticklabels(rotation=30)
rest_world_sum = import_df.groupby("country").value.sum().sort_values(ascending=False)[5:].values.sum()

top_5_importers_sum = import_df.groupby("country").value.sum().sort_values(ascending=False)[:5].values.sum()



labels=["Rest of the world", "Top 5"]



colors = ['#3498db','#16a085']

sizes=[rest_world_sum, top_5_importers_sum]



explode = [ 0.1, 0.1]

plt.rcParams["figure.figsize"] = (4,4)

plt.pie(sizes, labels = labels, shadow=True, colors=colors, explode= explode,autopct='%1.2f')

plt.title("5 most importers vs Rest of the world")
China_df = import_df.groupby(['country'])

China_df = China_df.get_group("CHINA P RP")

China_df[China_df.year== 2018].value.sum()



import_per_year_china = []

years = China_df["year"].unique()



for year in years:

    import_per_year_china.append(China_df[China_df.year== year].value.sum())



# if we don't want to plot from df. use plt.plot

plt.plot(years, import_per_year_china, color="red", marker='o')

plt.xlabel("Year", size=18)

plt.ylabel("Imports in million")

plt.fill_between(years, import_per_year_china, facecolor="#99ff99")
Iran_df = import_df.groupby("country").get_group("IRAN")

years = Iran_df.year.unique()

import_per_year_iran = []



for year in years :

    import_per_year_iran.append(Iran_df[Iran_df.year == year].value.sum())



plt.plot(years, import_per_year_iran, marker='o', color="green")

plt.xlabel('YEAR',size=17)

plt.ylabel('IMPORTS IN MILLION $',size=17)

plt.fill_between(years,import_per_year_iran,facecolor='#99ff99')

plt.title('TRADE TRENDS',size=20)
rest_of_imports=Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[5:].sum()

most_imports_from_Iran_values = list(Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[:5])

labels = list(Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[:5].index)







most_imports_from_Iran_values.append(rest_of_imports)

labels.append("REST OF THE IMPORTS")

explode = [ 0.1, 0.3, 0.3, 0.3, 0.3, 0.8]

colors = ['#99ff99','#ffcc99','#66b3ff','#33ff33','#cc9966','#d279d2']

plt.rcParams["figure.figsize"] = (10,10)

plt.pie(most_imports_from_Iran_values, labels = labels,explode=explode,colors = colors, shadow=True,autopct='%1.2f')

plt.title("6 most imports from iran")
export_country = export_df.groupby("country").value.sum().sort_values(ascending=False).head()

labels = list(export_country.index)

values = list(export_country)



most_exports_df = pd.DataFrame( columns=["country", "total_export"])

most_exports_df.country = labels

most_exports_df.total_export = values



most_exports_df
g = sns.catplot(x="country", y="total_export", data=most_exports_df, kind="bar")

g.set_xticklabels(rotation=30)
Iran_df = export_df.groupby("country").get_group("IRAN")



rest_of_exports=Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[5:].sum()

most_exports_from_Iran_values = list(Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[:5])

labels = list(Iran_df.groupby('Commodity').value.sum().sort_values(ascending=False)[:5].index)







most_exports_from_Iran_values.append(rest_of_imports)

labels.append("REST OF THE IMPORTS")



explode = [ 0.1, 0.3, 0.3, 0.3, 0.3, 0.3]

colors = ['#99ff99','#ffcc99','#66b3ff','#33ff33','#cc9966','#d279d2']

plt.rcParams["figure.figsize"] = (10,10)

plt.pie(most_exports_from_Iran_values, labels = labels,explode=explode,colors = colors, shadow=True,autopct='%1.2f')

plt.title("6 most exports from iran")
HSCode=pd.DataFrame()

HSCode['Start']=[1,6,15,16,25,28,39,41,44,47,50,64,68,71,72,84,86,90,93,94,97]

HSCode['End']=[5,14,15,24,27,38,40,43,46,49,63,67,70,71,83,85,89,92,93,96,98]

HSCode['Sections']=['Animals & Animal Products',

'Vegetable Products',

'Animal Or Vegetable Fats',

'Prepared Foodstuffs',

'Mineral Products',

'Chemical Products',

'Plastics & Rubber',

'Hides & Skins',

'Wood & Wood Products',

'Wood Pulp Products',

'Textiles & Textile Articles',

'Footwear, Headgear',

'Articles Of Stone, Plaster, Cement, Asbestos',

'Pearls, Precious Or Semi-Precious Stones, Metals',

'Base Metals & Articles Thereof',

'Machinery & Mechanical Appliances',

'Transportation Equipment',

'Instruments - Measuring, Musical',

'Arms & Ammunition',

'Miscellaneous',

'Works Of Art',]
HSCode.head()
import_df["Sections"]=import_df["HSCode"]

export_df["Sections"]=export_df["HSCode"]

for i in range(0,len(HSCode)):

    import_df.loc[(import_df["Sections"] >= HSCode['Start'][i]) & (import_df["Sections"] <= HSCode['End'][i]),"Sections"]=i

    export_df.loc[(export_df["Sections"] >= HSCode['Start'][i]) & (export_df["Sections"] <= HSCode['End'][i]),"Sections"]=i
IRAN_df_import=import_df.groupby(['country'])

IRAN_df_import=IRAN_df_import.get_group('IRAN')



IRAN_df_export = export_df.groupby(['country'])

IRAN_df_export=IRAN_df_export.get_group('IRAN')
most_imp_iran = IRAN_df_import.groupby(["year", "Commodity"]).agg({"value" : "sum"}).sort_values(by="value").tail(10)

most_exp_iran = IRAN_df_export.groupby(["year", "Commodity"]).agg({"value" : "sum"}).sort_values(by="value").tail(10)

most_imp_iran.plot.barh()
most_exp_iran.plot.barh()