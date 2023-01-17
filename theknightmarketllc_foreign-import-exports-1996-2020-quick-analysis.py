
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import_export_data  = pd.read_csv('../input/foreign-countries-importexportdata/1996_to_2020_data.csv')
import_export_data["Year"] = pd.to_datetime(import_export_data["Year"].astype(str))
import_export_data["Year"] = [x.year for x in import_export_data["Year"]]
World_Exports_for_class = import_export_data.pivot_table(index="Year", columns=["sitc_sdesc"], values="ExportsFASValueBasisYtdDec")
World_Exports_for_class.head()
fig, axes = plt.subplots(5,2, figsize = (16,15))
axes[0,0].plot(World_Exports_for_class["ANIMAL AND VEGETABLE OILS, FATS AND WAXES"], marker = "o", mfc = "r")
axes[0,0].set_title("ANIMAL AND VEGETABLE OILS, FATS AND WAXES")
axes[0,1].plot(World_Exports_for_class["BEVERAGES AND TOBACCO"], marker = "o", mfc = "brown")
axes[0,1].set_title("BEVERAGES AND TOBACCO")
axes[1,0].plot(World_Exports_for_class["CHEMICALS AND RELATED PRODUCTS"], marker = "o", mfc = "black")
axes[1,0].set_title("CHEMICALS AND RELATED PRODUCTS")
axes[1,1].plot(World_Exports_for_class["COMMODITIES & TRANSACTIONS NOT CLASSIFIED ELSEWHER"], marker = "o", mfc = "y")
axes[1,1].set_title("COMMODITIES & TRANSACTIONS NOT CLASSIFIED ELSEWHER")
axes[2,0].plot(World_Exports_for_class["CRUDE MATERIALS, INEDIBLE, EXCEPT FUELS"], marker = "o", mfc= "orange")
axes[2,0].set_title("CRUDE MATERIALS, INEDIBLE, EXCEPT FUELS")
axes[2,1].plot(World_Exports_for_class["FOOD AND LIVE ANIMALS"], marker = "o", mfc = "y", c = "black")
axes[2,1].set_title("FOOD AND LIVE ANIMALS")
axes[3,0].plot(World_Exports_for_class["MACHINERY AND TRANSPORT EQUIPMENT"], marker = "o", c = "g", mfc = "r")
axes[3,0].set_title("MACHINERY AND TRANSPORT EQUIPMENT")
axes[3,1].plot(World_Exports_for_class["MANUFACTURED GOODS CLASSIFIED CHIEFLY BY MATERIAL"], marker = "o", mfc = "black", c = "r")
axes[3,1].set_title("MANUFACTURED GOODS CLASSIFIED CHIEFLY BY MATERIAL")
axes[4,0].plot(World_Exports_for_class["MINERAL FUELS, LUBRICANTS AND RELATED MATERIALS"], marker = "o", c = "y", mfc = "r")
axes[4,0].set_title("MINERAL FUELS, LUBRICANTS AND RELATED MATERIALS")
axes[4,1].plot(World_Exports_for_class["MISCELLANEOUS MANUFACTURED ARTICLES"], marker = "o", c = "brown", mfc = "white")
axes[4,1].set_title("MISCELLANEOUS MANUFACTURED ARTICLES")
plt.tight_layout();

import_export_data.groupby("Country").mean().drop("World Total").sort_values("ExportsFASValueBasisYtdDec"
                                , ascending =False)["ExportsFASValueBasisYtdDec"].head(10).plot(kind = "bar",
                                    cmap ="coolwarm", figsize = (16,6))
plt.title("Top Ten Countries for Export");
canada = import_export_data[import_export_data["Country"] == "Canada"].pivot_table(index = "Year"
                        , columns = ["sitc_sdesc"], values = "ExportsFASValueBasisYtdDec")
canada.plot(figsize = (20,10), marker = "o", cmap = "magma")
plt.title("Canada Export Sector Performance");
plt.figure(figsize = (16,6))
sns.barplot(x = canada.index, y = "MACHINERY AND TRANSPORT EQUIPMENT", data=canada)
Mexico = import_export_data[import_export_data["Country"] == "Mexico"].pivot_table(index = "Year"
                        , columns = ["sitc_sdesc"], values = "ExportsFASValueBasisYtdDec")
Mexico.head()
Mexico.plot(figsize = (20,10), marker = "o")
plt.title("Mexico Export Sector Performance");
plt.figure(figsize = (16,6))
sns.barplot(x = Mexico.index, y = "MACHINERY AND TRANSPORT EQUIPMENT", data=canada, palette="ocean")
China = import_export_data[import_export_data["Country"] == "China"].pivot_table(index = "Year"
                        , columns = ["sitc_sdesc"], values = "ExportsFASValueBasisYtdDec")
China.head()
China.plot(figsize = (20,10), marker = "o", cmap = "magma")
plt.title("China Export Sector Performance");
plt.figure(figsize = (16,6))
sns.barplot(x = China.index, y = "MACHINERY AND TRANSPORT EQUIPMENT", data=canada, palette="summer")
Worlds_Export_Annual = import_export_data.groupby("Year").sum()
Worlds_Export_Annual["Annual Export pct Change"] = Worlds_Export_Annual["ExportsFASValueBasisYtdDec"].pct_change()
Worlds_Export_Annual["Annual Export pct Change"].plot(kind = "bar", figsize = (16,6))




