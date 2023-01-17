import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_palette("dark")
sns.set_style("whitegrid")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
emission_table = pd.read_csv("/kaggle/input/international-greenhouse-gas-emissions/greenhouse_gas_inventory_data_data.csv")
emission_table.head()
pd.set_option('display.max_colwidth', -1)
by_category  = emission_table.groupby(['category'])
category_count = by_category.count()
category_count
strp = category_count.index
io = strp[0]
io[108]
io
hdd = len(io)
hdd
io.find("_in_kilotonne_co2_equivalent",0,hdd)
# Here we get success
io[:81]
new_category_index = []
for string in strp:
    p = len(string)
    pos = string.find("_in_kilotonne_co2_equivalent",0,p)
    string = string[:pos]
    new_category_index.append(string)
new_category_index
new_category_index_reborn = []
for lingo in new_category_index:
    q = len(lingo)
    pos = lingo.find("_without",0,p)
    lingo = lingo[:pos]
    new_category_index_reborn.append(lingo)
new_category_index_reborn
short_category = ["co2","ghg(indirect co2)","ghg","hfc","ch4","nf3","n2o","pfc","sf6","hfc+pfc"]
category_count["Shorted_category"] = short_category
category_count
trying_emission = emission_table
replaced_emission = trying_emission.replace(to_replace=["carbon_dioxide_co2_emissions_without_land_use_land_use_change_and_"
                                     "forestry_lulucf_in_kilotonne_co2_equivalent","greenhouse_gas_ghgs_emissions_including_indirect_co2"
                                    "_without_lulucf_in_kilotonne_co2_equivalent","greenhouse_gas_ghgs_emissions_without_land_use_land_use"
                                    "_change_and_forestry_lulucf_in_kilotonne_co2_equivalent","hydrofluorocarbons_hfcs_emissions_in_kilotonne_co2_equivalent",
                                    "methane_ch4_emissions_without_land_use_land_use_change"
                                    "_and_forestry_lulucf_in_kilotonne_co2_equivalent","nitrogen_trifluoride_nf3_emissions_in_kilotonne_co2_equivalent",
                                    "nitrous_oxide_n2o_emissions_without_land_use_land_use_change" 
                                    "_and_forestry_lulucf_in_kilotonne_co2_equivalent","perfluorocarbons_pfcs_emissions_in_kilotonne_co2_equivalent",
                                    "sulphur_hexafluoride_sf6_emissions_in_kilotonne_co2_equivalent",
                                    "unspecified_mix_of_hydrofluorocarbons_hfcs_and_perfluorocarbons"
                                    "_pfcs_emissions_in_kilotonne_co2_equivalent"], value = ["CO2","GHG(Indirect CO2)","GHG","HFC","CH4","NF3","N2O","PFC","SF6","HFC+PFC"])

# replacing and changing the data and it's index for better EDA(Exploratory Data Analysis)
l = replaced_emission.groupby(["category"],as_index=False)
l.count()
plt.figure(figsize=(15,7))
ax = sns.countplot(replaced_emission["category"])
ax.set_xticklabels(ax.get_xticklabels(),rotation=40, ha="right", fontsize=14)
plt.tight_layout()
plt.xlabel("Gas category",fontsize=16)
plt.ylabel("Count",fontsize=16)
plt.rcParams["figure.figsize"] = [15, 10]
plt.show()
loct = replaced_emission.groupby(['category'])['value'].sum()
replaced_emission['Total Emitted Gas'] = replaced_emission['value'].groupby(replaced_emission['category']).transform('sum')
loct.values
new_dataframe_emission = pd.DataFrame(loct.index)
new_dataframe_emission["Total Amount Emitted(In Kilotones)"] = loct.values
new_dataframe_emission.sort_values(by=['Total Amount Emitted(In Kilotones)'], inplace=True,ascending=False)
new_dataframe_emission
replaced_emission.head()
Australia_data = replaced_emission[replaced_emission["country_or_area"]=="Australia"].groupby(["category","year"],as_index = False)
data_div = pd.pivot_table(replaced_emission,values="value",index = ["country_or_area", "year"],columns = ["category"])
data_div.head(10)
data_div.plot()
replaced_emission["country_or_area"].unique()
gases = data_div.columns.values
gases
# lets define a function that can plot the country data 
def plot_the_country(name):
    find = data_div.loc[name]
    plt.plot(find)
    plt.legend(gases)
    plt.tick_params(labelsize=12)
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.xlim(2000,2014)
plot_the_country("Australia")
plot_the_country("United States of America")
plot_the_country("Denmark")
plot_the_country("Japan")
area_div = pd.pivot_table(replaced_emission, values='value', index=['category', 'year'], columns=['country_or_area'])
area_div.head(20)
countries = area_div.columns.values
def country_wise_plot(name):
    cname = area_div.loc[name]
    plt.plot(cname)
    plt.tick_params(labelsize=14)
    plt.legend(countries, loc = "center left",bbox_to_anchor=(1, 0.5),fontsize = 18,ncol = 3)
    plt.rcParams["figure.figsize"] = [15, 10]
gases
country_wise_plot(gases[0])
def gas_accord_country1(gas_name, country_name):                          # years from 1990-2004
    data = area_div.loc[gas_name]
    data.plot( y = country_name)
    plt.legend(country_name,loc = "center left",bbox_to_anchor=(1, 0.5),fontsize = 18,ncol = 2)
    plt.tick_params(labelsize=14)
    plt.xlabel("Year",fontsize=14)
    plt.xlim(1990,2004)
    plt.rcParams["figure.figsize"] = [15, 10]
    
def gas_accord_country2(gas_name, country_name):
    data = area_div.loc[gas_name]
    data.plot( y = country_name)
    plt.legend(country_name,loc = "center left",bbox_to_anchor=(1, 0.5),fontsize = 18,ncol = 2)
    plt.tick_params(labelsize=14)
    plt.xlabel("Year",fontsize=14)
    plt.xlim(2004,2017)
    plt.rcParams["figure.figsize"] = [15, 10]
gases
countries_name = replaced_emission["country_or_area"].unique()
countries_name
gas_accord_country1(gases[0],countries_name[:8])
gas_accord_country2(gases[0],countries_name[:8])
gas_accord_country1(gases[0],countries_name[8:16])
gas_accord_country2(gases[0],countries_name[8:16])
gas_accord_country1(gases[0],countries_name[16:24])
gas_accord_country2(gases[0],countries_name[16:24])
gas_accord_country1(gases[0],countries_name[24:32])
gas_accord_country2(gases[0],countries_name[24:32])
gas_accord_country1(gases[0],countries_name[32:40])
gas_accord_country2(gases[0],countries_name[32:40])
gas_accord_country1(gases[0],countries_name[40:43])
gas_accord_country2(gases[0],countries_name[40:43])
data_div.head()
data_div["GHG"].plot()
data_div["GHG(Indirect CO2)"].plot()
cleaned_data = data_div
cleaned_data.head()
cleaned_data["Check"] = cleaned_data["GHG"] - cleaned_data["GHG(Indirect CO2)"]
cleaned_data.head()
cleaned_data[(cleaned_data["Check"] !=0) & (cleaned_data["Check"] < 0)]
cleaned_data = cleaned_data.drop("GHG(Indirect CO2)",axis = 1)
cleaned_data = cleaned_data.drop("Check",axis = 1)
cleaned_data["HFC+PFC"].isnull().sum()
Regular_data = cleaned_data[cleaned_data["HFC+PFC"].isnull()==False]
len(Regular_data)
Regular_data = Regular_data.reset_index()
Regular_data.head(10)

Regular_data.groupby("country_or_area").count()
gases
gas_accord_country1(gases[5],["Germany","United States of America"])
gas_accord_country2(gases[5],["Germany","United States of America"])
gas_accord_country1(gases[5],["European Union"])
gas_accord_country2(gases[5],["European Union"])
gases
cleaned_data.head()
countries_name
cleaned_data[cleaned_data["HFC"].isnull()==True]
gas_accord_country1(gases[4],countries_name[:10])
gas_accord_country2(gases[4],countries_name[:10])
gas_accord_country1(gases[4],countries_name[10:20])
gas_accord_country2(gases[4],countries_name[10:20])
gas_accord_country1(gases[4],countries_name[20:30])
gas_accord_country2(gases[4],countries_name[20:30])
gas_accord_country1(gases[4],countries_name[30:40])
gas_accord_country2(gases[4],countries_name[30:40])
gas_accord_country1(gases[4],countries_name[40:])
gas_accord_country2(gases[4],countries_name[40:])
cleaned_data[cleaned_data["PFC"].isnull()==True]
gas_accord_country1(gases[8],countries_name[10:20])
gas_accord_country2(gases[8],countries_name[10:20])
gas_accord_country1(gases[8],countries_name[20:30])
gas_accord_country2(gases[8],countries_name[20:30])
gas_accord_country1(gases[8],countries_name[30:40])
gas_accord_country2(gases[8],countries_name[30:40])
gas_accord_country1(gases[8],countries_name[40:43])
gas_accord_country2(gases[8],countries_name[40:43])
cleaned_data[cleaned_data["NF3"].isnull()==False]
nf3_data = cleaned_data[cleaned_data["NF3"].isnull()==False].reset_index()
nf3_data.groupby("country_or_area").count()
nf3_countries = nf3_data.groupby("country_or_area").count().index
nf3_countries
gas_accord_country1(gases[7],nf3_countries)
len(cleaned_data[cleaned_data["SF6"].isnull()==True])
gas_accord_country1(gases[9],countries_name[:10])
gas_accord_country2(gases[9],countries_name[:10])
gas_accord_country1(gases[9],countries_name[10:20])
gas_accord_country2(gases[9],countries_name[10:20])
gas_accord_country1(gases[9],countries_name[20:30])
gas_accord_country2(gases[9],countries_name[20:30])
gas_accord_country1(gases[9],countries_name[30:40])
gas_accord_country2(gases[9],countries_name[30:40])
gas_accord_country1(gases[9],countries_name[40:])
gas_accord_country2(gases[9],countries_name[40:])
gases
len(cleaned_data[cleaned_data["N2O"].isnull()==True])
gas_accord_country1(gases[6],countries_name[:10])
gas_accord_country2(gases[6],countries_name[:10])
gas_accord_country1(gases[6],countries_name[10:20])
gas_accord_country2(gases[6],countries_name[10:20])
gas_accord_country1(gases[6],countries_name[20:30])
gas_accord_country2(gases[6],countries_name[20:30])
gas_accord_country1(gases[6],countries_name[30:40])
gas_accord_country2(gases[6],countries_name[30:40])
gas_accord_country1(gases[6],countries_name[40:43])
gas_accord_country2(gases[6],countries_name[40:50])
len(cleaned_data[cleaned_data["GHG"].isnull()==True])
gases
gas_accord_country1(gases[1],countries_name[:5])
gas_accord_country1(gases[2],countries_name[:5])
cleaned_data.head()
cleaned_data["difference"] = cleaned_data["GHG"] - cleaned_data["CO2"]
cleaned_data.head()
gas_accord_country2(gases[1],countries_name[:5])
gas_accord_country1(gases[1],countries_name[5:10])
gas_accord_country2(gases[1],countries_name[5:10])
gas_accord_country1(gases[1],countries_name[10:15])
gas_accord_country2(gases[1],countries_name[10:15])
gas_accord_country1(gases[1],countries_name[15:25])
gas_accord_country2(gases[1],countries_name[15:25])
gas_accord_country1(gases[1],countries_name[25:30])
gas_accord_country2(gases[1],countries_name[25:30])
gas_accord_country1(gases[1],countries_name[30:35])
gas_accord_country2(gases[1],countries_name[30:35])
gas_accord_country1(gases[1],countries_name[35:40])
gas_accord_country2(gases[1],countries_name[35:40])
gas_accord_country1(gases[1],countries_name[40:43])
gas_accord_country2(gases[1],countries_name[40:43])
new_table = pd.pivot_table(replaced_emission, values='value',index=['category'],columns=['country_or_area'])
new_table
clean_new_table = new_table.fillna(0)
clean_new_table['Australia'].index
clean_new_table = clean_new_table.reset_index()
clean_new_table['Australia']
clean_new_table = clean_new_table.drop(clean_new_table.index[[2,3]])
def check_country(name):
    clean_new_table.plot(x = 'category', y = name)
    plt.tick_params(labelsize=14)
    plt.xlabel("Category Of GreenHouse Gases",fontsize=14)
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.legend(fontsize = 20)
def tabulation_new(name):
    point = clean_new_table[name].sum()
    data_storage = clean_new_table[['category',name]]
    data_storage['Percent'] = (data_storage[name]/point * 100)
    print(data_storage)
clean_new_table.columns
check_country('Australia')
tabulation_new('Australia')
check_country('Belgium')
tabulation_new('Belgium')
check_country('Canada')
tabulation_new('Canada')
check_country('European Union')
tabulation_new('European Union')
check_country('France')
tabulation_new('France')
check_country('Germany')
tabulation_new('Germany')
check_country('Italy')
tabulation_new('Canada')
check_country('Japan')
tabulation_new('Japan')
check_country('New Zealand')
tabulation_new('New Zealand')
check_country('Norway')
tabulation_new('Norway')
clean_new_table.columns
check_country('Russian Federation')
tabulation_new('Russian Federation')
clean_new_table
new_table2 = pd.pivot_table(replaced_emission, values='value',index=['country_or_area'],columns=['category'])
new_table2 = new_table2.fillna(0)
sum_of_total_emission = new_table2.sum(axis=1)
new_table2['Total'] = sum_of_total_emission
sum_of_total_emission.sort_values(ascending=False)

