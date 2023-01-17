import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import warnings
from matplotlib_venn import venn3
warnings.filterwarnings('ignore')
data = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)
europe = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
          "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary",
          "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
          "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia",
          "Spain", "Sweden", "United Kingdom of Great Britain and Northern Ireland"]
# checking if all EU countries are in database
countries = data["Q3"].drop(0)
countries_unique = countries.value_counts()

cntrs = list(countries_unique.index)

present = []
absent = []
for e in europe:
    if e in cntrs:
        present.append(e)
    else:
        absent.append(e)
print (absent)
db_EU = data[data["Q3"].isin(europe)]
db_EU["Q3"] = db_EU["Q3"].replace("United Kingdom of Great Britain and Northern Ireland","UK")
Q3 = db_EU["Q3"].value_counts()

plt.figure(figsize=(20,10))
plt.bar(Q3.index,Q3)
plt.xticks(Q3.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents from EU contries")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ages =['22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70-79','80+']

Q2 = db_EU["Q2"].value_counts()
Q2 = Q2.ix[ages]
plt.figure(figsize=(20,10))
plt.bar(Q2.index,Q2, )
plt.xticks(Q2.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Age group of respondents from EU contries")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
salaries = ['Not disclosed','0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000','70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000','200-250,000','250-300,000','300-400,000','400-500,000','500,000+']

db_EU.loc[:,"Q9"] = db_EU["Q9"].replace("I do not wish to disclose my approximate yearly compensation","Not disclosed")
Q9 = db_EU["Q9"].value_counts()
Q9 = Q9.reindex(salaries)

plt.figure(figsize=(20,10))
ax =plt.bar(Q9.index,Q9)
plt.xticks(Q9.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Salaries ranges of respondents from EU contries [$/annum]")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[0].set_color('#990000')
#alternative method:
#ax.patches[Q9.index.get_indexer(['Not disclosed'])[0]].set_facecolor('#990000')
not_disclosed = db_EU[db_EU["Q9"]=="Not disclosed"]
nd_countries = not_disclosed["Q3"].value_counts().rename("nd_counts", axis='columns')

plt.figure(figsize=(20,10))
ax = plt.bar(nd_countries.index,nd_countries)

plt.xticks(nd_countries.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents who do not want disclose salary \nby country (absolute values)")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[0].set_color('#f7b45d')
ax[1].set_color('#990000')
ax[2].set_color('#37f250')
df_ndc = nd_countries.to_frame()
df_Q3 = Q3.to_frame()

Q3_nd = df_Q3.merge(df_ndc, how="outer", left_index=True, right_index=True)
Q3_nd["percentage"] = Q3_nd["nd_counts"]/Q3_nd["Q3"]*100
Q3_nd.sort_values("percentage", inplace=True, ascending=False)

plt.figure(figsize=(20,10))
ax = plt.bar(Q3_nd.index,Q3_nd["percentage"])

plt.xticks(Q3_nd.index,rotation="vertical")
plt.margins(0,0.1)
plt.subplots_adjust(bottom=0.25)
plt.title("Number of respondents who do not want disclose salary \nby country (percentage)")
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
plt.rc('axes', axisbelow=True)
ax[4].set_color('#f7b45d')
ax[6].set_color('#990000')
ax[7].set_color('#37f250')
langs = db_EU[["Q16_Part_1","Q16_Part_2","Q16_Part_3"]]
pythons = langs[langs["Q16_Part_1"]=="Python"].index.values.tolist()
rs = langs[langs["Q16_Part_2"]=="R"].index.values.tolist()
sqls = langs[langs["Q16_Part_3"]=="SQL"].index.values.tolist()

plt.figure(figsize=(20,10))
plt.title("Venn diagram of languages used by respondents")
venn3([set(pythons),set(rs),set(sqls)],("Python","R","SQL"))