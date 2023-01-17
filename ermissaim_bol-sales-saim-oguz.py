from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import random

from pandas.plotting import scatter_matrix
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
files = os.listdir("../input")

filesxlxs = [pd.ExcelFile("/kaggle/input/"+i) for i in files  if i.endswith ('.xlsx')] 

filesxlxs_names = [i[:-5] for i in files if i[-5:] =='.xlsx']
columns = ['Type', 'EAN', 'Artikelomschrijving', 'Datum', 'Bestelnummer', 'Aantal','Tarief-\ngroep', 'Tarief','Bedrag', 'BTW %', 'Btw-bedrag', 'Bedrag\n(incl. BTW)', 'Land van verzending', 'Reden', 'Opmerking']

df_all = pd.DataFrame(columns=columns)



for i in filesxlxs:                             

    df = pd.read_excel(i, i.sheet_names[1])

    df.columns = list(df.iloc[4])

    df = df[5:]

    df = df[df.Bestelnummer.isna() == False ]

    df_all = pd.concat([df_all,df])

    
df_all
df_all.describe().T
bestels = df_all[['Bestelnummer','EAN', 'Artikelomschrijving', 'Datum', 'Aantal','Land van verzending']]

bestels
a = bestels.loc[bestels['Bestelnummer'] == "1106872706" ]

a
EAN_not_NaN = bestels[bestels.EAN.isna() == False ]

EAN_not_NaN
EAN_NaN = bestels[bestels.EAN.isna() == True]

EAN_NaN = EAN_NaN['Bestelnummer']

EAN_NaN
EAN_Full = pd.merge(EAN_NaN,EAN_not_NaN, how='left',on=['Bestelnummer'])

EAN_Full = EAN_Full.drop_duplicates(subset = 'Bestelnummer')

EAN_Full
bestels = pd.concat([EAN_not_NaN, EAN_Full], ignore_index=True)

bestels
## type cesidi kadar dataframe olusturup her secidi ayri ayri kaydediyoruz

type_list = df_all['Type'].unique().tolist()

type_list
types = [i for i in range(len(type_list)) ]

for i in range(len(type_list)):

    types[i] = df_all[df_all['Type'] == str(type_list[i])]

    types[i] = types[i][['Bestelnummer','Bedrag\n(incl. BTW)']]
thisdict = {

 'Verzendkosten':"Verzendkosten",

 'Verkoopprijs artikel(en), ontvangen van kopers en door bol.com door te storten': "Verkoopprijs",

 'Correctie verkoopprijs artikel(en)':"Corr_Verkoopprijs",

 'Compensatie':"Compensatie",

 'Bijdrage aan pakketzegel(s)':"DHL_DPD",

 'Bijdrage aan retourzegel(s)':"Retourzegel",

 'Pick&pack kosten':"Pick&pack",

 'Commissie':'Commissie',

 'Correctie commissie':"Corr_Commissie",

 'Compensatie zoekgeraakte artikel(en)':"Compensatie_Zoekgeraakte",

 'Bijdrage aan pakketzegel(s) Koninklijke PostNL B.V.':"PostNL",

 'Correctie verzendkosten':"Corr_Verzendkosten",

 'Correctie pick&pack kosten':"Corr_Pick&pack"

 }





for idx, val in enumerate(type_list):

    if val in thisdict:

        types[idx]=types[idx].rename(columns={'Bedrag\n(incl. BTW)':thisdict[val]})
display(types[0].head())

display(types[1].head())

display(types[2].head())

display(types[3].head())

display(types[4].head())

display(types[5].head())

display(types[6].head())

display(types[7].head())

display(types[8].head())

display(types[9].head())

display(types[10].head())

display(types[11].head())

display(types[12].head())

result=pd.merge(bestels,types[0], how='left',on=['Bestelnummer'])
result
for i in range(1,13):

    result=pd.merge(result,types[i], how='left',on=['Bestelnummer'])
result.tail()
result = result.drop_duplicates(subset = 'Bestelnummer')

result.loc[result.duplicated(keep= False),:]

result['Datum']=result['Datum'].dt.date

result.describe().T
result
Geslacht=[]

kind=['heer','mevrouw']

for i in range(len(result.Bestelnummer)):

    x=random.choice(kind)    

    Geslacht.append(x)

result['Geslacht']=Geslacht
Provincie=[]

kind=['Drenthe', 'Gelderland', 'Groningen', 'Flevoland', 'Friesland', 'Limburg', 'Noord-Brabant', 'Noord-Holland', 'Noord-Holland','Overijssel', 'Utrecht', 'Utrecht', 'Zeeland', 'Zuid-Holland', 'Zuid-Holland']

for i in range(len(result.Bestelnummer)):

    x=random.choice(kind)    

    Provincie.append(x)

result['Provincie']=Provincie
result = result[['Bestelnummer', 'EAN', 'Artikelomschrijving', 'Datum', 'Geslacht', 'Provincie','Land van verzending','Aantal',

                 'Verkoopprijs','Corr_Verkoopprijs','Commissie','Corr_Commissie','Verzendkosten','Corr_Verzendkosten', 'DHL_DPD',

                 'PostNL','Retourzegel', 'Pick&pack', 'Corr_Pick&pack',  'Compensatie_Zoekgeraakte', 'Compensatie',]]

# result.to_excel("output.xlsx")

result.describe().T
result
# Kac cesit urun satmisiz?

len(result['Artikelomschrijving'].value_counts().index)
# Kac adet urun satmisiz?

result['Aantal'].sum()
# ilk bes urunun urun-siparis grafigi

goods_name=result['Artikelomschrijving'].value_counts().index[:5]

goods_name
goods_quantity=result['Artikelomschrijving'].value_counts().values[:5]

goods_quantity

# birinci tum satislarin % 10 undan fazla satmis, ilk bes urun % 40 ini olusturuyor
fig, ax = plt.subplots()

fig=plt.figure(figsize=(20,20))

ax.set_title('Goods name-Quantity')

ax.set_xlabel('Goods name');

ax.set_xticklabels(goods_name, rotation=90)

ax.set_ylabel('Quantity');



ax.bar(goods_name,goods_quantity)

plt.show()
result.groupby(['Land van verzending']).sum()
labels = 'BE','NL'

sizes = [4945,65105]

explode = (0.2, 0)



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, explode=explode, autopct='%1.1f%%', 

        pctdistance=0.8, labeldistance=1.1, 

        textprops={'fontsize': 19,'color':"w"},

        radius=1.5, shadow=True, startangle=270, rotatelabels=False

       )



ax1.set_title("Ulke bazinda satislar/Ciro cinsinden",color='b',size=26, x = 0.9, y = 1.3);

plt.show()

# belcika sayi olarak % 10, ciro olarak ise % 7 
result.groupby(['Provincie']).sum()['Verzendkosten'].index
result.groupby(['Provincie']).sum()['Verzendkosten'].sort_values()
labels = 'Drenthe', 'Flevoland', 'Friesland', 'Gelderland', 'Groningen','Limburg', 'Noord-Brabant', 'Noord-Holland', 'Overijssel', 'Utrecht','Zeeland', 'Zuid-Holland'

sizes = [4160,4067,4940,4796,4898,5623,4711,9476,4081,8579,4726,10135]



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 

        pctdistance=0.8, labeldistance=1.1, 

        textprops={'fontsize': 19,'color':"w"},

        radius=1.5, shadow=True, startangle=270, rotatelabels=False

       )



ax1.set_title("Province bazinda satislar/Ciro cinsinden",color='b',size=26, x = 0.9, y = 1.3);

plt.show()

# 3 province satislarin % 40 ini olusturuyor
result['Geslacht'].value_counts()
result['Geslacht'].value_counts(normalize=True)
result_heer=result[result['Geslacht']=='heer']

result_mevrouw=result[result['Geslacht']=='mevrouw']
result_heer['Verzendkosten'].mean()
green_diamond = dict(markerfacecolor='g', marker='D')

fig3, ax3 = plt.subplots()

ax3.set_title('Heer vs. Verzendkosten')

ax3.boxplot(result_heer['Verzendkosten'], flierprops=green_diamond)
result_mevrouw['Verzendkosten'].mean()
result['Verzendkosten'].max(), result['Verzendkosten'].min()
result['Verzendkosten'].hist(bins=10, range=(-475,0))