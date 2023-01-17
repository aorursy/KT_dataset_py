# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
name = ['Tag', 'Datum','Dienst 8:30-17:00 Uhr','Nacht 17:00-8:30 Uhr']

new_name = ['Tag', 'Datum','Tagdienst','Nachtdienst']

url="https://www.ferienwiki.de/feiertage/de/baden-wuerttemberg"

dfs = pd.read_html(url)
dienst_pre = pd.read_csv('/kaggle/input/dienste-230919/Dienstplan aktuell.csv', sep=';', usecols = name , encoding = "ISO-8859-1")

dienst_pre['Datum'].astype('datetime64[ns]')

pd.to_datetime(dienst_pre['Datum'],format='%d.%m.%Y')

dienst_pre.columns = new_name

dienst_pre = dienst_pre.dropna()



dienst_pre["Tagdienst"]=[a.strip() for a in dienst_pre["Tagdienst"] ]

dienst_pre["Nachtdienst"]=[a.strip() for a in dienst_pre["Nachtdienst"] ]

dienst_pre.loc[dienst_pre["Nachtdienst"]=="BenDau","Nachtdienst"] = "Ben Dau"

dienst_pre.loc[dienst_pre["Tagdienst"]=="BenDau","Tagdienst"] = "Ben Dau"

dienst_pre.loc[dienst_pre["Nachtdienst"]=="?","Nachtdienst"] = np.nan

dienst_pre.loc[dienst_pre["Tagdienst"]=="?","Tagdienst"] = np.nan



dfs[0]["Datum"]= [dat[:dat.find("(")].strip() for dat in dfs[0]["Datum"] ]

pd.to_datetime(dfs[0]["Datum"],format='%d.%m.%Y')



dienst_pre['Feiertag']= [ (a in list(dfs[0]['Datum'])) for a in dienst_pre["Datum"] ]

dienst_pre['Wochenende']= [ (a in ["Sa","So"]) for a in dienst_pre["Tag"] ]

dienst_pre["Feiertag + Wochenende"]= dienst_pre['Feiertag'] + dienst_pre['Wochenende']



dienst_nacht_fei_we = dienst_pre.loc[dienst_pre["Feiertag + Wochenende"]]["Nachtdienst"]

dienst_nacht_woche = dienst_pre.loc[dienst_pre["Feiertag + Wochenende"] == False]["Nachtdienst"]

dienst_tag_fei_we = dienst_pre.loc[dienst_pre["Feiertag + Wochenende"]]["Tagdienst"]

dienst_tag_woche = dienst_pre.loc[dienst_pre["Feiertag + Wochenende"] == False]["Tagdienst"]



df_nacht_fei_we =pd.DataFrame(dienst_nacht_fei_we.value_counts(),dtype='int')

df_nacht_woche = pd.DataFrame(dienst_nacht_woche.value_counts(),dtype='int')

df_tag_fei_we = pd.DataFrame(dienst_tag_fei_we.value_counts(),dtype='int')

df_tag_woche = pd.DataFrame(dienst_tag_woche.value_counts(),dtype='int')
view = pd.concat([df_nacht_woche, df_nacht_fei_we, df_tag_fei_we], axis=1).fillna(0)

view.columns=["Nacht_Woche","Nacht_We_Feiertag", "Tag_We_Feiertag"]

view.sort_values(["Nacht_Woche","Nacht_We_Feiertag"], axis=0, 

                 ascending=False, inplace=True) 

view.astype(int)
view.plot.bar(stacked=True,figsize=(15, 9),

    fontsize=16,)
df_tag_woche