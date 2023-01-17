# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/euroleague-basketball-results-20032019/euroleague_dset_csv_03_20_upd.csv")

data.head()
set(data[data["WINNER"].str.contains("Efes")]["WINNER"])
set(data[data["WINNER"].str.contains("fener",case=False)]["WINNER"])
set(data[data["WINNER"].str.contains("gala",case=False)]["WINNER"])
homefener=(data["HT"].str.contains("fener",case=False))&(data["AT"].str.contains("gala",case=False))

homegalata=(data["AT"].str.contains("fener",case=False))&(data["HT"].str.contains("gala",case=False)) 



data[homefener | homegalata ]
homefener=(data["HT"].str.contains("fener",case=False))&(data["AT"].str.contains("efe",case=False))

homeefes=(data["AT"].str.contains("fener",case=False))&(data["HT"].str.contains("efe",case=False)) 



fenerefes=data[homefener | homeefes ]
fenerefes["WINNER"].value_counts()
fenerefes.groupby(by="HT").sum()[["HS","AS"]]
homepana=(data["HT"].str.contains("pana",case=False))&(data["AT"].str.contains("efe",case=False))

homeefes=(data["AT"].str.contains("pana",case=False))&(data["HT"].str.contains("efe",case=False)) 



panaefes=data[homepana | homeefes ]
panaefes["WINNER"].value_counts()
panaefes[panaefes["WINNER"].str.contains("efe",case=False)]
panaefes[(panaefes["HT"].str.contains("efe",case=False))& (panaefes["WINNER"].str.contains("pan",case=False))]
panaefes.to_csv("panaefes.csv",index=False)
homepana=(data["HT"].str.contains("pana",case=False))&(data["AT"].str.contains("oly",case=False))

homeoly=(data["AT"].str.contains("pana",case=False))&(data["HT"].str.contains("oly",case=False)) 



panaoly=data[homepana | homeoly ]

panaoly
panaoly["WINNER"].value_counts()
panascores=0

olyscores=0

for index,match in panaoly.iterrows():

    if "Oly" in match["HT"]:

        olyscores+=match["HS"]

        panascores+=match["AS"]

    else:

        olyscores+=match["AS"]

        panascores+=match["HS"]

print("Panascores: ",panascores)

print("Olimpiyacos Scores: ",olyscores)   

print("Difference: ",abs(panascores-olyscores))
homepana=(data["HT"].str.contains("pana",case=False))&(data["AT"].str.contains("fene",case=False))

homefener=(data["AT"].str.contains("pana",case=False))&(data["HT"].str.contains("fene",case=False)) 



panafener=data[homepana | homefener ]

panafener
panafener["WINNER"].value_counts()
panascores=0

fenerscores=0

for index,match in panaoly.iterrows():

    if "Fen" in match["HT"]:

        fenerscores+=match["HS"]

        panascores+=match["AS"]

    else:

        fenerscores+=match["AS"]

        panascores+=match["HS"]

print("Panascores: ",panascores)

print("Fener Scores: ",fenerscores)   

print("Difference: ",abs(panascores-fenerscores))
homewinng=data[data["HT"]==data["WINNER"]]

homewinning_count=homewinng.shape[0]

print("How Many Home Win: ",homewinning_count)



awaywinning=data[data["HT"]!=data["WINNER"]]

awaywinning_count=awaywinning.shape[0]

print("How Many Away Win: ",awaywinning_count)



total=data.shape[0]

print("Total: ",total)



print("HomeWinnig Rate: ",homewinning_count/total)

print("AwayWinning Rate: ",awaywinning_count/total)
awaywinning["WINNER"].value_counts().sort_values(ascending=False)[0:20]
homewinng["WINNER"].value_counts().sort_values(ascending=False)[0:20]
homewinng["WINNER"].value_counts().sort_values()[0:15]