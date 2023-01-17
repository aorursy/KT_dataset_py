# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf-8"))



# Get current size

fig_size = plt.rcParams["figure.figsize"]

 

# Set figure width to 12 and height to 9

fig_size[0] = 12

fig_size[1] = 8

plt.rcParams["figure.figsize"] = fig_size





# Any results you write to the current directory are saved as output.


pd.set_option('display.max_rows', 200)



df = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='ISO-8859-1', usecols=[0,1,8,82])



riigid = []

terror_relv_osakaal = []



for el in list(df["country_txt"].unique()):

    

    riik = df[df.country_txt == el]

    uus = riik["weaptype1_txt"].value_counts()

    try:

        osakaal = uus["Firearms"] / len(riik["weaptype1_txt"])

    except:

        osakaal = 0

    

    riigid.append(el)

    terror_relv_osakaal.append(round(osakaal*100))

a = pd.DataFrame({"Riigid": riigid, "Terroriaktid_relvadega_prots": terror_relv_osakaal})

a = a.sort_values(["Terroriaktid_relvadega_prots"], ascending=[False])

a = a.reset_index(drop=True)

a.head(30).set_index('Riigid')['Terroriaktid_relvadega_prots'].plot.bar()

    





aasta = []

relva_terror_arv = []

osakaal2 = []



for el in list(df["iyear"].unique()):

    

    riik = df[(df.iyear == el) &

             (df.country_txt == "United States")]

    uus = riik["weaptype1_txt"].value_counts()

    try:

        terror_relv = uus["Firearms"]

        osakaal = uus["Firearms"] / len(riik["weaptype1_txt"])

    except:

        terror_relv = 0

    aasta.append(el)

    relva_terror_arv.append(terror_relv)

    osakaal2.append(round(osakaal*100))



b = pd.DataFrame({"Aasta": aasta, "Terroriaktid_relvadega": relva_terror_arv, "Osakaal_prots": osakaal2})

b = b.sort_values(["Aasta"], ascending=[True])

b
plt.scatter(b.Aasta, b.Osakaal_prots, s=b.Terroriaktid_relvadega*100)

plt.xlabel("Aasta")

plt.ylabel("Relvade kasutamise protsent USA terrorismis")
