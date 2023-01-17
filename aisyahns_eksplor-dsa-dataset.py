# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
lokasi = pd.read_csv("../input/dsa-datasets/catatan_lokasi.csv")

lokasi5 = pd.read_csv("../input/tanggal5/tanggal_5.csv", header = None)
lokasi.sample(5)


unique , occur = np.unique(lokasi5["lokasi_dominan"].groupby["id"], return_counts = True)



listhasil = zip(unique, occur)



for elm in listhasil :

    print(elm[0], 'occur', elm[1])
lokasi.groupby("id")["lokasi_dominan"].unique()
profil = pd.read_csv("../input/dsa-datasets/data_profil.csv")
profil.sample(5)
np.unique(profil["divisi"], return_counts = True)
count = 0

for row in lokasi :

    if row.loc[row["tanggal"] == "28/05/19"]["lokasi_dominan"] != "Jakarta" :

        count += 1

    
de = profil[profil["divisi"] == 'Data Engineer']



de["umur"].describe()
bandung = lokasi[lokasi["lokasi_dominan"] == 'Kota Bandung']



bandung["id"].unique()
tgl13 = lokasi[lokasi["tanggal"] == "13/06/19"]



tgl13["lokasi_dominan"].unique()
medan = lokasi[lokasi["lokasi_dominan"] == "Kota Medan"]



medan["id"].unique()
ds = profil[profil["divisi"] == "Data Science"] 



ds["id"].nunique()
tanggal8 = lokasi[lokasi["tanggal"] == "08/06/19"]



tanggal9 = lokasi[lokasi["tanggal"] == "09/06/19"]
tanggal8.sample()
tanggal30 = lokasi[lokasi["tanggal"] == "30/05/19"]



tanggal30.groupby("lokasi_dominan")["id"].nunique()