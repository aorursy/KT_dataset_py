# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dc = pd.read_csv('/kaggle/input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv')

dc.head()



dc.info()



dc2 = dc.iloc[:11]

dc2
jumlah_gender = dc.groupby(['SEX'])['name'].count()

jumlah_gender



plt.barh(jumlah_gender.index, jumlah_gender)

plt.ylabel("Gender")

plt.title("Jumlah Gender DC Comics")

plt.show()
jenis_id = dc.groupby(['ID'])['name'].count()

jenis_id 



plt.barh(jenis_id.index, jenis_id)

plt.ylabel("Identitas")

plt.xlabel("Jumlah")

plt.title("Identitas Superhero DC Comics")

plt.show()
sifat_dc = dc.groupby(['ALIGN'])['name'].count()

sifat_dc



plt.bar(sifat_dc.index, sifat_dc)

plt.ylabel("Jumlah")

plt.xlabel("Sifat Karakter")

plt.title("Sifat Karakter Superhero DC Comics")

plt.show()
mata_dc = dc.groupby(['EYE'])['name'].count()

mata_dc 



plt.barh(mata_dc .index, mata_dc )

plt.ylabel("Warna Mata")

plt.xlabel("Jumlah")

plt.title("Warna Mata Superhero DC Comics")

plt.show()
rambut_dc = dc.groupby(['HAIR'])['name'].count()

rambut_dc 



plt.barh(rambut_dc .index, rambut_dc )

plt.ylabel("Warna Rambut")

plt.xlabel("Jumlah")

plt.title("Warna Rambut Superhero DC Comics")

plt.show()
lgbt_dc = dc.groupby(['GSM'])['name'].count()

lgbt_dc 



plt.barh(lgbt_dc .index, lgbt_dc )

plt.ylabel("Jumlah")

plt.xlabel("Gender")

plt.title("Kelainan Gender Superhero DC Comics")

plt.show()
status_dc = dc.groupby(['ALIVE'])['name'].count()

status_dc 



plt.barh(status_dc .index, status_dc )

plt.ylabel("Jumlah")

plt.xlabel("Status")

plt.title("Status Hidup/Mati Superhero DC Comics")

plt.show()
tahun_dc = dc.groupby(['YEAR'])['name'].count()

tahun_dc 



plt.fill(tahun_dc .index, tahun_dc )

plt.ylabel("Jumlah")

plt.xlabel("Tahun")

plt.title("Tahun terbit Superhero DC Comics")

plt.show()