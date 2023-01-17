# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Loe andmed dataseti failist ‘cwurData.csv’

df = pd.read_csv("../input/cwurData.csv")

df
#Kuva tabelist read, mis käivad Eesti ülikoolide kohta

riik = df.country

df[riik == "Estonia"]
#Kuva keskmine hariduse kvaliteedi näitaja grupeerituna riikide kaupa

df1 = df.groupby("country")["quality_of_education"].mean()

df1
#Järjesta saadud andmed keskmise hariduse kvaliteedi näitaja järgi kahanevalt

df2 = pd.DataFrame(df1)

df2.sort_values("quality_of_education", ascending=False)
#Leida mitu korda iga riigi ülikoole tabelis esineb a) Leida täpsemalt ainult 2015. aasta tulemuste kohta

aasta = df.year

df[aasta == 2015].country.value_counts()