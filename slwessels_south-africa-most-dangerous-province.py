import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

#%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Crimes = pd.read_csv("../input/SouthAfricaCrimeStats_v2.csv")#,converters={'a':str})#,header=0)

ProvincePopulation = pd.read_csv("../input/ProvincePopulation.csv",index_col=0)

ProvinceCrime = Crimes.groupby("Province").sum()

Provinces = Crimes["Province"].unique()
ProvinceCrime.transpose().plot(kind='line',linewidth=4,figsize= (5,5), marker='o')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Provincial Crimes: All Provinces')
for Province in Provinces:

    ProvinceCrime.ix[Province].plot(kind='bar',alpha=.75, rot=90,figsize=(5,5))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title('Provincial Total Crimes: '+Province)

    plt.show()
NormalisedProvinceCrime = ProvinceCrime.div(ProvincePopulation.Population, axis='index')
NormalisedProvinceCrime.transpose().plot(kind='line',linewidth=4,figsize= (5,5), marker='o')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title("Normalised Crime per province")
for Province in Provinces:

    NormalisedProvinceCrime.ix[Province].plot(kind='Bar',linewidth=4,figsize= (5,5))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title("Normalised Crime for "+ Province)

    plt.show()
NormalisedProvinceCrime.transpose().describe()