# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
codes=pd.read_excel('/kaggle/input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.xlsx',sheet_name='Country Codes')

medals=pd.read_excel('/kaggle/input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.xlsx',sheet_name='Medalists')

codes.head()
medals.head()
turkish=medals[medals["Country"]=="Turkey"]

turkish.head()
turkish.groupby("Year")["Medal"].count()
import seaborn as sns

import matplotlib.pyplot as plt



ax=sns.countplot(data=turkish,x="Discipline")

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()

plt.show()

turkish["Athlete"].value_counts()
sns.countplot(data=turkish,x="Gender")
men=turkish["Gender"].value_counts()[0]

wo=turkish["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
greece=medals[medals["Country"]=="Greece"]

sns.countplot(data=greece,x="Gender")
men=greece["Gender"].value_counts()[0]

wo=greece["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
usa=medals[medals["Country"]=="United States"]

sns.countplot(data=usa,x="Gender")
men=usa["Gender"].value_counts()[0]

wo=usa["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
usa=medals[medals["Country"]=="Iran"]

sns.countplot(data=usa,x="Gender")
deut=medals[medals["Country"]=="Germany"]

sns.countplot(data=deut,x="Gender")
men=deut["Gender"].value_counts()[0]

wo=deut["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
sns.countplot(data=medals,x="Gender")
men=medals["Gender"].value_counts()[0]

wo=medals["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
oly2012=[{"Servet Tazegül","Men"},

{"Nur Tatar","Women"},

{"Rıza Kayaalp","Men"}]

oly2012
oly2016=[{"Taha Akgül","Men"},

{"Daniyar Ismayilov","Men"},

{"Rıza Kayaalp","Men"},

{"Selim Yaşar","Men"},

{"Cenk İldem","Men"},

{"Yasmani Copello","Men"},

{"Soner Demirtaş","Men"},

{"Nur Tatar","Women"}]

oly2016
turkishwith2012=turkish.append(pd.DataFrame({"Athlete":["Servet Tazegül","Nur Tatar","Rıza Kayaalp"],"Gender":["Men","Women","Men"]}))
turkishwith12016=turkishwith2012.append(pd.DataFrame({"Athlete":["Taha Akgül","Daniyar Ismayilov","Rıza Kayaalp","Selim Yaşar",

                                                                "Cenk İldem","Soner Demirtaş","Yasmani Copello","Nur Tatar"

                                                                ],"Gender":["Men","Men","Men","Men","Men","Men","Men","Women"]}))
sns.countplot(data=turkishwith12016,x="Gender")
men=turkishwith12016["Gender"].value_counts()[0]

wo=turkishwith12016["Gender"].value_counts()[1]

print("ratio 1: ",men/wo)

print("ratio 2: ",wo/men)
turkishwith12016.to_csv("turkeymedalsall.csv",index=False)