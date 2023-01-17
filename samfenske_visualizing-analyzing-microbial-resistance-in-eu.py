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
ecdc=pd.read_csv('/kaggle/input/ECDC_surveillance_data_Antimicrobial_resistance.csv')

ecdc
len(ecdc['RegionName'].value_counts())
ecdc['Population'].value_counts()
ecdc['Bacteria']=[ecdc['Population'][index].split('|',1)[0] for index in range(len(ecdc))]

ecdc['Antibiotic']=[ecdc['Population'][index].split('|',1)[1] for index in range(len(ecdc))]

ecdc=ecdc.drop(columns=['Population','HealthTopic'])

ecdc
ecdc['Bacteria'].value_counts()
ecdc['Antibiotic'].value_counts()
ecdc['Distribution'].value_counts()
ecdc['Category'].value_counts()
ecdc['CategoryIndex'].value_counts()
ecdc=ecdc.drop(columns='CategoryIndex')
ecdc['Value'].value_counts()
ecdc=ecdc[-ecdc['Value'].isin(['-'])].reset_index().drop(columns='index')

ecdc
type(ecdc['Value'][0])
ecdc['Value']=ecdc['Value'].astype(float)
#look at data for United Kingdom



uk=ecdc[ecdc['RegionName'].isin(['United Kingdom'])]

uk['Bacteria'].value_counts()
uk[uk['Bacteria'].isin(['Escherichia coli'])]['Antibiotic'].value_counts()
uk[uk['Bacteria'].isin(['Escherichia coli']) & uk['Antibiotic'].isin(['Aminoglycosides'])]['Category'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

p=sns.catplot(x="Bacteria", y="Value",hue='Antibiotic', data=uk,height=9,aspect=3).set_xticklabels(rotation=30)

plt.title('Scatter Plot of Microbial Resistance of Bacterium in the UK')
sns.catplot("Bacteria", data=uk, aspect=1.5, kind="count", color="b").set_xticklabels(rotation=60)

plt.title('Data points for each bacteria in UK')
p=sns.catplot(x="Bacteria", y="Value",hue='Antibiotic', data=ecdc,height=9,aspect=3).set_xticklabels(rotation=30)

sns.set_palette("Paired", 15)

p.set(yticks=[0,25,50,75,100])

plt.title('Scatter Plot of Microbial Resistance of Bacterium for full dataset')
#look at data for Ireland



ireland=ecdc[ecdc['RegionName'].isin(['Ireland'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="Bacteria", y="Value",hue='Antibiotic', data=ireland,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
#look at ecoli resistance by country



ecoli=ecdc[ecdc['Bacteria'].isin(['Escherichia coli'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="RegionName", y="Value",hue='Antibiotic', data=ecoli,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
#isolate aminoglycoside data



Aminoglycosides=ecdc[ecdc['Antibiotic'].isin(['Aminoglycosides'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="RegionName", y="Value",hue='Bacteria', data=Aminoglycosides,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
ecdc['Category'].value_counts()
gender=ecdc[ecdc['Category'].isin(['Male','Female'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="Bacteria", y="Value",hue='Category', data=gender,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
age=ecdc[-ecdc['Category'].isin(['Male','Female'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="Bacteria", y="Value",hue='Category', data=age,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
sns.catplot("Category", data=age, aspect=1.5, kind="count", color="b").set_xticklabels(rotation=60)
kids=age[age['Category'].isin(['5-18'])]

sns.set_palette("Paired", 15)

p=sns.catplot(x="Bacteria", y="Value",hue='Antibiotic', data=kids,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
sns.set_palette("Paired", 15)

p=sns.catplot(x="RegionName", y="Value",hue='Bacteria', data=kids,height=9,aspect=3).set_xticklabels(rotation=30)

p.set(yticks=[0,25,50,75,100])
resistant=kids[kids['Value']>90]

p=sns.catplot(x="Bacteria", y="Value",hue='Antibiotic', data=resistant,height=9,aspect=3).set_xticklabels(rotation=30)
counts=resistant['Bacteria'].value_counts()
sns.set_palette("Paired", 15)

sns.catplot("Bacteria", data=resistant, hue='Antibiotic',aspect=1.5, kind="count").set_xticklabels(rotation=60)