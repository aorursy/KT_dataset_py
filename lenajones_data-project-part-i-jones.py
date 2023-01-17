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
import pandas as pd

def_area_2004_2019 = pd.read_csv("../input/brazilian-amazon-rainforest-degradation/def_area_2004_2019.csv")

el_nino_la_nina_1999_2019 = pd.read_csv("../input/brazilian-amazon-rainforest-degradation/el_nino_la_nina_1999_2019.csv")

inpe_brazilian_amazon_fires_1999_2019 = pd.read_csv("../input/brazilian-amazon-rainforest-degradation/inpe_brazilian_amazon_fires_1999_2019.csv")
#AMOUNT OF FIRES FOR EACH REGION

def_area_2004_2019.head()
#Correlation Matrix



#Note: Ano/Estados represents "Year of occurence"

#"AM" represents "Deforested area in Amazonas state (km²)"

#"AMZ LEGAL" represents "Sum of deforested area in Brazil (km²)."



def_area_2004_2019[['Ano/Estados','AMZ LEGAL']].corr()
#from Ms. Lougheed's example and with help from Ariana



from scipy import stats

pearson_coef, p_value = stats.pearsonr(def_area_2004_2019['Ano/Estados'], def_area_2004_2019['AMZ LEGAL'])



#To what extent are the two columns are correlated?

print(pearson_coef)



#How certain are we about this correlation? Very sure, as the p-value < 0.01.

print(p_value)
#From Python Graph Gallery



import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



def_area_2004_2019 = pd.DataFrame(np.random.random((20,16)))



correlation_graph1 = sns.heatmap(def_area_2004_2019, xticklabels=False, yticklabels=False)



plt.title("Heatmap Displaying Correlation Between the Year and the Sum of Deforested Areas in Brazil (km²)", fontsize =15)



plt.xlabel('Years', fontsize = 10) 

plt.ylabel('Sum of Deforested Areas in Brazil (km²)', fontsize = 10) 



plt.show()

# libraries

import seaborn as sns

import matplotlib.pyplot as plt



sns.regplot(x=def_area_2004_2019['Ano/Estados'], y=def_area_2004_2019['AMZ LEGAL'], marker='o')

plt.show()

#SEVERITY OF WEATHER PATTERNS EACH YEAR

el_nino_la_nina_1999_2019.head()

#The comparisons are not numerical in this graph, so I will not look for correlations in growth/decay.
#SPECIFIC PLACEMENT OF REGIONS AND AMOUNTS OF FIRESPOTS IN EACH REGION

inpe_brazilian_amazon_fires_1999_2019.head()
inpe_brazilian_amazon_fires_1999_2019[['year', 'latitude', 'longitude', 'firespots']].corr()

#There's very little correlation...
year = inpe_brazilian_amazon_fires_1999_2019.loc[inpe_brazilian_amazon_fires_1999_2019['year'] == 2019][inpe_brazilian_amazon_fires_1999_2019['state'] == 'AMAZONAS']
import numpy as np

import matplotlib.pyplot as plt



height = year['firespots']

bars = year['state']

y_pos = np.arange(len(bars))

plt.figure(figsize=(15,10))

plt.rcParams.update({'font.size': 20})

plt.xlabel('Amount of Firespots')

plt.ylabel('State of Amazonas at Various Times Throughout 2019')

plt.title('Firespots in Amazonas, Brazil in 2019')

 

# Create horizontal bars

plt.barh(y_pos, height, color=['black'], edgecolor='red', linewidth=4)



# Create names on the y-axis

plt.yticks(y_pos, ' ')

 

# Show graphic

plt.show()
