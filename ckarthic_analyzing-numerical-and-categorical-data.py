#Load Libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import random
#Load Titanic Dataset

titanic = pd.read_csv("../input/train.csv") #sns.load_dataset('titanic')

titanic.isnull().sum()

titanic = titanic.fillna(method='ffill')

print(titanic.shape)
print(titanic.dtypes)
print(titanic.head())
titanic.describe()
#single histogram

# magic command. sets up Matplotlib to display the plots inline that are just static images.

#%matplotlib notebook - This would have created interactive plots

%matplotlib inline 

plt.hist(titanic['Age']) 

plt.title("Passengers By Age")

plt.xlabel("Passenger Age")

plt.ylabel("Passenger Count")

#hollow histogram

agebins = np.linspace(0,80,20) # we can specify the number of bins, there by controlling the shape of the historgram. 

plt.hist(titanic[titanic.Sex == 'male'].Age, agebins, alpha = 0.5, label = 'male' )

plt.hist(titanic[titanic.Sex == 'female'].Age, agebins, alpha = 0.5, label = 'female')

plt.legend(loc = 'upper right')

plt.title("Passengers By Age and Gender")

plt.xlabel("Passenger Age")

plt.ylabel("Passenger Count")
#single boxplot

plt.boxplot(titanic['Age'])

plt.title("Passengers By Age")

plt.ylabel("Age")
#paired boxplot 1

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-whitegrid')

titanic.boxplot(column = 'Age', by = 'Survived', figsize= (6,5))

plt.title('Survival By Age')

plt.ylabel("Passenger Age")
#paired boxplot 2



with sns.axes_style(style='ticks'):

    g = sns.factorplot("Sex", "Age", "Pclass", data=titanic, kind="box")

    g.set_titles("Passenger Age across gender and class traveled")

    g.set_axis_labels("Sex", "Age");
#paired boxplot 3

titanic['alive'] = np.where(titanic.Survived == 1, 'yes','no')

with sns.axes_style(style='ticks'):

    g = sns.factorplot("Sex", "Age", "alive", data=titanic, kind="box")

    g.set_axis_labels("Sex", "Age");
#scatterplot

plt.scatter(x=titanic.Age, y=titanic.Fare)

plt.title("Passenger Age Vs fare")

plt.xlim(0,80)

plt.ylim(0,500)

#scatterplot 2 

pclass = titanic.Pclass.astype('category')

plt.scatter(x=titanic.Age, y=titanic.Fare, c=pclass, alpha = 0.5, cmap = 'viridis')

plt.xlim(0,80)

plt.ylim(0,500)

plt.colorbar(ticks = range(4) ,label = 'Passenger class')
#frequency table1 - passengers survived vs died

pd.crosstab(titanic.Survived, columns='count')
'''frequency table 2 - Number of passengers across the age distribution. Age is a numeric column. Hence it is converted into

categorical data by using pd.cut function. This creates a set number of bins for age'''

agebins = pd.cut(titanic.Age, 5)

agetab =  pd.crosstab(agebins, columns='count')

agetab.index = ["Under 16", "16 to 32", "32 to 48", "48 to 64", "over 64"]

agetab
#contingency table 1 - Passenger survival across the class of travel

classbysurvivalTab =   pd.crosstab(titanic.Pclass, columns=titanic.alive, margins= True)

classbysurvivalTab.columns = ["Died", 'Survived', 'RowTotal']

classbysurvivalTab.index = ["First Class", "Second Class", "Third Class", "ColTotal"]

classbysurvivalTab
#contingency table 2 - Data presented as proportions. This provides a better visual understanding than raw numbers above

classbysurvivalTab.div(classbysurvivalTab["RowTotal"], axis=0)# /classbysurvivalTab.ix["ColTotal"]