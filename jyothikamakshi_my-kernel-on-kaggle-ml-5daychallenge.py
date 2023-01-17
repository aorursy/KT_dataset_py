import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#mydata = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='encoding')

#mydata = pd.read_csv('../input/conversionRates.csv')

mydata = pd.read_csv('../input/multipleChoiceResponses.csv',encoding ='unicode-escape')

mydata.head()
##Size of my data set

mydata.shape
##To get the number of null values in each column

mydata.isnull().sum()
mydata.describe()
test = mydata.Country.value_counts()

test =pd.DataFrame(test)

import matplotlib.pyplot as plt

import numpy as np

test =test.sort_values('Country',ascending=False)[:10]

%matplotlib inline

width = 1/1.5

plt.xticks(rotation=90)

plt.bar(test.index, test.Country, width, color="green")

import seaborn as sns



%matplotlib inline

plt.xticks(rotation=90)

sns.barplot(test.index, test.Country)

plt.ylabel('Count')

