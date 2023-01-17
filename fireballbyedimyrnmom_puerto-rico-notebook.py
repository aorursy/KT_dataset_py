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



Sci=pd.read_csv('../input/puerto-rico-dataset/science-and-technology_pri.csv')

Crime=pd.read_csv('../input/puerto-rico-dataset/Incidencia_Crime_Map.csv')

climate=pd.read_csv('../input/puerto-rico-dataset/climate-change_pri.csv')

info=pd.read_csv('../input/puerto-rico-dataset/QuickFacts Feb-20-2020.csv')
import numpy as np

import copy

import matplotlib.pyplot as plt

%matplotlib inline
#basic information

info
a=info.drop(['Fact Note', 'Value Note for Puerto Rico'], axis=1)

b= a.head(65)

b
#small dataset

Sci
Sci.info()
#remove the first row

Sci = Sci.iloc[1:]

Sci.head(2)
plt.bar(Sci["Year"],Sci["Indicator Name"],color=["green","blue","pink"])

 

plt.xlabel("Year",color="blue")

plt.ylabel("Indicator Name",color="green")

plt.title("Compare",color="green")

plt.show()
y=Sci["Indicator Name"]

x=Sci["Year"]

def lineplot(x, y, x_label="", y_label="", title=""):

    # Create the plot object

    _, ax = plt.subplots()



        # alpha=transparency of the line

    ax.plot(x, y, lw = 2, color = '#539caf', alpha = 1)



    # Label the axes and provide a title

    ax.set_title(title)

    ax.set_xlabel(x_label)

    ax.set_ylabel(y_label)



print(lineplot(x,y))
#large dataset

Crime
Crime.info()
Crime['Area Policiaca']=Crime['Area Policiaca'].astype('category')

Crime['Area']=Crime['Area Policiaca'].cat.codes

Crime.head(2)
Crime.describe()
Crime.plot.scatter(y="Delito", x='Area', title='Crime by Area', color='purple')
#mid-size dataset

climate
climate.info()
#remove the first row

climate = climate.iloc[1:]

climate.head(2)
climate['Value']=climate['Value'].astype('float')
climate['Year']=climate['Year'].astype(int)
climate.describe()
climate.corr()
import seaborn as sns

x=climate['Year']

y=climate['Value']

sns.distplot(x)
sns.distplot(y)
sns.pairplot(climate, hue='Indicator Name', size=10)