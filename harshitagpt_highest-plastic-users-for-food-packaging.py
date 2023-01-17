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
%matplotlib inline
import matplotlib.pyplot as plt
food_data=pd.read_csv("../input/FoodFacts.csv",low_memory=False)
#Searching for manufacturers which use plastic stuffs for packing food
df=food_data[['manufacturing_places','packaging_tags']][(food_data['packaging_tags']=='plastique')|
                                                                  (food_data['packaging_tags']=='plastic')]
df=df.dropna()
print(df.head(20))
#Calculating which country uses what extent of platic stuffs
data=df['manufacturing_places'].value_counts(sort=True,dropna=False)
print(data.head(10))
#This indicates that Australia is the biggest plastic user for food packaging
#Defining a new column stating the extent of usage of plastic as 1
df['value']=1

def plast(country):
    return df[df.manufacturing_places == country].value.sum()
#Plastic Extent for some of the highest users
fr_plast=plast('France')
ge_plast=plast('Germany')
au_plast=plast('Australia')
us_plast=plast('United States')
iy_plast=plast('Italy')
al_plast=plast('Allemagne')
ch_plast=plast('China')
uk_plast=plast('United Kingdom')

countries=['FR','GE','AU','US','IY','AL','CH','UK']
plastic=[fr_plast,ge_plast,au_plast,us_plast,iy_plast,al_plast,ch_plast,uk_plast]
ypos=np.arange(len(countries))
plt.bar(ypos,plastic,align='center',alpha=0.5,facecolor='r')
plt.xticks(ypos, countries)
plt.annotate('Biggest user',xy=(3,146),xytext=(5,120),arrowprops=dict(facecolor='blue'))
plt.show()
