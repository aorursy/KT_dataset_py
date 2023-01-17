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
dataset = pd.read_csv('../input/for-watercup-analysis/maharastra population districtwise - Sheet1.csv', index_col = 'District')

dataset  = dataset.drop('s.no',axis = 1)

dataset.head()
#We will try  the whole data and see what we will get
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (40,10))

sns.lineplot(data = dataset)

plt.title('overall dataset graph ')
#you can see there is a lot of variations in the dataset. It is because of scale between the columns. Some data is too high

# To bring the complete data into equal scale we need to apply feture scaling. But here it is not necessary as we will focus 1,2 columns



#Lets see by plotiing population location wise
plt.figure(figsize = (40,18))

sns.barplot(x = dataset.index , y  = dataset['population_2011'])

plt.title("Population of 2011 census")
plt.figure(figsize = (40,15))

sns.barplot(x = dataset.index , y = dataset['population_2001'])

plt.title('Population of 2001 census')
#Now we will plot the sexration trend.
plt.figure(figsize = (40,15))

plt.title("Sexration of according to location")

sns.lineplot(data = dataset['Sex ratio(per 1000 boys)'])
#Further movement

#For this we will use the seaborn map which directly gives us the realtion between these two terms.

#If there is relation between these two then it will datker in color. 
plt.figure(figsize = (15,10))

sns.heatmap(dataset.corr(), annot = True)
#As you can see there is not much realtion between these paramters. 



#but there is a very good realtion between the population and Area(sq km). so here we found out something logical



#Now we will use different plot to dig in more deeper in order to understand graphs more deeply or to see is there any relation between

#any column

plt.figure(figsize = (15,10))

sns.jointplot(x = dataset['Area(sq km)'], y = dataset['population_2011'], kind = 'kde')
#You can see from above figure the density is more in the range of 5000 - 10000 Area sqkm. That is the place where population increased

#Lets do the same for 2001 census
plt.figure(figsize = (15,10))

sns.jointplot(x = dataset['Area(sq km)'], y = dataset['population_2001'], kind = 'kde')
#Yes the range is same for both of the dataset. According to dataset, area actaully affects the poputalion of the specific region



#Now let's see is there any relation between literacy and population which we failed to see in heatmap
plt.figure(figsize = (15,10))

sns.jointplot(x = dataset['literacy'], y = dataset['population_2011'], kind = 'kde')
#So accoring to dataset if the literacy increases, population also increases. The area is dense between the range 70--90 and you

#can see a little cloud at 85 -- 90 which shows literacy direcly to propotional to population.



#so this was the visualization of the dataset in a very simple way
plt.figure(figsize=(50,20))

plt.pie(dataset['Area(sq km)'],labels=dataset.index,autopct='%1.1f%%')

explode=[0,0,0.1,0,0]



plt.title('Areawise distribution of land',color='black',fontsize=30)

plt.show()
#because of so much of data,it looks weird. However you can try your own way.