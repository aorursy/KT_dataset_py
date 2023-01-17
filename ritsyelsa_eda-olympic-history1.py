# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
master_data=pd.read_csv('../input/athlete_events.csv')
master_data
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(20,8)})
boxplot1=sns.boxplot(x='Year', y='Age', hue='Sex', data=master_data)
boxplot1.set_title('Variation in Age through years according to sex')
plt.xlabel('Years')
plt.ylabel('Age')
plt.show()
cross_year_sex=pd.crosstab(master_data['Year'],master_data['Sex'])
sns.set(rc={'figure.figsize':(20,8)})
plot2=cross_year_sex.plot.bar()
plot2.set_title('Boxplot of Male to Female participants through years')
plt.show()
medals_num=pd.crosstab(master_data['Year'],master_data['Medal'])
medals_num
plot3=medals_num.plot.bar()
plot3.set_title('Boxplot of medals won through years')
plt.show()
#The code will work only on jupyter notebook
#Not on the kaggle kernel Notebook
# import numpy as np
# years=master_data['Year']
# years=np.sort(years)
# years=list(years)

# import matplotlib.pyplot as plt
# import seaborn as sns

# my_dpi=96
 
# for i in years:
#     fig = plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
#     sub1=master_data[master_data['Year']==i]
#     sub2=sub1[sub1['Medal']=='Gold']
#     sub3=sub2.groupby(['NOC'])['Medal'].count()
#     sub3.reset_index()
#     sub3.plot.bar()
#     plt.title("Gold medals won by different countries through year {}".format(i))
#     plt.xlabel("Countries")
#     plt.ylabel("No of gold medals")
#     filename='step'+str(i)+'.png'
#     plt.savefig(filename, dpi=96)
#     plt.gca()

# #The result for the same after animations done with the help of R is a gif image 
# #and is provided


# #In a similar way the graphs for silver medals can be generated
#Barplot for bronze,Gold and Silver medals for each country
#(Only for those countries which have atleast one medal in each category)
medals_countries=pd.crosstab(master_data['NOC'],master_data['Medal'])
sns.set(rc={'figure.figsize':(15,40)})
set1=medals_countries.loc[medals_countries['Bronze'] >0 ]
set2=set1.loc[medals_countries['Gold'] >0 ]
set3=set2.loc[medals_countries['Silver'] >0 ]
set3
plot4=set3.plot.barh()
plot4.set_title('Boxplot of medals won by each country')
plt.show()
# Number of USA medals over time
data_USA=master_data[master_data['NOC']=='USA']
# data_USA
sns.set(rc={'figure.figsize':(20,10)})
USA_medals=pd.crosstab(data_USA['Year'],data_USA['Medal'])
plot5=USA_medals.plot.bar()
plot5.set_title('Medals won by USA over time')
plt.show()