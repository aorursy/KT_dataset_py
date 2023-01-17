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



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

#import matplotlib

#matplotlib.style.use('ggplot')



#%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/SouthAfricaCrimeStats_v2.csv",converters={'a':str})#,header=0)
df.head(5)
#Some issues with negative stats have been highlighted by MarcDeveaux. I believe that all crime insidents should be positive

#Let's force all crime counts to positive number 

#df[df.columns[df.dtypes != np.object]] = df[df.columns[df.dtypes != np.object]].abs()
Data = df[['Province', 'Station',

       'Category', '2005-2006', '2006-2007', '2007-2008', '2008-2009',

       '2009-2010', '2010-2011', '2011-2012', '2012-2013', '2013-2014',

       '2014-2015']]

       
Crimes_Province = df.groupby(['Province'])['2005-2006','2006-2007','2007-2008','2008-2009',

   '2009-2010','2010-2011','2011-2012','2012-2013','2013-2014','2014-2015'].sum().plot(kind='bar',alpha=0.75, rot=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Crime per province clustered by')
gauteng = df[df['Province']=='Gauteng']

myAreas = gauteng[gauteng['Station'].isin(['Linden','Parkview','Randburg'])] 
#gauteng['Station'].unique()

#myAreas
Crimes_Station = myAreas.groupby(['Station'])['2005-2006','2006-2007','2007-2008','2008-2009',

   '2009-2010','2010-2011','2011-2012','2012-2013','2013-2014','2014-2015'].sum().plot(kind='bar',alpha=0.75, rot=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Crime per province clustered by')
#Taking a closer look at my Station area.



gauteng.groupby(['Station']).sum().transpose().plot(kind='line',alpha=0.75, rot=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Crime per Station over period')
Data.groupby(['Province']).sum().transpose().plot(kind='line',linewidth=1, rot=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Crime per province over period')

#sns.barplot(data=Data.groupby(['Province']).sum().transpose())
Linden = Data[Data['Station']=="Linden"]
#Taking a closer look at my Station area.



Linden.groupby(['Category']).sum().transpose().plot(kind='line',alpha=0.75, rot=90)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Crime per province over period')