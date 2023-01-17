# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
data =pd.read_csv('../input/Accidents0515.csv')
data.info()
data.corr()
#correlation map
f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt='.1f',ax=ax)
data.head(10)
data.columns
#Line Plot
data.Number_of_Casualties.plot(kind='line',color ='g',label='Number_of_Casualties',linewidth=1,alpha=0.5,grid=True,linestyle= ':')
data.Number_of_Vehicles.plot(color='blue',label='Araç Sayısı',linewidth=1,alpha=0.5,linestyle='-.')
plt.legend(loc='upper_right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title(' Kayıp Araç ve İnsan Sayısı')
plt.show()
#Scatter Plot
data.plot(kind='scatter',x='Number_of_Casualties',y='Number_of_Vehicles',alpha=0.5,color='red')
plt.xlabel('Number_of_Casualties')
plt.ylabel('Number_of_Vehicles')
plt.title('Araç Kaybı ve Insan Kaybı')
plt.show()
#Histogram
data.Location_Easting_OSGR.plot(kind='hist',bins=100,figsize=(10,10))
plt.show()
data[np.logical_and(data['Location_Easting_OSGR']>308738.0 , data['Location_Northing_OSGR']>178970.0)]
