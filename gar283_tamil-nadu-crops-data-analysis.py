# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# read csv file as pandas dataframe
data = pd.read_csv("../input/tn.csv")
# checking contents of file
data.head()
# information of dataframe
data.info()
data['Crop_Year'].value_counts().sort_values()
data['State_Name'].value_counts()
del data['State_Name']
# checking dataframe info after deletion of column
data.info()
data.District_Name.nunique()
data['Crop'].str.upper().nunique()
data['Season'].value_counts()
# Get current size of figure
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: current size
print("Current size:", fig_size)
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 20
plt.rcParams["figure.figsize"] = fig_size
data.groupby(['Crop'])['District_Name'].nunique().sort_values().plot(kind='barh')
plt.xlabel("Number of districts")
data.Production.max()
data[data.Production == 1250800000.0]
Coconut_data_2011 = data[(data.Crop == 'Coconut ') & (data.Crop_Year == 2011)][['District_Name','Area','Production']]
Coconut_data_2011['Production per hectare'] = Coconut_data_2011['Production']/Coconut_data_2011['Area']
Coconut_data_2011['Production per hectare'].plot(kind = 'bar')
plt.xticks( np.arange(31), (Coconut_data_2011['District_Name']) )
plt.title("Coconut production per hectare districtwise in Tamil Nadu")
#plt.xticks(Coconut_data_2011['District_Name'])
data.Production.min()
len(data[data.Production == 0.0])
data.Production.mean()
data[data.Production >= 910330.4]['Crop'].unique()
data[data.Production <= 910330.4]['Crop'].nunique()
ax = data[data.Crop == 'Sugarcane'].groupby('Crop_Year')['Production'].sum().div(100).plot()
data[data.Crop == 'Sugarcane'].groupby('Crop_Year')['Area'].sum().plot(ax=ax)
plt.xlabel("Year")
plt.ylabel("Production/100")
plt.legend(loc='best')
plt.title("Production of Sugarcane in Tamil Nadu in relation to area sown from 1997 to 2013")
#plt.ylabel("Sugarcane Production")

Tapioca_data = data[data.Crop == 'Tapioca'][['District_Name','Crop_Year','Production']]
ax = Tapioca_data.pivot(index='District_Name', columns='Crop_Year', values='Production').T.plot(kind='bar',subplots=True,layout=(8,4),legend=False)
