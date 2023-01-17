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
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

#Used to generate graphs inside the notebook
%matplotlib inline 

#Specify the path to data 
PathToData='../input/pakistan-education-performance-dataset/Consolidated (Educational Dataset).csv'

#Read data
data = pd.read_csv(PathToData)

#Displaying starting rows
print(data.head())
print(data.shape)
print(data.describe())
print(data.dtypes)
print(data.dtypes.unique())
null_columns=data.columns[data.isnull().any()]
print(data[null_columns].isnull().sum())
data[data.isnull().any(axis=1)][null_columns].head()
data=data.fillna(0)
data.head(3)# checking only 3 because Area contained Nan in the 3rd row which has now been replaced with 0
dropColumns=['Boundary wall, Building condition satisfactory, Drinking water and 2 more (clusters)','Show Sheet','Table of Contents','Color By Measure Name','Number of Records'
             ,'MeasureGroup 1 Measures','Color By Measure Value','Other Factors Measure Value','MeasureGroup 2 Measures','Country','Analysis Level Selector']

#Analysis Level Selector is removed because it is same as province

## Drop was not working due to an issue in data formatting
#data.drop(dropColumns, axis=1,inplace=True)

#Alternative to drop is to select other than the columns we want to drop
columns=[col for col in data.columns if col not in dropColumns]

#Columns that were selected
print(columns)

#Displaying the data
data=data[columns]
data


data2 = data.copy()
NonNumerical=[c for c in data.columns if data[c].dtype in ['O']]

print(NonNumerical)

for i in NonNumerical:# 2) iterate over non numerical columns

  # 3) removing % and dividing by 100
  if(data2[i][0][-1]=='%'):

    data2[i] = data2[i].map(lambda x: str(x)[:-1]) #Succssfully removes the percentage sign
    data2[i]=pd.to_numeric(data2[i],errors='coerce')
    # data2[i]=data2[i].astype('float64')##Gives error
    data2[i] = data2[i]/100
data = data2
print(data.head())
# data2=data.copy()
print(data.dtypes.unique())
import numpy as np
data.replace(0, np.nan, inplace=True)
data.head()
null_columns=data.columns[data.isnull().any()]
print(data[null_columns].isnull().sum())
Numerical=[c for c in data.columns if data[c].dtype not in ['O','object']]
Numerical = [c for c in Numerical if c not in ['No Facility'] ]
print(Numerical)

print(data[Numerical].dtypes.unique())
# data2= data.copy()#Copy of datamade
# data=data2.copy()
data[Numerical]
# data.to_csv (r'/content/drive/My Drive/Sem8/DS/Project/Processed(ContainsMissingValues).csv', index = False, header=True)
data[Numerical]=data[Numerical].ffill(axis = 0) 
data[Numerical]
# dropColumns=['Boundary wall, Building condition satisfactory, Drinking water and 2 more (clusters)','Show Sheet','Table of Contents','Color By Measure Name','Number of Records']
data = data[[c for c in data.columns if c not in ['No Facility']]]
"""
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
data[Numerical]=imp_mean.fit_transform(data[Numerical])
data[Numerical]
"""
"""
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)

data[Numerical] = imputer.fit_transform(data[Numerical])
data[Numerical]

"""


print('Cities belong to the following areas:\n\nAzad Jammu Kashmir(AJK), Balochistan, Federally Administered Tribal Areas(FATA),\nGilgit Baltistan(GB), Khyber Pakhtunkhwa(KP), Islamabd Capital Territory(ICT), Punjab, Sindh: ')
print(data.Province.unique())

print('\nData Collected of following cities:\n')
c=5
for i in data.City.unique():
  if(c==10):
    print()
    c=0
  print(i, end =", ")
  c+=1
import seaborn as sn
import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))

corrMatrix = data.corr()
# print (corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()
df_num = data.select_dtypes(include = ['float64', 'int64'])# func to select particular type columns
print(df_num.head())
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
df_num_corr = df_num.corr()['Retention score'][:-1] # -1 to ignore the Retention score column
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)#Descending sort
print("{} features are strongly correlated with Retention score. They are:\n{}".format(len(golden_features_list), golden_features_list))
corr = df_num.corr() # Finding correlations among columns after dropping saleprice
plt.figure(figsize=(12, 10))#adjusting image size

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
data2=data.copy()
#Seperating the 'year' column
data2=data[Numerical[:-1]]

data2_column_maxes = data2.max()
data2_max = data2_column_maxes.max()
data2 = data2 / data2_max

# data2
data3=data.copy()
data3[Numerical[:-1]]=data2.copy()
#Normalized data
data = data3
# data
data.groupby('Province')['Total number of schools'].mean().plot()
plt.title('Total number of Schools')
plt.show()

data.groupby('Province')['% Complete Primary Schools'].mean().plot()
plt.title('% Students Completing Primary Schools')
plt.show()

data.groupby('Year')['Drone attacks in Pakistan','Bomb Blasts Occurred'].mean().plot()
plt.title('Fall in number of Drone attacks and Bomb blasts')
plt.show()

data.groupby('Year')['Gender parity score'].mean().plot()
plt.title('Gender Parity score')
plt.show()
data.groupby('Year')['Global Terrorism Index - Pakistan'].mean().plot()
plt.title('Global Terrorism Index - Pakistan')
plt.show()
data.groupby('Year')['Pakistan Economic Growth'].mean().plot()
plt.title('Pakistan Economic Growth')
plt.show()
data.groupby('Year')['% Boys Enrolled','% Girls Enrolled'].mean().plot()
plt.title('% Student Enrolled')
plt.show()
data.groupby('Year')['Education score','Retention score', 'Learning score','Enrolment score'].mean().plot()
plt.title('Comparison of several Scores')
plt.show()