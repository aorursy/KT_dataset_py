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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

dataset=pd.read_csv('../input/forest-cover-type-prediction/train.csv')
df=dataset.copy()
df.shape
df.info()
df.isna().sum()
df=df.drop(['Id'],axis=1)
df.head()
df.rename(columns={'Wilderness_Area1':'Rawah','Wilderness_Area2':'Neota',
'Wilderness_Area3': 'Comanche Peak','Wilderness_Area4' : 'Cache la Poudre'},inplace=True)
cover_types={1:'Spruce',2 :'L.Pine',3 :'P.Pine',4 : 'Willow',5 : 'Aspen', 6 : 'Douglas-fir',7: 'Krummholz'}
df=df.replace({'Cover_Type':cover_types})
df.skew()
df.groupby(['Cover_Type']).describe().T
g=sns.factorplot(x='Cover_Type',kind='count',data=df,color='darkseagreen')
g.set(title='Sampling distribution of cover types')
g.set_xticklabels(rotation=90)
df1=df.copy()
df1["Wild_area"] = df.iloc[:,10:14].idxmax(axis=1)
df1['Soil'] = df.iloc[:,14:54].idxmax(axis=1)
plt.figure(figsize=(10,5))
sns.countplot(x='Wild_area',data=df1,palette="Set3",hue='Cover_Type')
soil_columns=['Soil_Type'+str(i)for i  in range(1,41)]
abundance=[df[df[i]==1][i].count() for i in soil_columns]
num = [i for i in range(1,41)]
plt.figure(figsize=(10,5))
g=sns.barplot(x=num,y=abundance,palette='ch:.25')
g.set(title="Abundance of Soil Types",ylabel="No. of patches",xlabel="Soil Type")
sns.despine()
dd=df.groupby(['Cover_Type'])[soil_columns].sum()
dd.T.plot(kind = 'bar', figsize = (18,10),stacked=True)
plt.title('Abundance of Soil type with respect to cover types',fontsize=15)
def distp(feature,a,b):
    return sns.distplot(df[feature],color=a,ax=axs[b],kde=False)
fig, axs = plt.subplots(ncols=7,figsize=(22,10))
fig.suptitle("Distribution of observations",fontsize='20')

distp('Elevation','green',0)
distp('Aspect','turquoise',1)
distp('Slope','yellow',2)
distp('Horizontal_Distance_To_Hydrology','navy',3)
distp('Vertical_Distance_To_Hydrology','brown',4)
distp('Horizontal_Distance_To_Roadways','orange',5)
distp('Horizontal_Distance_To_Fire_Points','purple',6)
df1.drop(df1.columns[14:54], axis=1, inplace=True)
df1.drop(df1.columns[10:14],axis=1,inplace=True)
sns.pairplot(data=df1,hue='Cover_Type',palette='Set1')
fig, axs = plt.subplots(ncols=3,figsize=(15,4))
fig.suptitle("Positive Correlations",fontsize='20')

sns.lineplot(x= "Aspect",y="Hillshade_3pm",data=df,color='green',ax=axs[0])
sns.lineplot(x= "Hillshade_Noon",y="Hillshade_3pm",data=df,color='green',ax=axs[1])
sns.lineplot(x= "Horizontal_Distance_To_Hydrology",y="Vertical_Distance_To_Hydrology",color="green",data=df,ax=axs[2])

fig, axs = plt.subplots(ncols=3,figsize=(15,4))
fig.suptitle("Negative Correlations",fontsize='20')

sns.lineplot(x= "Hillshade_3pm",y="Hillshade_9am",data=df,color='red',ax=axs[0])
sns.lineplot(x= "Slope",y="Elevation",data=df,color='red',ax=axs[1])
sns.lineplot(x= "Hillshade_Noon",y="Slope",data=df,color='red',ax=axs[2])
plt.figure(figsize=(18,12))
corr_matrix=df1.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix,annot=True ,cbar = True,cmap="YlGnBu",mask=mask)
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Elevation",fontsize='20')
sns.swarmplot(x= "Cover_Type",y="Elevation",data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y="Elevation",data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Aspect",fontsize='20')
sns.boxplot(x= "Cover_Type",y="Aspect",data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y="Aspect",data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
degrees = df1['Aspect']
radians = np.deg2rad(degrees)

bin_size = 20
a , b=np.histogram(degrees, bins=np.arange(0, 360+bin_size, bin_size))
centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='polar')
ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.title('Aspect')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Slope",fontsize='20')


sns.violinplot(x= "Cover_Type",y="Slope",data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y="Slope",data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Horizontal Distance To Hydrology",fontsize='20')


sns.swarmplot(x= "Cover_Type",y='Horizontal_Distance_To_Hydrology',data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y='Horizontal_Distance_To_Hydrology',data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Vertical Distance To Hydrology",fontsize='20')


sns.swarmplot(x= "Cover_Type",y='Vertical_Distance_To_Hydrology',data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y='Vertical_Distance_To_Hydrology',data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Horizontal Distance To Roadways",fontsize='20')
sns.violinplot(x= "Cover_Type",y='Horizontal_Distance_To_Roadways',data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y='Horizontal_Distance_To_Roadways',data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')
fig, axs = plt.subplots(ncols=2,figsize=(15,6))
fig.suptitle("Horizontal Distance To Fire Points",fontsize='20') 
sns.boxplot(x= "Cover_Type",y='Horizontal_Distance_To_Fire_Points',data=df1,palette='Set2',ax=axs[0])
sns.swarmplot(x= "Wild_area",y='Horizontal_Distance_To_Fire_Points',data=df1,palette="Set2",ax=axs[1],hue='Cover_Type')