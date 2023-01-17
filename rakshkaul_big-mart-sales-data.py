# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dftr=pd.read_csv('../input/bigmart-sales-data/Train.csv')

dfts=pd.read_csv('../input/bigmart-sales-data/Test.csv')
dftr.head()
for x in dftr.columns:

    print(x,len(dftr[x].unique()))
dftr.describe()
outlet = dftr["Item_Identifier"].value_counts()

print(outlet)
dftr['Item_Fat_Content'].replace(to_replace=['reg'],value='Regular',inplace=True)
dftr['Item_Fat_Content'].replace(to_replace=['low fat','LF'], value='Low Fat',inplace=True)
dftr['Item_Fat_Content'].unique()
dftr['Item_Fat_Content'].value_counts()
dftr['Item_Outlet_Sales'].plot.hist(grid=True,rwidth=.5,bins=6)
dftr['Item_Visibility'].plot.hist(grid=True,bins=5,rwidth=0.9,color='blue')

plt.xlabel('Product Visibility')

plt.ylabel('No. of Products')

plt.title('Frequency Distribution Chart For Product Visibility')
cat_fea = dftr.select_dtypes(include=[np.object])

cat_fea.describe()
cat_var = [x for x in dftr.dtypes.index if dftr.dtypes[x]=='object']

for x in cat_var:

    print('Freq of :',x)

    print(dftr[x].value_counts())

    print("")
plt.figure(3)

plt.subplot(131)

dftr.Item_Type.value_counts().plot(kind='bar',title='Item Type',figsize=(15,5))

plt.subplot(132)

dftr.Outlet_Size.value_counts().plot(kind='bar',title='Outlet Size',figsize=(15,5))

plt.subplot(133)

dftr.Outlet_Establishment_Year.value_counts().plot(kind='bar',title='Outlet Year',figsize=(15,5))
plt.figure(3)

plt.subplot(131)

dftr.Outlet_Identifier.value_counts().plot(kind='bar',title='Outlet Identifier',figsize=(15,5))

plt.subplot(132)

dftr.Outlet_Type.value_counts().plot(kind='bar',title='Outlet Type',figsize=(15,5))

plt.subplot(133)

dftr.Outlet_Location_Type.value_counts().plot(kind='bar',title='Location Type',figsize=(15,5))
plt.figure(figsize=(10,9))



plt.subplot(311)

sns.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=dftr,palette='Set1')



plt.subplot(312)

sns.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=dftr,palette='Set1')



plt.subplot(313)

sns.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=dftr,palette='Set1')



plt.subplots_adjust(wspace = 0.2, hspace = 0.4, top=1.5)
plt.figure(figsize=(20,10))

plt.subplot(211)

sns.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',data=dftr,palette='Set1')

plt.subplot(212)

sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',data=dftr,palette = 'Set1')

plt.subplots_adjust(hspace = 0.9, top = 0.9)
dftr.apply(lambda x: sum(x.isnull()))
dftr.head(5)

dftr['Item_Outlet_Sales'].sum().round()
Tier1_Sales = dftr[dftr.Outlet_Location_Type == 'Tier 1'].Item_Outlet_Sales.sum().round()

Tier1_Stores = dftr[dftr.Outlet_Location_Type =='Tier 1'].Outlet_Location_Type.value_counts()

Tier2_Sales = dftr[dftr.Outlet_Location_Type == 'Tier 2'].Item_Outlet_Sales.sum().round()

Tier2_Stores = dftr[dftr.Outlet_Location_Type =='Tier 2'].Outlet_Location_Type.value_counts()

Tier3_Sales = dftr[dftr.Outlet_Location_Type == 'Tier 3'].Item_Outlet_Sales.sum().round()

Tier3_Stores = dftr[dftr.Outlet_Location_Type =='Tier 3'].Outlet_Location_Type.value_counts()

Sum = Tier1_Sales+Tier2_Sales+Tier3_Sales

Tier_Sales = [['Tier 1',Tier1_Stores,Tier1_Sales],['Tier 2',Tier2_Stores,Tier2_Sales],['Tier 3',Tier3_Stores,Tier3_Sales],]

pd.DataFrame(Tier_Sales,columns=['Outlet_Location','Stores','Sales'])
Tier1=(4482059/2388)*100

Tier2=(6472314/2785)*100

Tier3=(7636753/3350)*100

sales_contri = [['Tier1',Tier1],['Tier2',Tier2],['Tier3',Tier3]]

pd.DataFrame(sales_contri,columns=['Location','Sales Contribution'])
for x in dftr['Item_Type'].unique():

    print(x)

    print(dftr[dftr.Item_Type==x].Item_Outlet_Sales.sum().round())