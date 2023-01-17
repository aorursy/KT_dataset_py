# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

import missingno as msno

%matplotlib inline

import warnings

warnings.simplefilter(action='ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
car=pd.read_csv('/kaggle/input/cardataset/data.csv')
car.head()
plt.style.use('seaborn')

fig= plt.figure(figsize=(16,10))



Market=car['Market Category'].value_counts().head(8).to_frame()

market= Market.style.background_gradient(cmap='Reds')

market



plt.style.use('seaborn')

sns.set_style('whitegrid')

plt.figure(figsize=(15, 3))

car.isnull().mean()



allna = (car.isnull().sum() / len(car))*100

allna = allna.drop(allna[allna == 0].index).sort_values()

plt.figure(figsize=(8, 4))

allna.plot.barh(color=('red', 'black'), edgecolor='black')

plt.title('Missing values percentage per column', fontsize=15, weight='bold' )

plt.xlabel('Percentage', weight='bold', size=15)

plt.ylabel('Features with missing values', weight='bold')

plt.yticks(weight='bold')

plt.show()
NA=car[['Engine Fuel Type','Engine HP', 'Engine Cylinders', 'Number of Doors', 'Market Category']]
NAcat=NA.select_dtypes(include='object')

NAnum=NA.select_dtypes(exclude='object')

print('We have :',NAcat.shape[1],'categorical features with missing values')

print('We have :',NAnum.shape[1],'numerical features with missing values')
NAnum.head(3)
car['Engine HP']=car['Engine HP'].fillna(method='ffill')

car['Engine Cylinders']=car['Engine Cylinders'].fillna(method='ffill')

car['Number of Doors']=car['Number of Doors'].fillna(method='ffill')
NAcat.head(3)
car['Engine Fuel Type']=car['Engine Fuel Type'].fillna(method='ffill')

car['Market Category']=car['Market Category'].fillna('Crossover')
car.isnull().sum().sort_values(ascending=False).head()
carr=car['Make'].value_counts().head(5).to_frame()

m= carr.style.background_gradient(cmap='Blues')

colors=['blue','red','yellow','green','brown']

labels= ['Chevrolet','Ford','Volkswagen','Toyota','Dodge']

sizes= ['1123','881','809','746','626']

explode=[0.1,0.1,0.1,0.1,0.1]

values=car['Make'].value_counts().head(5).to_frame()



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=None,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('TOP 5 Car brands in the dataset',color='black',fontsize=10)

plt.show()



modelp = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(10, 'MSRP')

modelp.groupby(['Make']).sum()['MSRP'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='MSRP',

                                                                     title='Top 10 expensive Car Brands')
Chevrolet=car[car['Make'].str.contains('Chevrolet')]

chev=Chevrolet.sort_values(by=['MSRP'],ascending=False).nlargest(6, 'MSRP')

chevmodel= chev.style.background_gradient(cmap='Greens')

chevmodel
sns.set({'figure.figsize':(20,10)})

VS=sns.barplot(x=car['MSRP'], y=car['Vehicle Style'])

plt.title('Vehicle Style and MSRP', weight='bold', fontsize=18)

plt.xlabel('MSRP', weight='bold',fontsize=14)

plt.ylabel('Vehicle Style', weight='bold', fontsize=14)

plt.xticks(weight='bold')

plt.yticks(weight='bold')



modelp1 = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nsmallest(10, 'MSRP')

modelp1.groupby(['Make']).sum()['MSRP'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='MSRP',

                                                                     title='Top 10 least expensive Car Brands')
F=car[car['Make'].str.contains('FIAT')]

FI=F.sort_values(by=['MSRP'],ascending=False).nsmallest(5, 'MSRP')

Fiat= FI.style.background_gradient(cmap='Oranges')

Fiat
car_corr=car.corr()

f,ax=plt.subplots(figsize=(12,7))

sns.heatmap(car_corr, cmap='viridis')

plt.title("Correlation between features", 

          weight='bold', 

          fontsize=18)

plt.show()
plt.figure(figsize=(15,5))

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

plt.scatter(x=car['Engine Cylinders'], y=car['Engine HP'], color='maroon', alpha=0.7)

plt.title('Engine Cylinders on Engine HP', weight='bold', fontsize=18)

plt.xlabel('Engine Cylinders', weight='bold',fontsize=14)

plt.ylabel('Engine HP', weight='bold', fontsize=14)

plt.xticks(weight='bold')

plt.yticks(weight='bold')





#first row sec col

ax1 = plt.subplot2grid((1,2), (0, 1))

sns.regplot(x=car['Engine HP'], y=car['MSRP'], color='maroon')

plt.title('Engine HP on Engine MSRP', weight='bold', fontsize=18)

plt.xlabel('MSRP', weight='bold',fontsize=14)

plt.ylabel('Engine HP', weight='bold', fontsize=14)

plt.xticks(weight='bold')

plt.yticks(weight='bold')



plt.show()
plt.figure(figsize=(15,5))

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

sns.regplot(x=car["Engine HP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})

plt.title('Highway MPG and Engine HP', weight='bold', fontsize=18)

plt.xlabel('Engine HP', weight='bold',fontsize=14)

plt.ylabel('Highway MPG', weight='bold', fontsize=14)

plt.xticks(weight='bold')

plt.yticks(weight='bold')



#first row sec col

ax1 = plt.subplot2grid((1,2), (0, 1))

sns.regplot(x=car["MSRP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})

plt.title('Highway MPG and MSRP', weight='bold', fontsize=18)

plt.xlabel('MSRP', weight='bold',fontsize=14)

plt.ylabel('Highway MPG', weight='bold', fontsize=14)

plt.xticks(weight='bold')

plt.yticks(weight='bold')


fig= plt.figure(figsize=(16,10))

#2 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((2,2),(0,0))

sns.boxplot(x=car['Number of Doors'], y=car['MSRP'],color='Red')

plt.title('Number of Doors', weight='bold', fontsize=14)



#first row sec col

ax1 = plt.subplot2grid((2,2), (0, 1))

plt.scatter(x=car['Driven_Wheels'], y=car['MSRP'], color='Orange')

plt.title('Driven Wheels', weight='bold', fontsize=14)







#Second row first column



ax1 = plt.subplot2grid((2,2), (1, 0))

sns.barplot(x=car['Transmission Type'], y=car['MSRP'])

plt.xticks(rotation=35)

plt.title('Transmission Type', weight='bold', fontsize=14)





#second row second column

ax1 = plt.subplot2grid((2,2), (1, 1))

sns.barplot(x=car['Vehicle Size'], y=car['MSRP'])



plt.yticks(weight='bold')

plt.title('Vehicle Size', weight='bold', fontsize=14)





plt.show()
carmodel = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(6, 'Popularity')

carmodel.groupby(['Make']).sum()['Popularity'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='Popularity',

                                                                     title='Top 5 popular Car Brands')
Ford=car[car['Make'].str.contains('Ford')]

Fordm=Ford.sort_values(by=['Popularity'],ascending=False).nlargest(3, 'MSRP')

Ford2= Fordm.style.background_gradient(cmap='Reds')

Ford2
EFT=car[['Engine Fuel Type',

      'Engine HP',

      'Engine Cylinders',

      'highway MPG',

         'MSRP',

        'city mpg']].groupby(['Engine Fuel Type']).agg('median').sort_values(by=['Engine HP'],ascending=False)

EFT1= EFT.style.background_gradient(cmap='Purples')

EFT1
