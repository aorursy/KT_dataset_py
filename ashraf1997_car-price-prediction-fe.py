import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
car=pd.read_csv('/kaggle/input/cardataset/data.csv')
car.head()
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
print(NAcat.shape[1],'categorical features with missing values')
print(NAnum.shape[1],'numerical features with missing values')
NAnum.head()
car['Engine HP']=car['Engine HP'].fillna(method='ffill')
car['Engine Cylinders']=car['Engine Cylinders'].fillna(method='ffill')
car['Number of Doors']=car['Number of Doors'].fillna(method='ffill')
NAcat.head()
car['Engine Fuel Type']=car['Engine Fuel Type'].fillna(method='ffill')
plt.style.use('seaborn')
fig= plt.figure(figsize=(16,10))

market=car['Market Category'].value_counts().head(5).to_frame()
m= market.style.background_gradient(cmap='Blues')
m
car['Market Category']=car['Market Category'].fillna('Crossover')
car.isnull().sum().sort_values(ascending=False).head()
modelp = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(10, 'MSRP')
modelp.groupby(['Make']).sum()['MSRP'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='MSRP',title='Top 10 expensive Car Brands')
Chev=car[car['Make'].str.contains('Chevrolet')]
ch=Chev.sort_values(by=['MSRP'],ascending=False).nlargest(5, 'MSRP')
chevmodel= ch.style.background_gradient(cmap='Greens')
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
sns.heatmap(car_corr, cmap='viridis')
plt.title("Correlation between features", 
          weight='bold', 
          fontsize=18)
plt.show()
plt.figure(figsize=(15,5))
plt.scatter(x=car['Engine Cylinders'], y=car['Engine HP'], color='maroon', alpha=0.7)
plt.title('Engine Cylinders on Engine HP', weight='bold', fontsize=18)
plt.xlabel('Engine Cylinders', weight='bold',fontsize=14)
plt.ylabel('Engine HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
plt.figure(figsize=(15,5))
sns.regplot(x=car['Engine HP'], y=car['MSRP'], color='red')
plt.title('Engine HP on MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Engine HP', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
plt.figure(figsize=(15,5))
sns.regplot(x=car['Engine Cylinders'], y=car['MSRP'], color='red')
plt.title('Engine Cylinders on MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Engine Cylinders', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.show()
plt.figure(figsize=(15,5))
sns.regplot(x=car["Engine HP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('Highway MPG and Engine HP', weight='bold', fontsize=18)
plt.xlabel('Engine HP', weight='bold',fontsize=14)
plt.ylabel('Highway MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.figure(figsize=(15,5))
sns.regplot(x=car["Engine HP"], y=car["city mpg"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('City MPG and Engine HP', weight='bold', fontsize=18)
plt.xlabel('Engine HP', weight='bold',fontsize=14)
plt.ylabel('City MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.figure(figsize=(15,5))
sns.regplot(x=car["MSRP"], y=car["highway MPG"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('Highway MPG and MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Highway MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.figure(figsize=(15,5))
sns.regplot(x=car["MSRP"], y=car["city mpg"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('City MPG and MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('City MPG', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.style.use('seaborn')
fig= plt.figure(figsize=(16,10))
#first row, first col
ax1 = plt.subplot2grid((2,2),(0,0))
sns.boxplot(x=car['Number of Doors'], y=car['MSRP'],color='Red')

#first row sec col
ax1 = plt.subplot2grid((2,2), (0, 1))
plt.scatter(x=car['Driven_Wheels'], y=car['MSRP'], color='Orange')
plt.ylabel('MSRP')
plt.title('Driven Wheels')

#Second row first column
ax1 = plt.subplot2grid((2,2), (1, 0))
sns.barplot(x=car['Transmission Type'], y=car['MSRP'])
plt.xticks(rotation=35)

#second row second column
ax1 = plt.subplot2grid((2,2), (1, 1))
sns.barplot(x=car['Vehicle Size'], y=car['MSRP'])

plt.yticks(weight='bold')

plt.show()
modelp = car.groupby(['Make']).sum()[['MSRP','Popularity','Engine HP','Engine Cylinders','Number of Doors']].nlargest(5, 'Popularity')
modelp.groupby(['Make']).sum()['Popularity'].nlargest(10).iplot(kind='bar', xTitle='Make', yTitle='Popularity',title='Top 5 popular Car Brands')
#F=car.loc[car['Make'] == 'BMW']
F=car[car['Make'].str.contains('Ford')]
Ford=F.sort_values(by=['Popularity'],ascending=False).nlargest(10, 'MSRP')
Ford= Ford.style.background_gradient(cmap='Reds')
Ford
sns.regplot(x=car["MSRP"], y=car["Popularity"], line_kws={"color":"red","alpha":1,"lw":5})
plt.title('Popularity and MSRP', weight='bold', fontsize=18)
plt.xlabel('MSRP', weight='bold',fontsize=14)
plt.ylabel('Popularity', weight='bold', fontsize=14)
plt.xticks(weight='bold')
plt.yticks(weight='bold')
EFT=car[['Engine Fuel Type',
      'Engine HP',
      'Engine Cylinders',
      'highway MPG',
         'MSRP',
        'city mpg']].groupby(['Engine Fuel Type']).agg('median').sort_values(by=['Engine HP'],ascending=False)
EFT1= EFT.style.background_gradient(cmap='Purples')
EFT1