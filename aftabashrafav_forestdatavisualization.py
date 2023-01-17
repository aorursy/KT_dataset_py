import os

os.chdir('../input/')

os.getcwd()

os.listdir()
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
my_data=pd.read_csv("train(1).csv",index_col=['Id'])
my_data.head(20)
my_data.describe()
my_data.info()
my_data.shape
my_data.columns
#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.

my_data['Wild_area'] = (my_data.iloc[:, 10:14] == 1).idxmax(1)

my_data['Soil_type'] = (my_data.iloc[:, 15:55] == 1).idxmax(1)

sns.scatterplot(x=my_data['Elevation'],y=my_data['Horizontal_Distance_To_Roadways'],hue=my_data['Cover_Type'],palette='rainbow');
sns.scatterplot(x=my_data['Slope'],y=my_data['Hillshade_Noon'])
sns.scatterplot(x=my_data['Elevation'],y=my_data['Horizontal_Distance_To_Roadways']);
sns.scatterplot(x=my_data['Aspect'],y=my_data['Hillshade_9am']);
#Count of the entries from different wilderness areas.

plt.figure(figsize=(10,10))

sns.countplot(my_data['Wild_area']);
plt.figure(figsize=(50,8))

sns.countplot(my_data['Soil_type']);
plt.figure(figsize=(13,15))

plt.subplots_adjust(hspace=0.5)



plt.subplot(5,2,1)

a1=sns.boxplot(y=my_data['Elevation'],x=my_data['Cover_Type']);



plt.subplot(5,2,2)

sns.boxplot(y=my_data['Aspect'],x=my_data['Cover_Type']);



plt.subplot(5,2,3)

#Boxplot between Slope and Cover type

sns.boxplot(y=my_data['Slope'],x=my_data['Cover_Type'],palette='rainbow');





plt.subplot(5,2,4)

#Boxplot between Horizontal_Distance_To_Hydrology and Cover type

sns.boxplot(y=my_data['Horizontal_Distance_To_Hydrology'],x=my_data['Cover_Type'],palette='rainbow');



plt.subplot(5,2,5)

#Boxplot between Vertical_Distance_To_Hydrology and Cover type

sns.boxplot(y=my_data['Vertical_Distance_To_Hydrology'],x=my_data['Cover_Type'],palette='rainbow');





plt.subplot(5,2,6)

#Boxplot between Horizontal_Distance_To_Roadways and Cover type

sns.boxplot(y=my_data['Horizontal_Distance_To_Roadways'],x=my_data['Cover_Type'],palette='rainbow');



plt.subplot(5,2,7)

#Boxplot between Hillshade_9am and Cover type

sns.boxplot(y=my_data['Hillshade_9am'],x=my_data['Cover_Type'],palette='rainbow');



plt.subplot(5,2,8)

#Boxplot between Hillshade_Noon and Cover type

sns.boxplot(y=my_data['Hillshade_Noon'],x=my_data['Cover_Type'],palette='rainbow');



plt.subplot(5,2,9)

#Boxplot between Hillshade_3pm and Cover type

sns.boxplot(y=my_data['Hillshade_3pm'],x=my_data['Cover_Type'],palette='rainbow');



plt.subplot(5,2,10)

#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type

sns.boxplot(y=my_data['Horizontal_Distance_To_Fire_Points'],x=my_data['Cover_Type'],palette='rainbow');
plt.figure(figsize=(16,16))

corrMetrix=my_data.iloc[:,:10].corr()

sns.heatmap(corrMetrix,vmin=-1,cmap='coolwarm',annot=True,square=True)

plt.show()
my_data_deg=my_data[['Elevation','Aspect','Slope','Cover_Type']]

sns.pairplot(my_data_deg,hue='Cover_Type')
plt.figure(figsize=(13,15))

plt.subplots_adjust(hspace=0.5)





plt.subplot(2,2,1)

a1=sns.swarmplot(data=my_data,x='Wild_area',y='Slope',hue='Cover_Type')



plt.subplot(2,2,3)

a2=sns.swarmplot(data=my_data,x='Wild_area',y='Elevation',hue='Cover_Type')



plt.subplot(2,2,2)

a3=sns.swarmplot(data=my_data,x='Cover_Type',y='Slope')



plt.figure(figsize=(13,15))

plt.subplots_adjust(hspace=0.5)



plt.subplot(3,2,1)

a1=sns.kdeplot(data=my_data['Elevation'],shade=True)

#a1.set_xticklabels(a1.get_xticklabels(),rotation=15)



plt.subplot(3,2,3)

a2=sns.kdeplot(data=my_data['Slope'],shade=True)

#a2.set_xticklabels(a2.get_xticklabels(),rotation=15)



plt.subplot(3,2,5)

a3=sns.kdeplot(data=my_data['Aspect'],shade=True);

#a3.set_xticklabels(a3.get_xticklabels(),rotation=15)





plt.subplot(3,2,2)

a4=sns.kdeplot(data=my_data['Horizontal_Distance_To_Fire_Points'],shade=True)

#a4.set_xticklabels(a4.get_xticklabels(),rotation=15)



plt.subplot(3,2,4)

a5=sns.kdeplot(data=my_data['Horizontal_Distance_To_Roadways'],shade=True);

#a5.set_xticklabels(a5.get_xticklabels(),rotation=15)



plt.subplot(3,2,6)

a6=sns.kdeplot(data=my_data['Horizontal_Distance_To_Hydrology'],shade=True)

#a6.set_xticklabels(a6.get_xticklabels(),rotation=15)
#Distribution of elevation values in the data.

sns.distplot(my_data['Elevation'],kde=False,color='red', bins=100);

plt.ylabel('Frequency',fontsize=10)
#Distribution of frequency of various soil types in the data.

my_data['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));

plt.xlabel('Frequency',fontsize=10)