import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
import os
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format
import colorsys
DataSetGender = pd.read_csv('../input/gender_submission.csv')
DataSetGender.shape
DataSetGender.isnull().sum()
TotalDataSetGender = len(DataSetGender)
print("Total Number of Data Count :", TotalDataSetGender)
DataSetGender.fillna(0)
DataSetGender.head(10)
DataSetTest = pd.read_csv('../input/test.csv')
DataSetTest.shape
DataSetTest.isnull().sum()
TotalDataSetTest = len(DataSetTest)
print("Total Number of Data Count :", TotalDataSetTest)
DataSetTest.fillna(0)
DataSetTest.head(10)
DataSetTrain = pd.read_csv('../input/train.csv')
DataSetTrain.shape
DataSetTrain.isnull().sum()
TotalDataSetGender = len(DataSetGender)
print("Total Number of Data Count :", TotalDataSetGender)
DataSetTrain.fillna(0)
DataSetTrain.head(10)
SexCount = DataSetTrain['Sex'].value_counts()
print(SexCount)
pivot = DataSetTrain["Sex"].value_counts().nlargest(10)
pivot.plot.bar()
plt.show()
palette = 'cividis'
plt.figure(figsize=(8,5))
sn.heatmap(DataSetTrain.drop('Survived', axis=1).corr(), annot=True, cmap=palette+'_r')
DataSetTrain[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
DataSetTrain.groupby(['Embarked','Survived', 'Pclass'])['Survived'].count()
sn.set(font_scale=1)
g = sn.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=DataSetTrain, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.9)
g.fig.suptitle('How Passengers Survived by Class');
SurvivedCount = DataSetTrain["Survived"].value_counts().nlargest(10)
print("Survived Count \n")
print(SurvivedCount)
pivot = DataSetTrain["Survived"].value_counts().nlargest(10)
pivot.plot.bar()
plt.show()
DataSetTrain[DataSetTrain["Survived"]==1][["Name", "Sex" , "Age", "Pclass", "Cabin"]]
DataSetTrain[DataSetTrain["Survived"]==0][["Name", "Sex" , "Age", "Pclass", "Cabin"]]
f, ax = plt.subplots()
sn.countplot('Sex', hue='Survived', data=DataSetTrain, ax=ax)
ax.set_title('Survived and Dead with Sex')
plt.show()
UniqueCabin = DataSetTrain['Cabin'].unique()
print("All Unique Cabin Name \n")
print(UniqueCabin)
TotalSurvived = DataSetTrain.groupby(['Survived'])['Fare'].sum().nlargest(15)
print("Total Fare for Survived and Not Survived\n")
print(TotalSurvived)
FareMax = DataSetTrain['Fare'].max()
print ("Max Fare Mode is :", round(FareMax,2))
FareMin = DataSetTrain['Fare'].min()
print ("Min Fare Mode is :", round(FareMin,2))
FareMean = DataSetTrain['Fare'].mean()
print ("Mean Fare Mode is :", round(FareMean,2))
SurvivedFareData=DataSetTrain[DataSetTrain['Survived']==1]
SurvivedFareMax = SurvivedFareData['Fare'].max()
print ("Survived Max Fare Mode is :", round(SurvivedFareMax,2))
SurvivedFareMin = SurvivedFareData['Fare'].min()
print ("Survived Min Fare Mode is :", round(SurvivedFareMin,2))
SurvivedFareMean = SurvivedFareData['Fare'].mean()
print ("Survived Mean Medal Mode is :", round(SurvivedFareMean,2))
NotSurvivedFareData=DataSetTrain[DataSetTrain['Survived']==0]
NotSurvivedFareMax = NotSurvivedFareData['Fare'].max()
print ("Not Survived Max Fare Mode is :", round(NotSurvivedFareMax,2))
NotSurvivedFareMin = NotSurvivedFareData['Fare'].min()
print ("Not Survived Min Fare Mode is :", round(NotSurvivedFareMin,2))
NotSurvivedFareMean = NotSurvivedFareData['Fare'].mean()
print ("Not Survived Mean Medal Mode is :", round(NotSurvivedFareMean,2))