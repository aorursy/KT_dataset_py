import numpy as np
import pandas as pd
#import seaborn
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../input/chexpert-train-csv-modified/train - train (1).csv")
data.head()
data1 = data['View']
data1.head()
#Dropping all lateral views
indexNames = data[ data['View'] == 'Lateral' ].index
data.drop(indexNames , inplace=True)
data1 = data
data1
plt.figure(figsize=(16,6)) 
ax = sns.barplot(x="Consolidation", y="Age", data=data1)
print("Consolidation" )
print(data['Consolidation'].value_counts())
print()
print("Pneumonia" )
print(data['Pneumonia'].value_counts())
print()
print("No findings")
print( data['No Finding'].value_counts())
print()
print("Lung Opacity"  )
print(data['Lung Opacity'].value_counts())

#data2 = data1.pivot('No Finding' , 'Lung Opacity' , 'Consolidation' )
#ax = sns.heatmap(flights)

pd.crosstab(data.Consolidation, data.Pneumonia)
pd.crosstab(data['Lung Opacity'], data.Pneumonia)
pd.crosstab(data['Lung Opacity'], data.Consolidation)
data1.groupby(['Pneumonia','Consolidation', 'Lung Opacity']).size().unstack(fill_value=0)
data['Pneumonia'].value_counts().plot(kind='barh')
ax = sns.countplot(x="Pneumonia", hue="Consolidation" , data=data1)