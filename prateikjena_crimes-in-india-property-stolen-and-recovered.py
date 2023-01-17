
import numpy as np 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

property=pd.read_csv("../input/crime-in-india/10_Property_stolen_and_recovered.csv")
property
property.Area_Name.value_counts()
property.Year.value_counts()
property.Group_Name.value_counts()
property.Sub_Group_Name.value_counts()
a=property.groupby(['Group_Name']).get_group('Robbery - Property')
a.Sub_Group_Name.value_counts()
a=property.groupby(['Group_Name']).get_group('Total Property')
a.Sub_Group_Name.value_counts()
#Since Sub_Group_Name is an irrelevant variable as previously proven ,we will use property1 as the base dataset
property1=property.drop(['Sub_Group_Name'],axis=1)

property_bystate=property1.groupby(['Area_Name'],as_index=False).sum()
property_bystate.drop("Year",axis=1,inplace=True)
plt.figure(figsize = (20, 10))
chart=sns.barplot(x=property_bystate.Area_Name,y=property_bystate.Cases_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
#Cases of Property Stolen across the year of all the States 
sortbyyear=property1.groupby(['Year'],as_index=False).sum()
sortbyyear
plt.figure(figsize = (20, 10))
chart=sns.barplot(x=sortbyyear.Year,y=sortbyyear.Cases_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
#Value of Property Stolen across the year of all the States 

plt.figure(figsize = (20, 10))
chart=sns.barplot(x=sortbyyear.Year,y=sortbyyear.Value_of_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
plt.figure(figsize = (20, 10))
chart=sns.barplot(x=property_bystate.Area_Name,y=property_bystate.Value_of_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
a=property1.groupby(['Area_Name']).get_group('Maharashtra')
plt.figure(figsize = (20, 10))
chart=sns.barplot(x=a.Year,y=a.Value_of_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
a=property1.groupby(['Area_Name']).get_group('Maharashtra')
plt.figure(figsize = (20, 10))
chart=sns.barplot(x=a.Year,y=a.Cases_Property_Stolen)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
property.groupby(['Group_Name']).get_group('Robbery - Property')

sns.scatterplot(x=property.Cases_Property_Recovered,y=property.Cases_Property_Stolen)
property_bystate['Difference']=property_bystate["Cases_Property_Stolen"]- property_bystate["Cases_Property_Recovered"]
property_bystate

property_bystate['Percent_Recovery_Cases']=(property_bystate["Cases_Property_Recovered"]/ property_bystate["Cases_Property_Stolen"])*100

property_bystate.Percent_Recovery_Cases.sort_values()
property_bystate.Percent_Recovery_Cases.sort_values().mean()