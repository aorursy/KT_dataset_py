import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
DF = pd.read_csv('../input/conposcovidloc/conposcovidloc.csv')
print (DF.describe())
print(DF.dtypes)
DF.head()
rows = DF.shape[0]
columns = DF.shape[1]
print(rows)
print(columns)
DF.Accurate_Episode_Date.value_counts(sort=False).plot.bar(figsize=(50,15))
DF.Outcome1.value_counts(sort=False).plot.pie(autopct='%1.0f%%',shadow=True, startangle=90)
print(DF.groupby('Outcome1').size())
explode = (0,0,0.2,0.3,0.2)
DF.Client_Gender.value_counts(sort=False).plot.pie(autopct='%1.0f%%',shadow=True,startangle = 60,explode = explode)
print(DF.groupby('Client_Gender').size())
DF.Reporting_PHU_City.value_counts(sort=True).plot.bar(figsize = (10,5))
DF.Age_Group.value_counts(sort=True).plot.bar(figsize=(10,5))
r_DF = DF.loc[DF['Outcome1'] == 'Resolved']
print(r_DF.Client_Gender.value_counts(sort=False).plot.bar(figsize=(5,5),title='Resolved Outcome v Gender'))
f_DF = DF.loc[DF['Outcome1'] == 'Fatal']
print(f_DF.Client_Gender.value_counts(sort=False).plot.bar(figsize=(5,5),title='Fatal Outcome v Gender'))
ff_DF = DF[(DF['Outcome1'] == 'Fatal') & (DF['Client_Gender'] == 'FEMALE')]
fm_DF = DF[(DF['Outcome1'] == 'Fatal') & (DF['Client_Gender'] == 'MALE')]
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
ff_DF.Age_Group.value_counts(sort=False).plot.bar(ax=axes[0], title='Fatal age group - Female')
fm_DF.Age_Group.value_counts(sort=False).plot.bar(ax=axes[1], title='Fatal age group - Male')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
r_DF.Reporting_PHU.value_counts(sort=False).plot.bar(title='PHU and Resolved',ax=axes[0])
f_DF.Reporting_PHU.value_counts(sort=False).plot.bar(title='PHU and Fatal',ax=axes[1])
#To confirm if the numbers in the graphs are right:
print(r_DF.loc[r_DF['Reporting_PHU']== 'Toronto Public Health'].shape[0])
print(f_DF.loc[f_DF['Reporting_PHU']== 'Toronto Public Health'].shape[0])
import geopandas
gdf = geopandas.GeoDataFrame(DF, geometry=geopandas.points_from_xy(DF.Reporting_PHU_Longitude, DF.Reporting_PHU_Latitude))
geopandas.datasets.available
rows = gdf.shape[0]
columns = gdf.shape[1]
print(rows)
print(columns)
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#Restricting to North America - Canada
ax = world[world.name == 'Canada'].plot(figsize=(20,10),color='white', edgecolor='black')
#Plotting geodataframe
gdf.plot(ax=ax, color='red', figsize=(10,10))
plt.show()
nr_DF = DF.loc[DF['Outcome1'] == 'Not Resolved']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
nr_DF.Case_AcquisitionInfo.value_counts(sort=False).plot.bar(title='Case_AcquisitionInfo and Not Resolved', ax=axes[0])
r_DF.Case_AcquisitionInfo.value_counts(sort=False).plot.bar(title='Case_AcquisitionInfo and Resolved', ax=axes[1])
f_DF.Case_AcquisitionInfo.value_counts(sort=False).plot.bar(title='Case_AcquisitionInfo and Fatal', ax=axes[2])