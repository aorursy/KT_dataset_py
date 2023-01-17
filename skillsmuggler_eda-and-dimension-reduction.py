# Load libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Load data

df = pd.read_csv('../input/train.csv')
df.info(verbose=False)
# Column names

print(len(df.columns))
df.columns
# Soil_Type and Wilderness_Area list

soil_list = []
for i in range(1, 41):
    soil_list.append('Soil_Type' + str(i))

wilderness_area_list = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']

print(soil_list, "\n")
print(wilderness_area_list)
# Check for null values

print("Total number of NaN values: ", df.isnull().sum().sum())
# Check for multiple Wilderness_Area for a single Cover_Type

wilderness_area_test = df[wilderness_area_list].sum(axis=1)
wilderness_area_test.unique()
# Check for multiple Soil_Types for a single Cover_Type

soil_type_test = df[soil_list].sum(axis=1)
soil_type_test.unique()
# Wilderness areas

def wilderness_compress(df):
    
    df[wilderness_area_list] = df[wilderness_area_list].multiply([1, 2, 3, 4], axis=1)
    df['Wilderness_Area'] = df[wilderness_area_list].sum(axis=1)
    return df
# Soil types

def soil_compress(df):
    
    df[soil_list] = df[soil_list].multiply([i for i in range(1, 41)], axis=1)
    df['Soil_Type'] = df[soil_list].sum(axis=1)
    return df
# Compressing features

df = wilderness_compress(df)
df = soil_compress(df)

df[['Wilderness_Area', 'Soil_Type']].head()
# Useful columns

cols = df.columns.tolist()
columns = cols[1:11] + cols[56:]

print("Useful columns: ", columns)

values = df[columns]
labels = df['Cover_Type']

print("Values: ", values.shape)
print("Labels: ", labels.shape)
# Set style

sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})
# Cover Type countplot

ax = sns.countplot(labels, alpha=0.75)
ax.set(xlabel='Cover Type', ylabel='Number of labels')
plt.show()
# Elevation distribution

ax = sns.distplot(df['Elevation'], color='pink')
plt.show()
# Aspect distribution

ax = sns.distplot(df['Aspect'], color='#add8e6')
plt.show()
# Slope distribution

ax = sns.distplot(df['Slope'], color='#90ee90')
plt.show()
# Distance to hydrology (Horizontal versus Vertical) with Elevation

ax = plt.scatter(x=df['Horizontal_Distance_To_Hydrology'], y=df['Vertical_Distance_To_Hydrology'], c=df['Elevation'], cmap='jet')
plt.xlabel('Horizontal Distance')
plt.ylabel('Vertical Distance')
plt.title("Distance to Hydrology with Elevation")
plt.show()
# Hillshade at 9am distribution

ax = sns.distplot(df['Hillshade_9am'], color='#fcd14d')
plt.show()
# Hillshade at noon distribution

ax = sns.distplot(df['Hillshade_Noon'], color='#fdb813')
plt.show()
# Hillshade at 3pm distribution

ax = sns.distplot(df['Hillshade_3pm'], color='orange')
plt.show()
# Wilderness area countplot

ax = sns.countplot(df['Wilderness_Area'], alpha=0.75)
ax.set(xlabel='Wilderness Area Type', ylabel='Number of Areas', title='Wilderness areas - Count')
plt.show()
# Wilderness area to Cover type mapping

ax = plt.scatter(x=df['Wilderness_Area'], y=df['Cover_Type'], c=df['Wilderness_Area'], cmap='Set2', s=500, marker='s', alpha=0.01)
plt.xlabel('Wilderness Area Type')
plt.xticks([1, 2, 3, 4])
plt.ylabel('Cover Type')
plt.title("Wilderness Area versus Cover Type")
plt.show()
# Soil Type countplot

ax = sns.countplot(df['Soil_Type'], alpha=0.75)
ax.set(xlabel='Soil Type', ylabel='Count', title='Soil types - Count')
plt.show()
# Soil Type jointplot

ax = sns.jointplot(x='Soil_Type', y='Cover_Type', data=df, kind='kde', color='purple')
plt.show()
# Reduced dataset

clean = df[['Id', 'Cover_Type'] + columns]
clean.head()
# Clean data columns

print(len(clean.columns))
clean.columns
# CSV file

clean.to_csv('clean_train.csv', index=False)
