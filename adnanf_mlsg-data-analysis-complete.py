import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
# ignore warning about "Dtypes"
cars = pd.read_csv('/kaggle/input/fuel-economy/database.csv')
# print out the number of rows and columns (dimensions)
cars.shape
# print the names of the columns
cars.columns
cars = cars[['Vehicle ID', 'Year', 'Make', 'Model', 'Class', 'Drive', 'Transmission',
            'Transmission Descriptor', 'Engine Index', 'Engine Descriptor',
           'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
           'Supercharger', 'Fuel Type', 'Fuel Type 1',
            'Combined MPG (FT1)']]
cars.shape
# print first 5 rows
cars.head()
cars.tail()
cars.info()
cars.describe()
cars.head(2)
id_columns = ['Make', 'Model', 'Class', 'Drive', 'Transmission',
            'Transmission Descriptor', 'Engine Cylinders', 'Engine Displacement', 'Turbocharger',
           'Supercharger', 'Fuel Type', 'Fuel Type 1',
            'Combined MPG (FT1)']
duplicates = cars.duplicated(subset=id_columns)

duplicates.sum()
cars[duplicates].head()
cars_dedup = cars.drop_duplicates(subset=id_columns)
cars_dedup.shape
cars['Make'].value_counts()
cars['Class'].value_counts()
# remember Drive had a few missing values
cars['Drive'].value_counts(dropna=False)
cars['Turbocharger'].value_counts(dropna=False)
cars['Supercharger'].value_counts(dropna=False)
cars['Fuel Type'].value_counts()
cars['Engine Cylinders'].plot.hist()
cars[cars['Engine Cylinders']==16]
cars[cars['Engine Cylinders']==2].head()
cars[cars['Engine Cylinders']==2].shape
cars[cars['Engine Cylinders']==2]['Make'].unique()
cars[cars['Engine Cylinders']==2].tail()
cars['Engine Displacement'].plot.hist()
cars[cars['Engine Displacement']==0]
cars[cars['Engine Displacement']==8].head()
cars['Combined MPG (FT1)'].plot.hist()
cars[cars['Combined MPG (FT1)']>120]
not_missing_drive = cars['Drive'].notnull()
not_missing_drive.sum()
cars['Turbocharger'] = cars['Turbocharger'].fillna('No')
cars['Supercharger'] = cars['Supercharger'].fillna('No')
no_mpg_outlier = cars['Combined MPG (FT1)'] < (cars['Combined MPG (FT1)'].median() + cars['Combined MPG (FT1)'].std())
no_mpg_outlier.sum()
no_electric_cars = cars['Fuel Type'] != 'Electricity'
cars_cleaned = cars[(~duplicates) & (not_missing_drive) & (no_mpg_outlier)]
cars_cleaned = cars_cleaned[['Class', 'Drive', 'Engine Cylinders', 
                             'Engine Displacement', 'Turbocharger','Supercharger', 
                             'Fuel Type', 'Combined MPG (FT1)']]

cars_cleaned.shape
import seaborn as sns
sns.pairplot(cars_cleaned)
sns.boxplot(x='Drive', y='Combined MPG (FT1)', data=cars_cleaned)
sns.boxplot(x='Turbocharger', y='Combined MPG (FT1)', data=cars_cleaned)
sns.boxplot(x='Supercharger', y='Combined MPG (FT1)', data=cars_cleaned)
sns.boxplot(x='Engine Cylinders', y='Combined MPG (FT1)', data=cars_cleaned)