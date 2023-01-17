import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe_companies = pd.read_csv("../input/companies_sorted.csv")

dataframe_companies.head()
dataframe_companies = dataframe_companies[['name','year founded','industry','size range','country','current employee estimate']]

dataframe_companies.head()
print(dataframe_companies.info())
dataframe_companies.boxplot('year founded','size range',figsize = (10,10))
print('Total null values in the dataset: ',dataframe_companies.isnull().values.sum())

print('Column wise distribution of null values in the dataset')

print(dataframe_companies.isnull().sum())
#Removing the row where company name is null. Since it's completely useless. There are 3 rows where company name is null.

#axis = 0 defines that we need to delete the row. If axis is 1 then the column would be deleted.

#subset defines which column to consider for null values.

dataframe_companies = dataframe_companies.dropna(axis = 0,subset = ['name'])



#Cross checking if the null values were deleted properly.

print('Total null values in the dataset: ',dataframe_companies.isnull().values.sum())

print('Column wise distribution of null values in the dataset')

print(dataframe_companies.isnull().sum())
#fill the null values in the country by inserting "missing" in the column where it's null or empty.

dataframe_companies['country'].fillna('missing',inplace = True)

dataframe_companies['industry'].fillna('missing',inplace = True)



#Keep only those rows where we have atleast 3 non null column values. Drop rest of them.

dataframe_companies.dropna(thresh = 3, inplace = True)



#Cross checking 

print('Total null values in the dataset: ',dataframe_companies.isnull().values.sum())

print('Column wise distribution of null values in the dataset')

print(dataframe_companies.isnull().sum())
#Drawing a histogram for the year.

dataframe_companies.hist('year founded',bins = 10)
print(dataframe_companies['year founded'].median())
dataframe_companies.fillna(dataframe_companies['year founded'].median(), inplace = True)



#Cross checking 

print('Total null values in the dataset: ',dataframe_companies.isnull().values.sum())

print('Column wise distribution of null values in the dataset')

print(dataframe_companies.isnull().sum())
dataframe_companies.industry.value_counts()

labelEncoder = LabelEncoder()

industry_labels = labelEncoder.fit_transform(dataframe_companies['industry'])

industry_mappings = {index: label for index, label in enumerate(labelEncoder.classes_)}

print(industry_mappings)
dataframe_companies['industry_mapping'] = LabelEncoder().fit_transform(dataframe_companies['industry'])

dataframe_companies.head()
dataframe_companies.country.value_counts()

dataframe_companies['country_mapping'] = LabelEncoder().fit_transform(dataframe_companies['country'])

dataframe_companies.head()
dataframe_companies.rename(index=str, columns={"size range": "size_range","year founded": "year_founded","current employee estimate":"current_employee_estimate"},inplace = True)

dataframe_companies.head()

dataframe_companies.size_range.value_counts()
dataframe_companies['size_range_mapping'] = LabelEncoder().fit_transform(dataframe_companies['size_range'])

dataframe_companies.head()
dataframe_companies['year_founded'] = dataframe_companies['year_founded'].astype(np.int64)

print(dataframe_companies.info())

dataframe_companies.head()
dataframe_companies.to_csv('Companies_Cleaned_Dataset.csv', sep=',', encoding='utf-8')
