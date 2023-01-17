import os
os.getcwd()
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Import data
wine_reviews = pd.read_csv('..\\winemag-data_first150k.csv', index_col=0)  #Removed unnamed index column
wine_reviews.columns = ['Country', 'Description', 'Designation', 'Points', 'Price', 'Province', 'Region', 'Subregion', 'Variety', 'Winery']
wine_reviews.head(10)
#Checking Data 
wine_reviews.info()
#Checking Data 
print(wine_reviews.shape)
print("Length of Wine Reviews dataframe:", len(wine_reviews))
print(wine_reviews.columns)
#Shaping type
wine_reviews.Country=wine_reviews.Country.astype('category')
wine_reviews.Description=wine_reviews.Description.astype('category')
wine_reviews.Designation=wine_reviews.Designation.astype('category')
wine_reviews.Province=wine_reviews.Province.astype('category')
wine_reviews.Region=wine_reviews.Region.astype('category')
wine_reviews.Subregion=wine_reviews.Subregion.astype('category')
wine_reviews.Variety=wine_reviews.Variety.astype('category')
wine_reviews.Winery=wine_reviews.Winery.astype('category')
wine_reviews.info()
#Cleaning data
#Imported pywrangle (pip install pywrangle  to check null values)
import pywrangle as pw
pw.show_col_nulls(wine_reviews)
