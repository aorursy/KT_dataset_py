import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data_summer_rating =  pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')
data_summer_rating.head()
data_summer_rating.info()
plt.figure(figsize = (20, 10))
sns.heatmap(data_summer_rating.corr(), annot = True, cmap = 'coolwarm', center = 0)
plt.show()
data_categories = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.csv')
data_categories.head()
data_categories.info()
data_categories_sorted = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/unique-categories.sorted-by-count.csv')
data_categories_sorted.head()
data_categories_sorted.info()