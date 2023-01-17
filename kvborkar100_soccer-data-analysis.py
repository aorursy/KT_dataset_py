import sqlite3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
c = sqlite3.connect('../input/database.sqlite')
data = pd.read_sql_query("SELECT * FROM Player_Attributes",c)
# Checking first five columns of the dataset
data.head()
# Total No of Columns
data.columns
len(data.columns)
data.describe().T
# Check for the NULL values in the dataset
data.isnull().sum()
# Number of rows before deleting 
original_rows = data.shape[0]
print("Number of rows befor deletion: {}".format(original_rows))

# Deleting rows with NULL values
data = data.dropna()

#Number of rows after deletion
new_rows = data.shape[0]
print("Number of rows befor deletion: {}".format(new_rows))

data.dtypes
object_dtype = list(data.select_dtypes(include=['object']).columns)
print(object_dtype)
# Lets try to check correlation between different features
corr_data = data.corr()
# Plotting these values using seaborns Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_data,cmap='viridis')
plt.figure(figsize=(12,12),dpi=100)
sns.clustermap(corr_data,cmap='viridis')
#Extracting Highly Related features to Overall Rating
cols = corr_data.nlargest(11,columns='overall_rating')['overall_rating'].index
corr_matrix2 = np.corrcoef(data[cols].values.T)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix2, cbar=True,cmap='viridis', annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)