# Importing pandas libraby
import pandas as pd  
import matplotlib.pyplot as plt

# Reading the dataset. Here the file is in .tsv format. Hence we mention the sep as '\t'.
amazonData=pd.read_csv('../input/amazon-alexa/aa.tsv',sep='\t')
# Group the customer records based on the rating values. 
amazonData.groupby(['rating']).rating.count().plot.bar()
plt.ylabel('Number of Customers')
plt.xlabel('Rating')
# In the below code, I used 'variation' as a marker to count the rows per product design.
amazonData.groupby(['variation']).rating.count().sort_values(ascending=False).plot.bar()
plt.xlabel('Design Variation')
plt.ylabel('Number of Sales')
ratingPerdesign= amazonData.groupby(['variation','rating']).rating.count().unstack().plot(kind='bar',stacked=True)
plt.xlabel('Design Variations')