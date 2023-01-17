#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#reading a .CSV file
data= pd.read_csv("../input/us-president-height-dataset/president_heights.csv")
print(data.head())
#creating an array of data
height= np.array(data["height(cm)"])
print(height)
#printing required results
print("Mean of heights= ", height.mean())
print("Standard Deviation= ", height.std())
print("Minimum height= ", height.min())
print("Maximum height= ", height.max())
print("Median of heights= ", np.median(height))
print("10th percentile of height= ", np.percentile(height, 10))
#visual representation
plt.hist(height)
plt.title("Distribution of US President height")
plt.xlabel("height(cm)")
plt.ylabel("Number")
plt.show()