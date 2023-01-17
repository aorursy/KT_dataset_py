# import libraries needed to plot the graphs accordingly
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools import plotting
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# input the breast cancer dataset in a pandas dataframe
data = pd.read_csv("../input/data.csv")
data = data.drop(['Unnamed: 32','id'],axis = 1)
# take a look at the data and the available characteristics
data.head()
data.shape # (569, 31)
data.columns 
m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()

#finding the most frequent malignant radius mean
frequent_malignant_radius_mean = m[0].max()
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_radius_mean)

#finding the most frequent bening radius mean
frequent_bening_radius_mean = b[0].max()
index_frequent_bening_radius_mean = list(b[0]).index(frequent_bening_radius_mean)
most_frequent_bening_radius_mean = b[1][index_frequent_bening_radius_mean]
print("Most frequent bening radius mean is: ",most_frequent_bening_radius_mean)
m = plt.hist(data[data["diagnosis"] == "M"].area_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].area_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Area Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Area Mean for Bening and Malignant Tumors")
plt.show()

#finding the most frequent malignant area mean
frequent_malignant_area_mean = m[0].max()
index_frequent_malignant_area_mean = list(m[0]).index(frequent_malignant_area_mean)
most_frequent_malignant_area_mean = m[1][index_frequent_malignant_area_mean]
print("Most frequent malignant area mean is: ",most_frequent_malignant_area_mean)

#finding the most frequent bening radius mean
frequent_bening_area_mean = b[0].max()
index_frequent_bening_area_mean = list(b[0]).index(frequent_bening_area_mean)
most_frequent_bening_area_mean = b[1][index_frequent_bening_area_mean]
print("Most frequent bening area mean is: ",most_frequent_bening_area_mean)
m = plt.hist(data[data["diagnosis"] == "M"].perimeter_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].perimeter_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Perimeter Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Perimeter Mean for Bening and Malignant Tumors")
plt.show()

frequent_malignant_perimeter_mean = m[0].max()
index_frequent_malignant_perimeter_mean = list(m[0]).index(frequent_malignant_perimeter_mean)
most_frequent_malignant_perimeter_mean = m[1][index_frequent_malignant_perimeter_mean]
print("Most frequent malignant perimeter mean is: ",most_frequent_malignant_perimeter_mean)

frequent_bening_perimeter_mean = b[0].max()
index_frequent_bening_perimeter_mean = list(b[0]).index(frequent_bening_perimeter_mean)
most_frequent_bening_perimeter_mean = b[1][index_frequent_bening_perimeter_mean]
print("Most frequent bening perimeter mean is: ",most_frequent_bening_perimeter_mean)

#Box plot for radius mean
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

#Box plot for area mean
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['area_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

#Box plot for perimeter mean
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['perimeter_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()

f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Heat Map (showing the correlation matrix)')
plt.savefig('graph.png')
plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(data.radius_mean,data.area_mean,kind="regg")
plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(data.perimeter_mean,data.area_mean,kind="regg")
plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(data.radius_mean,data.perimeter_mean,kind="regg")
plt.show()

