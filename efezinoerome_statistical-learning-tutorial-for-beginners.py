# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import libraries
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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read data as pandas data frame
data = pd.read_csv("../input/data.csv")
print(data.columns)
print(data.head())
data = data.drop(['Unnamed: 32','id'],axis = 1)
# quick look to data
print(data.head())
print(100*'*')
print(data.shape) # (569, 31)
print(100*'*')
print(data.columns)
print(data.info())
data.diagnosis.value_counts()
print(data[data.diagnosis == "M"]['radius_mean'].min())
print(data[data.diagnosis == "M"]['radius_mean'].max())
print(100*'*')
print(data[data.diagnosis == "M"]['radius_mean'].value_counts().sort_values(ascending = False))
fig, ax = plt.subplots(1,1)
fig.set_size_inches(10,5)
sns.distplot(data[data.diagnosis == "M"].radius_mean,label= 'Malignant', kde = False, bins = 30, ax = ax)
sns.distplot(data[data.diagnosis == "B"].radius_mean,label= 'Benign', kde = False, bins = 30, ax = ax)
ax.set_title('"Histogram of Radius Mean for Bening and Malignant Tumors"')
ax.set_xlabel('Radius Mean Values')
ax.set_ylabel('Frequency')

ax.legend()
m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
frequent_malignant_radius_mean = m[0].max()
index_frequent_malignant_radius_mean = list(m[0]).index(frequent_malignant_radius_mean)
most_frequent_malignant_radius_mean = m[1][index_frequent_malignant_radius_mean]
print("Most frequent malignant radius mean is: ",most_frequent_malignant_radius_mean)
print(m[0])
print(100*'*')
print(m[1])
fig, ax = plt.subplots(1,2)
fig.set_size_inches(15,5)
data_benign = data[data["diagnosis"] == "B"]
sns.boxplot(data = data, x = 'diagnosis', y ='radius_mean', ax = ax[0])
sns.boxplot(data = data, x = 'diagnosis', y ='texture_mean', ax = ax[1])
data_bening = data[data["diagnosis"] == "B"]
data_malignant = data[data["diagnosis"] == "M"]
desc = data_bening.radius_mean.describe()
Q1 = desc[4] #25% percentile
Q3 = desc[6] #75% percentile
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")
data_bening[data_bening.radius_mean < lower_bound].radius_mean
print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)
desc
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.savefig('graph.png')
plt.show()
print("mean: ",data_bening.radius_mean.mean())
print(100*'*')
print("variance: ",data_bening.radius_mean.var())
print(100*'*')
print("standart deviation (std): ",data_bening.radius_mean.std())
print(100*'*')
print("describe method: ",data_bening.radius_mean.describe())
plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Benign',normed = False,cumulative = False)
plt.show()
plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Benign',normed = True,cumulative = True)
sorted_data = np.sort(data_bening.radius_mean)
y = np.arange(len(sorted_data))/float(len(sorted_data)-1)
plt.plot(sorted_data,y,color='red')
plt.title('CDF of bening tumor radius mean')
plt.show()
mean_diff = data_malignant.radius_mean.mean() - data_bening.radius_mean.mean()
var_bening = data_bening.radius_mean.var()
var_malignant = data_malignant.radius_mean.var()
var_pooled = (len(data_bening)*var_bening +len(data_malignant)*var_malignant ) / \
float(len(data_bening)+ len(data_malignant))
effect_size = mean_diff/np.sqrt(var_pooled)
print("Effect size: ",effect_size)
plt.figure(figsize = (15,10))
sns.jointplot(data.radius_mean,data.area_mean,kind="regg")
plt.show()
# Also we can look relationship between more than 2 distribution
sns.set(style = "white")
df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.show()
np.cov(data.radius_mean,data.area_mean)
print("Covariance between radius mean and area mean: ",data.radius_mean.cov(data.area_mean))
print("Covariance between radius mean and fractal dimension se: ",data.radius_mean.cov(data.fractal_dimension_se))
p1 = data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
p2 = data.radius_mean.cov(data.area_mean)/(data.radius_mean.std()*data.area_mean.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ',p2)
ranked_data = data.rank()
spearman_corr = ranked_data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)
salary = [1,4,3,2,5,4,2,3,1,500]
print("Mean of salary: ",np.mean(salary))
print("Median of salary: ",np.median(salary))
statistic, p_value = stats.ttest_rel(data.radius_mean,data.area_mean)
print('p-value: ',p_value)
