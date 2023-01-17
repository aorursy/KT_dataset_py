# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import libraris

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings 
warnings.filterwarnings('ignore')
import os
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.value_counts
df.shape
df.columns
df['Unnamed: 32']
df['Unnamed: 32'].value_counts
df['Unnamed: 32'].mean()
df=df.drop(['Unnamed: 32'],axis=1)
df.columns
df['diagnosis'].value_counts()
m = plt.hist(df[df['diagnosis']=='M'].radius_mean,bins=30,fc = (1,0,0,0.5),label='Malignant')
print(m[0])
print(m[1])
print(m[2])
b = plt.hist(df[df['diagnosis']=='B'].radius_mean,bins=30,fc = (0,1,0,0.5),label='Bening')
print(b[0])
print(b[1])
print(b[2])
plt.legend()
plt.xlabel('Radius Mean Values')
plt.ylabel('Frequency')
plt.title('Histogram of radius mean for Bening and Malignant Tumors')
plt.show()
data_bening = df[df['diagnosis']=='B']
data_malignant=df[df['diagnosis']=='M']
data_bening.describe()
data_malignant.describe()
desc=data_bening.radius_mean.describe()
desc.describe()
print(desc[0])
print(desc[1])
print(desc[2])
Q1=desc[4]
Q1
Q2=desc[5]
Q2
Q3=desc[6]
Q3
IQR=Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")
data_bening[data_bening.radius_mean < lower_bound].radius_mean
print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)
melted_data = pd.melt(df,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
print(melted_data)
plt.figure(figsize = (15,10))
sns.boxplot(x = melted_data["variable"], y = melted_data["value"], hue=melted_data["diagnosis"],data=melted_data)
plt.show()
print("mean: ",data_bening.radius_mean.mean())
print("variance: ",data_bening.radius_mean.var())
print("standart deviation (std): ",data_bening.radius_mean.std())
print("describe method: ",data_bening.radius_mean.describe())
plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Bening',density=True,cumulative = True)

sorted_data = np.sort(data_bening.radius_mean)

y = np.arange(len(sorted_data))/float(len(sorted_data)-1)

plt.plot(sorted_data,y,color='red')

plt.title('CDF of bening tumor radius mean')

plt.show()
plt.figure(figsize = (15,10))
sns.jointplot(df.radius_mean,df.area_mean,kind="regg")
plt.show()
sns.set(style = "white")
df1 = df.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df1,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()
f,ax=plt.subplots(figsize = (18,18))
sns.heatmap(df.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.title('Correlation Map')
plt.savefig('graph.png')
plt.show()
np.cov(df1.radius_mean,df1.area_mean)
print("Covariance between radius mean and area mean: ",df1.radius_mean.cov(df1.area_mean))
print("Covariance between radius mean and fractal dimension se: ",df1.radius_mean.cov(df1.fractal_dimension_se))
p1 = df1.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
p2 = df1.radius_mean.cov(df1.area_mean)/(df1.radius_mean.std()*df1.area_mean.std())
print('Pearson correlation: ')
print(p1)
print('Pearson correlation: ',p2)
ranked_data = df1.rank()
spearman_corr = ranked_data.loc[:,["area_mean","radius_mean"]].corr(method= "pearson")
print("Spearman's correlation: ")
print(spearman_corr)
salary = [1,4,3,2,5,4,2,3,1,500]
print("Mean of salary: ",np.mean(salary))
print("Median of salary: ",np.median(salary))
statistic, p_value = stats.ttest_rel(df1.radius_mean,df1.area_mean)
print('p-value: ',p_value)
# parameters of normal distribution
mu, sigma = 110, 20  # mean and standard deviation
s = np.random.normal(mu, sigma, 100000)
print("mean: ", np.mean(s))
print("standart deviation: ", np.std(s))
# visualize with histogram
plt.figure(figsize = (10,7))
plt.hist(s, 100, density=False)
plt.ylabel("frequency")
plt.xlabel("IQ")
plt.title("Histogram of IQ")
plt.show()