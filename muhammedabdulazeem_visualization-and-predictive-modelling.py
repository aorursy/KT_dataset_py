# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',100)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
plt.style.use("ggplot")
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.drop(['Unnamed: 32','id'],axis = 1,inplace=True)
data.shape
data.describe(include='all')
data_y = data['diagnosis']
data_x = data.drop('diagnosis',axis=1)

data_standardization = (data_x - data_x.mean())/data_x.std()
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,0:10]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_violen,split=True, inner="quart")
plt.xticks(rotation=90)
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,10:20]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_violen,split=True, inner="quart")
plt.xticks(rotation=90)
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,20:31]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_violen,split=True, inner="quart")
plt.xticks(rotation=90)
plt.figure(figsize=(20,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data_violen)
plt.xticks(rotation=90)
## we can visulize any one for the reference

sns.jointplot(data_x.loc[:,'concavity_worst'], data_x.loc[:,'concave points_worst'], kind="regg", color="#ce1414")
sns.jointplot(data_x.loc[:,'perimeter_worst'], data_x.loc[:,'area_worst'], kind="regg", color="#ce1414")
sns.jointplot(data_x.loc[:,'symmetry_worst'], data_x.loc[:,'fractal_dimension_worst'], kind="regg", color="#ce1414")
sns.set(style="white")
df = data_x.loc[:,['radius_worst','perimeter_worst','area_worst']]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_violen)

plt.xticks(rotation=90)
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,0:10]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_violen)
plt.xticks(rotation=90)
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,10:20]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_violen)
plt.xticks(rotation=90)
data_violen = pd.concat([data['diagnosis'],data_standardization.iloc[:,20:31]],axis=1)
data_violen = pd.melt(data_violen,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(20,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_violen)
plt.xticks(rotation=90)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data_x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
m = plt.hist(data[data["diagnosis"] == "M"].radius_mean,bins=30,fc = (1,0,0,0.5),label = "Malignant")
b = plt.hist(data[data["diagnosis"] == "B"].radius_mean,bins=30,fc = (0,1,0,0.5),label = "Bening")
plt.legend()
plt.xlabel("Radius Mean Values")
plt.ylabel("Frequency")
plt.title("Histogram of Radius Mean for Bening and Malignant Tumors")
plt.show()
data_bening = data[data["diagnosis"] == "B"]
data_malignant = data[data["diagnosis"] == "M"]
desc = data_bening.radius_mean.describe()
Q1 = desc[4]
Q3 = desc[6]
IQR = Q3-Q1
lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR
print("Anything outside this range is an outlier: (", lower_bound ,",", upper_bound,")")
data_bening[data_bening.radius_mean < lower_bound].radius_mean
print("Outliers: ",data_bening[(data_bening.radius_mean < lower_bound) | (data_bening.radius_mean > upper_bound)].radius_mean.values)
melted_data = pd.melt(data,id_vars = "diagnosis",value_vars = ['radius_mean', 'texture_mean'])
plt.figure(figsize = (15,10))
sns.boxplot(x = "variable", y = "value", hue="diagnosis",data= melted_data)
plt.show()
#plt.hist(data_bening.radius_mean,bins=50,fc=(0,1,0,0.5),label='Bening',normed = True,cumulative = True)
#sorted_data = np.sort(data_bening.radius_mean)
#y = np.arange(len(sorted_data))/float(len(sorted_data)-1)
#plt.plot(sorted_data,y,color='red')
#plt.title('CDF of bening tumor radius mean')
#plt.show()
mean_diff = data_malignant.radius_mean.mean() - data_bening.radius_mean.mean()
var_bening = data_bening.radius_mean.var()
var_malignant = data_malignant.radius_mean.var()
var_pooled = (len(data_bening)*var_bening +len(data_malignant)*var_malignant ) / float(len(data_bening)+ len(data_malignant))
effect_size = mean_diff/np.sqrt(var_pooled)
print("Effect size: ",effect_size)

plt.figure(figsize = (15,10))
sns.jointplot(data.radius_mean,data.area_mean,kind="regg")
plt.show()
## we can also look at the relationship using seaborn's pais-plot

sns.set(style = "white")
df = data.loc[:,["radius_mean","area_mean","fractal_dimension_se"]]
g = sns.PairGrid(df,diag_sharey = False,)
g.map_lower(sns.kdeplot,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,lw =3)
plt.show()
statistic, p_value = stats.ttest_rel(data.radius_mean,data.area_mean)
print('p-value: ',p_value)

if p_value < 0.05:
    print('Reject the Null Hypothesis')
else:
    print('Except the Null hypothesis')
