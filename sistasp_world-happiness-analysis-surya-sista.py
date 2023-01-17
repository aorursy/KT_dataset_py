# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/2017.csv')
data.info()
data.describe()
data.head()
#histogram
sns.distplot(data['Happiness_Score']);
#skewness and kurtosis
print("Skewness: %f" % data['Happiness_Score'].skew())
print("Kurtosis: %f" % data['Happiness_Score'].kurt())

#histogram
sns.distplot(data['Family']);
#scatter plot Freedom/Happiness Score
var = 'Family'
data = pd.concat([data['Happiness_Score'], data[var]], axis=1)
data.plot.scatter(x=var, y='Happiness_Score', ylim=(0,10));
data=pd.read_csv('../input/2017.csv')
data.info()
#histogram
sns.distplot(data['Freedom']);
#scatter plot Freedom/Happiness Score
var = 'Freedom'
data = pd.concat([data['Happiness_Score'], data[var]], axis=1)
data.plot.scatter(x=var, y='Happiness_Score', ylim=(0,10));
data=pd.read_csv('../input/2017.csv')
data.info()
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#Happiness Score correlation matrix
k = 6 #number of variables for heatmap
cols = corrmat.nlargest(k, 'Happiness_Score')['Happiness_Score'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Let us make this a bit better
#scatterplot
sns.set()
cols = ['Happiness_Score', 'Whisker_low', 'Whisker_high', 'Economy_GDP_per_Capita', 'Health_Life_Expectancy', 'Family']
sns.pairplot(data[cols], size = 2.5)
plt.show();
corr=data.corr()["Happiness_Score"]
corr
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
#Some multivariate analysis
data['Whisker_low'].corr(data['Economy_GDP_per_Capita'])
sns.jointplot(data['Whisker_low'],data['Economy_GDP_per_Capita'],color='gold');